import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vmas import make_env
from scenarios.triangle_fill import Scenario


def parse_args() -> argparse.Namespace:
    # Pre-parse: check for --config before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to YAML config file")
    pre_args, remaining = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config is not None:
        import yaml
        with open(pre_args.config, "r") as f:
            config_defaults = yaml.safe_load(f) or {}
        # Convert YAML keys from kebab-case to underscore (argparse stores as underscore)
        config_defaults = {k.replace("-", "_"): v for k, v in config_defaults.items()}

    p = argparse.ArgumentParser(
        description=(
            "MAPPO-style PPO training for VMAS triangle_fill (shared actor, centralized critic). "
            "All tensors stay on the chosen device; no numpy/CPU buffer conversions."
        )
    )
    p.add_argument("--config", default=None, help="Path to YAML config file")

    # Core
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--n-agents", type=int, default=30)
    # Kilobot-like motion is slow; longer episodes make it easier to correct course and reach the triangle.
    p.add_argument("--max-episode-steps", type=int, default=1000)

    # Training length
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--total-updates", type=int, default=2000)

    # Model size (saved in checkpoints for reproducibility)
    p.add_argument("--actor-hidden", type=int, default=256)
    p.add_argument("--critic-hidden", type=int, default=512)

    # PPO hyperparams
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    # Entropy coefficient (encourages exploration). You can optionally schedule it down over training.
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument(
        "--ent-coef-end",
        type=float,
        default=None,
        help="If set, linearly decay ent_coef from --ent-coef to --ent-coef-end.",
    )
    p.add_argument(
        "--ent-decay-updates",
        type=int,
        default=0,
        help="Number of updates over which to decay entropy (0 disables; if --ent-coef-end is set and this is 0, uses total-updates).",
    )
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    # With T=256, B=64, N=30:
    # - policy samples = 256*64*30 = 491,520
    # - value  samples = 256*64     = 16,384
    # Large minibatches (e.g., 8192) give very few gradient steps per epoch and can slow down escape from plateaus.
    p.add_argument("--minibatch-size-policy", type=int, default=2048)
    p.add_argument("--minibatch-size-value", type=int, default=2048)
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable mixed-precision (AMP). Off by default — PPO actor gradients are sensitive to fp16.",
    )
    p.add_argument(
        "--lr-end-factor",
        type=float,
        default=0.1,
        help="Cosine-decay LR to lr * lr_end_factor over training. Set to 1.0 to disable decay.",
    )

    # Task config
    p.add_argument("--formation-w", type=float, default=1.0)
    p.add_argument("--formation-sinkhorn-tau", type=float, default=0.001)
    p.add_argument("--formation-sinkhorn-iters", type=int, default=100)
    p.add_argument("--formation-eps", type=float, default=1e-8)
    p.add_argument("--formation-template-seed", type=int, default=0)
    p.add_argument("--target-spacing-mm", type=float, default=45.0, help="Target nearest-neighbor distance in mm.")
    p.add_argument("--spacing-w", type=float, default=1.0)
    p.add_argument("--progress-reward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--success-bonus",
        type=float,
        default=0.05,
        help="Per-step settle bonus when formation_loss < success_threshold.",
    )
    p.add_argument("--success-threshold", type=float, default=0.05)
    p.add_argument("--safe-collision-w", type=float, default=0.5)
    p.add_argument("--safe-action-w", type=float, default=0.02)
    # Environment/scenario knobs (for scientific ablations; passed into Scenario.make_world via make_env kwargs).
    # If you don't set these, the scenario defaults are used.
    p.add_argument("--pile-center-x-mm", type=float, default=None)
    p.add_argument("--pile-center-y-mm", type=float, default=None)
    # Domain randomization: sample pile center Y uniformly in [min, max] each episode reset.
    p.add_argument("--pile-center-y-mm-min", type=float, default=None)
    p.add_argument("--pile-center-y-mm-max", type=float, default=None)
    p.add_argument("--pile-halfwidth-mm", type=float, default=None)
    p.add_argument("--turn-v-frac", type=float, default=None)
    p.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override scenario observation normalization (default: scenario setting).",
    )
    p.add_argument(
        "--obs-top-k-neighbors",
        type=int,
        default=None,
        help="Override Top-K neighbor count in observation (default: scenario setting).",
    )
    p.add_argument(
        "--obs-include-goal-rel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include goal-relative observation (position relative to formation center).",
    )

    # Resume
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument(
        "--resume-rng",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore torch RNG states from checkpoint (default: true)",
    )

    # Repro
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic torch settings (slower).",
    )

    # Logging / checkpoint
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--out-dir", default="/home/user/Yihuai/Code/VMAS/runs/triangle_fill")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")

    # Apply config file defaults (CLI args still override)
    if config_defaults:
        p.set_defaults(**config_defaults)

    return p.parse_args()


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """CTDE critic: DeepSets over per-agent state + global scale."""

    def __init__(self, per_agent_dim: int, hidden: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(per_agent_dim, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.Tanh(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, agent_features: torch.Tensor, log_rms: torch.Tensor) -> torch.Tensor:
        phi_out = self.phi(agent_features)
        mean_p = phi_out.mean(dim=-2)
        max_p = phi_out.max(dim=-2).values
        pooled = torch.cat([mean_p, max_p, log_rms.unsqueeze(-1)], dim=-1)
        return self.rho(pooled).squeeze(-1)


def build_critic_input(env_agents, v0: float) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.stack([a.state.pos for a in env_agents], dim=1)
    rot = torch.stack([a.state.rot for a in env_agents], dim=1)
    vel = torch.stack([a.state.vel for a in env_agents], dim=1)

    centroid = pos.mean(dim=1, keepdim=True)
    centered = pos - centroid
    rms = centered.pow(2).sum(dim=-1).mean(dim=-1).sqrt().clamp(min=1e-6)
    pos_norm = centered / rms.unsqueeze(-1).unsqueeze(-1)

    cos_th = torch.cos(rot).squeeze(-1)
    sin_th = torch.sin(rot).squeeze(-1)
    speed = vel.norm(dim=-1) / max(v0, 1e-8)
    agent_features = torch.cat(
        [pos_norm, cos_th.unsqueeze(-1), sin_th.unsqueeze(-1), speed.unsqueeze(-1)],
        dim=-1,
    )
    log_rms = torch.log(rms)
    return agent_features, log_rms


def reset_done_envs(env, obs_list: List[torch.Tensor], dones: torch.Tensor) -> List[torch.Tensor]:
    if not torch.any(dones):
        return obs_list

    done_idx = torch.where(dones)[0].tolist()
    for i in done_idx:
        obs_i = env.reset_at(i)
        for a in range(len(obs_list)):
            obs_list[a][i] = obs_i[a][0]

    return obs_list


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
    trunc_values: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t, b = rewards.shape
    adv = torch.zeros((t, b), device=rewards.device)
    lastgaelam = torch.zeros((b,), device=rewards.device)

    for step in reversed(range(t)):
        nextnonterminal = (~dones[step]).float()
        nextvalues = last_value if step == t - 1 else values[step + 1]
        bootstrap = nextvalues * nextnonterminal
        if trunc_values is not None:
            bootstrap = bootstrap + trunc_values[step]
        delta = rewards[step] + gamma * bootstrap - values[step]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[step] = lastgaelam

    returns = adv + values
    return adv, returns


def metric_keys_for_training() -> List[str]:
    return [
        "formation_loss",
        "spacing_loss",
        "formation_score",
        "collision_mean",
        "action_mean",
        "sinkhorn_entropy",
        "speed_mean",
    ]


def mean_metrics(infos: List[Dict[str, torch.Tensor]], metric_keys: List[str]) -> Dict[str, float]:
    if not infos:
        return {}
    info0 = infos[0]
    out = {}
    for k in metric_keys:
        v = info0.get(k, None)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            out[k] = v.float().mean().item()
    return out


def _checkpoint_payload(
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
    args: argparse.Namespace,
    update: int,
    scaler_actor = None,
    scaler_critic = None,
) -> dict:
    payload = {
        "update": update,
        "args": vars(args),
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "actor_opt_state": actor_opt.state_dict(),
        "critic_opt_state": critic_opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if args.device == "cuda" and torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    # ⭐ AMP: Save scaler states for resume
    if scaler_actor is not None:
        payload["scaler_actor_state"] = scaler_actor.state_dict()
    if scaler_critic is not None:
        payload["scaler_critic_state"] = scaler_critic.state_dict()
    return payload


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)

def _infer_obs_dim_expected(ckpt: dict) -> int | None:
    """Infer actor observation dimension from checkpoint (best-effort)."""
    ckpt_args = ckpt.get("args", {}) or {}
    obs_dim = ckpt_args.get("obs_dim", None)
    if obs_dim is not None:
        try:
            return int(obs_dim)
        except Exception:
            pass
    w = ckpt.get("actor_state", {}).get("net.0.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return None

def _coerce_rng_state(state) -> torch.Tensor:
    """Coerce various serialized RNG state formats into a torch.ByteTensor.

    Why: across PyTorch versions / environments, a saved RNG state may deserialize as:
    - torch.ByteTensor (ideal)
    - torch.Tensor with different dtype
    - list[int] / tuple[int]
    - bytes / bytearray
    We normalize to uint8 tensor so torch.set_rng_state won't crash.
    """
    if state is None:
        raise TypeError("RNG state is None")
    if isinstance(state, torch.Tensor):
        return state.to(dtype=torch.uint8, device="cpu")
    if isinstance(state, (bytes, bytearray)):
        return torch.tensor(list(state), dtype=torch.uint8)
    if isinstance(state, (list, tuple)):
        return torch.tensor(state, dtype=torch.uint8)
    raise TypeError(f"Unsupported RNG state type: {type(state)}")

def _ent_coef_at_update(args: argparse.Namespace, update: int) -> float:
    """Linear entropy decay helper (keeps training code readable and ablation-friendly)."""
    if args.ent_coef_end is None:
        return float(args.ent_coef)
    total = int(args.ent_decay_updates) if int(args.ent_decay_updates) > 0 else int(args.total_updates)
    total = max(total, 1)
    t = min(max(update - 1, 0), total) / float(total)  # 0 -> 1
    return float(args.ent_coef) + t * (float(args.ent_coef_end) - float(args.ent_coef))


def main() -> None:
    args = parse_args()

    if os.environ.get("TMPDIR") is None:
        print("WARNING: TMPDIR not set; if /tmp is unavailable, set TMPDIR=$PWD/.tmp")

    # Seed only matters for the initial RNG state (unless you resume RNG from checkpoint).
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    scenario = Scenario()
    # Build scenario kwargs (only pass explicitly-set values, otherwise scenario defaults apply).
    scenario_kwargs = {
        "n_agents": args.n_agents,
        "formation_w": float(args.formation_w),
        "formation_sinkhorn_tau": float(args.formation_sinkhorn_tau),
        "formation_sinkhorn_iters": int(args.formation_sinkhorn_iters),
        "formation_eps": float(args.formation_eps),
        "formation_template_seed": int(args.formation_template_seed),
        "target_spacing_mm": float(args.target_spacing_mm),
        "spacing_w": float(args.spacing_w),
        "progress_reward": bool(args.progress_reward),
        "success_bonus": float(args.success_bonus),
        "success_threshold": float(args.success_threshold),
        "safe_collision_w": float(args.safe_collision_w),
        "safe_action_w": float(args.safe_action_w),
    }
    if (args.pile_center_y_mm_min is None) ^ (args.pile_center_y_mm_max is None):
        raise ValueError("--pile-center-y-mm-min and --pile-center-y-mm-max must be set together")
    if args.pile_center_y_mm_min is not None and args.pile_center_y_mm_max is not None:
        scenario_kwargs["pile_center_y_mm_range"] = (float(args.pile_center_y_mm_min), float(args.pile_center_y_mm_max))
    if args.pile_center_x_mm is not None or args.pile_center_y_mm is not None:
        scenario_kwargs["pile_center_mm"] = (
            float(args.pile_center_x_mm or 0.0),
            float(args.pile_center_y_mm or 0.0),
        )
    if args.pile_halfwidth_mm is not None:
        scenario_kwargs["pile_halfwidth_mm"] = float(args.pile_halfwidth_mm)
    if args.turn_v_frac is not None:
        scenario_kwargs["turn_v_frac"] = float(args.turn_v_frac)
    if args.normalize_obs is not None:
        scenario_kwargs["normalize_obs"] = bool(args.normalize_obs)
    if args.obs_top_k_neighbors is not None:
        scenario_kwargs["obs_top_k_neighbors"] = int(args.obs_top_k_neighbors)
    if args.obs_include_goal_rel is not None:
        scenario_kwargs["obs_include_goal_rel"] = bool(args.obs_include_goal_rel)

    # If resuming, infer the expected obs_dim and ask the scenario to pad observations to match.
    # This avoids state_dict load failures when the scenario observation vector has evolved.
    ckpt_preview = None
    if args.resume:
        try:
            ckpt_preview = torch.load(Path(args.resume), map_location=args.device)
            obs_dim_expected = _infer_obs_dim_expected(ckpt_preview)
            if obs_dim_expected is not None:
                scenario_kwargs["obs_pad_to_dim"] = int(obs_dim_expected)
        except Exception as e:
            print(f"WARNING: failed to pre-load resume checkpoint for obs_dim inference: {e}")

    env = make_env(
        scenario=scenario,
        num_envs=args.num_envs,
        device=args.device,
        continuous_actions=False,
        dict_spaces=False,
        max_steps=args.max_episode_steps,
        seed=args.seed,
        **scenario_kwargs,
    )

    obs_list = env.reset()
    n_agents = len(env.agents)
    if n_agents != args.n_agents:
        raise RuntimeError(f"env has n_agents={n_agents} but --n-agents={args.n_agents}")

    obs_dim = obs_list[0].shape[-1]
    try:
        setattr(args, "obs_dim", int(obs_dim))
        setattr(args, "obs_mode", "local_bodyframe_topk_v1")
        setattr(args, "obs_top_k_neighbors", int(args.obs_top_k_neighbors) if args.obs_top_k_neighbors is not None else 8)
    except Exception:
        pass
    metric_keys = metric_keys_for_training()

    actor = Actor(obs_dim=obs_dim, hidden=args.actor_hidden).to(env.device)
    critic = Critic(per_agent_dim=5, hidden=args.critic_hidden).to(env.device)
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr_critic)

    actor_sched = optim.lr_scheduler.CosineAnnealingLR(
        actor_opt, T_max=args.total_updates, eta_min=args.lr_actor * args.lr_end_factor,
    )
    critic_sched = optim.lr_scheduler.CosineAnnealingLR(
        critic_opt, T_max=args.total_updates, eta_min=args.lr_critic * args.lr_end_factor,
    )

    is_cuda = (args.device == "cuda" and torch.cuda.is_available())
    use_autocast = args.amp and is_cuda
    use_grad_scaler = args.amp and is_cuda
    scaler_actor = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)
    scaler_critic = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer if enabled
    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir={out_dir / 'tensorboard'}")

    start_update = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = load_checkpoint(ckpt_path, device=env.device)

        actor.load_state_dict(ckpt["actor_state"])
        critic.load_state_dict(ckpt["critic_state"])
        actor_opt.load_state_dict(ckpt["actor_opt_state"])
        critic_opt.load_state_dict(ckpt["critic_opt_state"])

        # ⭐ AMP: Restore scaler states if available (best-effort)
        if use_grad_scaler and "scaler_actor_state" in ckpt:
            try:
                scaler_actor.load_state_dict(ckpt["scaler_actor_state"])
            except Exception as e:
                print(f"WARNING: failed to restore actor scaler state: {e}")
        if use_grad_scaler and "scaler_critic_state" in ckpt:
            try:
                scaler_critic.load_state_dict(ckpt["scaler_critic_state"])
            except Exception as e:
                print(f"WARNING: failed to restore critic scaler state: {e}")

        # Note: obs_dim compatibility is handled by obs_pad_to_dim in the scenario.
        # If you resume from an old checkpoint with different obs_dim, the scenario will
        # automatically pad observations to match the checkpoint's expected dimension.

        if args.resume_rng:
            # RNG restore is best-effort: older/other-version checkpoints may deserialize RNG states
            # into non-ByteTensor types. We coerce when possible; otherwise warn and continue.
            if "torch_rng_state" in ckpt:
                try:
                    torch.set_rng_state(_coerce_rng_state(ckpt["torch_rng_state"]))
                except Exception as e:
                    print(f"WARNING: failed to restore torch RNG state ({type(ckpt.get('torch_rng_state'))}): {e}")

            if args.device == "cuda" and "cuda_rng_state_all" in ckpt and torch.cuda.is_available():
                try:
                    cuda_states = ckpt["cuda_rng_state_all"]
                    if isinstance(cuda_states, (list, tuple)):
                        cuda_states = [_coerce_rng_state(s) for s in cuda_states]
                    torch.cuda.set_rng_state_all(cuda_states)
                except Exception as e:
                    print(f"WARNING: failed to restore CUDA RNG state: {e}")

        start_update = int(ckpt.get("update", 0)) + 1
        print(f"Resumed from {ckpt_path} at update={start_update-1}. Continuing training.")

        # Guardrail: if you resume from a checkpoint that is already beyond --total-updates,
        # training would run zero iterations and would otherwise overwrite ckpt_final/config files.
        if start_update > args.total_updates:
            print(
                f"ERROR: nothing to train. Checkpoint update={start_update-1} but --total-updates={args.total_updates}.\n"
                f"Set --total-updates >= {start_update} (e.g., if you want +4000 updates, use --total-updates {start_update-1 + 4000})."
            )
            return

        # Save a record of resume provenance
        (out_dir / "resume.json").write_text(
            json.dumps({"resume_path": str(ckpt_path), "resume_update": start_update - 1}, indent=2),
            encoding="utf-8",
        )

    # Always write the current run config.
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    T = args.rollout_steps
    B = env.batch_dim

    for update in range(start_update, args.total_updates + 1):
        obs_buf = torch.zeros((T, B, n_agents, obs_dim), device=env.device)
        act_buf = torch.zeros((T, B, n_agents), device=env.device, dtype=torch.long)
        logp_buf = torch.zeros((T, B, n_agents), device=env.device)
        rew_buf = torch.zeros((T, B), device=env.device)
        done_buf = torch.zeros((T, B), device=env.device, dtype=torch.bool)
        val_buf = torch.zeros((T, B), device=env.device)
        trunc_val_buf = torch.zeros((T, B), device=env.device)
        feat_buf = torch.zeros((T, B, n_agents, 5), device=env.device)
        lrms_buf = torch.zeros((T, B), device=env.device)

        # Rolling averages of env-provided diagnostics (Scenario.info()).
        # Keep defaults for missing keys so logging never crashes mid-training.
        metric_acc = {k: 0.0 for k in metric_keys}
        metric_steps = 0

        rollout_start = time.time()
        for t in range(T):
            # ⭐ AMP: Use autocast for inference (but not for env.step)
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_autocast):
                feat_t, lrms_t = build_critic_input(env.agents, scenario.v0)
                feat_buf[t] = feat_t
                lrms_buf[t] = lrms_t
                val_buf[t] = critic(feat_t, lrms_t)

                # ⭐ BATCHED ACTOR FORWARD (2-10x speedup!)
                # Stack all agent observations and forward in one pass instead of N separate calls.
                obs_all = torch.stack(obs_list, dim=1)  # [B, N, obs_dim]
                obs_all_flat = obs_all.reshape(B * n_agents, obs_dim)  # [B*N, obs_dim]

                logits_all = actor(obs_all_flat)  # [B*N, n_actions] - single batched forward!
                dist = torch.distributions.Categorical(logits=logits_all)
                a_flat = dist.sample()  # [B*N]
                logp_flat = dist.log_prob(a_flat)  # [B*N]

                # Reshape back to [B, N]
                a = a_flat.view(B, n_agents)  # [B, N]
                logp = logp_flat.view(B, n_agents)  # [B, N]

                # Store in buffers
                obs_buf[t] = obs_all  # [B, N, obs_dim]
                act_buf[t] = a  # [B, N]
                logp_buf[t] = logp  # [B, N]

                # Environment expects list of [B, 1] tensors (one per agent)
                actions_for_env = [a[:, i].unsqueeze(-1) for i in range(n_agents)]

            obs_list, rews, dones, infos = env.step(actions_for_env)
            rew_buf[t] = rews[0]
            done_buf[t] = dones

            if torch.any(dones):
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_autocast):
                    trunc_feat, trunc_lrms = build_critic_input(env.agents, scenario.v0)
                    trunc_val_buf[t] = critic(trunc_feat, trunc_lrms) * dones.float()

            m = mean_metrics(infos, metric_keys)
            for k in metric_acc.keys():
                metric_acc[k] += float(m.get(k, 0.0))
            metric_steps += 1

            obs_list = reset_done_envs(env, obs_list, dones)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_autocast):
            feat_last, lrms_last = build_critic_input(env.agents, scenario.v0)
            last_value = critic(feat_last, lrms_last)
            adv, ret = compute_gae(rew_buf, done_buf, val_buf, last_value, args.gamma, args.gae_lambda, trunc_values=trunc_val_buf)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten policy batch: (T,B,N)
        obs_flat = obs_buf.reshape(T * B * n_agents, obs_dim)
        act_flat = act_buf.reshape(T * B * n_agents)
        old_logp_flat = logp_buf.reshape(T * B * n_agents)
        adv_flat = adv.unsqueeze(-1).expand(T, B, n_agents).reshape(T * B * n_agents)  # shared-return -> same advantage for all agents

        # Flatten value batch: (T,B)
        global_feat_flat = feat_buf.reshape(T * B, n_agents, 5)
        global_lrms_flat = lrms_buf.reshape(T * B)
        ret_flat = ret.reshape(T * B)
        old_val_flat = val_buf.reshape(T * B)

        batch_policy = obs_flat.shape[0]
        batch_value = global_feat_flat.shape[0]
        mb_policy = min(args.minibatch_size_policy, batch_policy)
        mb_value = min(args.minibatch_size_value, batch_value)

        # Accumulate losses and gradient norms for logging
        acc_policy_loss = 0.0
        acc_entropy = 0.0
        acc_actor_grad_norm = 0.0
        acc_value_loss = 0.0
        acc_critic_grad_norm = 0.0
        n_actor_steps = 0
        n_critic_steps = 0

        for _epoch in range(args.update_epochs):
            ent_coef_now = _ent_coef_at_update(args, update)

            # Actor update
            perm_p = torch.randperm(batch_policy, device=env.device)
            for start_i in range(0, batch_policy, mb_policy):
                mb = perm_p[start_i : start_i + mb_policy]

                # ⭐ AMP: Use autocast for forward pass and loss computation
                with torch.amp.autocast(device_type='cuda', enabled=use_autocast):
                    logits = actor(obs_flat[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(act_flat[mb])
                    entropy = dist.entropy().mean()

                    ratio = (new_logp - old_logp_flat[mb]).exp()
                    pg1 = -adv_flat[mb] * ratio
                    pg2 = -adv_flat[mb] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    policy_loss = torch.max(pg1, pg2).mean()

                    actor_loss = policy_loss - ent_coef_now * entropy

                actor_opt.zero_grad(set_to_none=True)
                # ⭐ AMP: Use scaler for backward pass
                scaler_actor.scale(actor_loss).backward()
                scaler_actor.unscale_(actor_opt)
                # Compute gradient norm before clipping
                actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                scaler_actor.step(actor_opt)
                scaler_actor.update()

                # Accumulate for logging
                acc_policy_loss += policy_loss.item()
                acc_entropy += entropy.item()
                acc_actor_grad_norm += actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else float(actor_grad_norm)
                n_actor_steps += 1

            # Critic update
            perm_v = torch.randperm(batch_value, device=env.device)
            for start_i in range(0, batch_value, mb_value):
                mb = perm_v[start_i : start_i + mb_value]

                # ⭐ AMP: Use autocast for forward pass and loss computation
                with torch.amp.autocast(device_type='cuda', enabled=use_autocast):
                    values = critic(global_feat_flat[mb], global_lrms_flat[mb])

                    v_old = old_val_flat[mb]
                    v_clipped = v_old + torch.clamp(values - v_old, -args.value_clip, args.value_clip)
                    v_loss_unclipped = (values - ret_flat[mb]).pow(2)
                    v_loss_clipped = (v_clipped - ret_flat[mb]).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    critic_loss = args.vf_coef * value_loss

                critic_opt.zero_grad(set_to_none=True)
                # ⭐ AMP: Use scaler for backward pass
                scaler_critic.scale(critic_loss).backward()
                scaler_critic.unscale_(critic_opt)
                # Compute gradient norm before clipping
                critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                scaler_critic.step(critic_opt)
                scaler_critic.update()

                # Accumulate for logging
                acc_value_loss += value_loss.item()
                acc_critic_grad_norm += critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else float(critic_grad_norm)
                n_critic_steps += 1

        actor_sched.step()
        critic_sched.step()

        if update % args.log_every == 0:
            rollout_time = time.time() - rollout_start
            mean_rew = rew_buf.mean().item()
            metric_mean = {k: (v / max(metric_steps, 1)) for k, v in metric_acc.items()}

            # Action histogram across the whole rollout (diagnostic for "all STOP" or other degenerate policies).
            # act_buf: [T,B,N] with discrete ids {0,1,2,3}.
            act_flat_all = act_buf.reshape(-1)
            counts = torch.bincount(act_flat_all, minlength=4).float()
            fracs = (counts / counts.sum().clamp_min(1.0)).tolist()
            print(
                f"update={update:5d} mean_rew={mean_rew:+.4f} rollout_s={rollout_time:.2f} "
                + " ".join([f"{k}={metric_mean.get(k, 0.0):.3f}" for k in metric_keys])
                + f" ent_coef={_ent_coef_at_update(args, update):.4f}"
                + f" act_frac=[STOP {fracs[0]:.2f}, LEFT {fracs[1]:.2f}, RIGHT {fracs[2]:.2f}, STRAIGHT {fracs[3]:.2f}]"
            )

            # TensorBoard logging
            if writer is not None:
                # Core RL metrics
                writer.add_scalar("train/mean_reward", mean_rew, update)
                writer.add_scalar("train/episode_length", T, update)
                
                # Losses
                if n_actor_steps > 0:
                    writer.add_scalar("train/policy_loss", acc_policy_loss / n_actor_steps, update)
                    writer.add_scalar("train/entropy", acc_entropy / n_actor_steps, update)
                    writer.add_scalar("train/actor_grad_norm", acc_actor_grad_norm / n_actor_steps, update)
                if n_critic_steps > 0:
                    writer.add_scalar("train/value_loss", acc_value_loss / n_critic_steps, update)
                    writer.add_scalar("train/critic_grad_norm", acc_critic_grad_norm / n_critic_steps, update)
                
                # Learning rate (from scheduler)
                writer.add_scalar("train/lr_actor", actor_sched.get_last_lr()[0], update)
                writer.add_scalar("train/lr_critic", critic_sched.get_last_lr()[0], update)
                
                # Entropy coefficient
                writer.add_scalar("train/ent_coef", _ent_coef_at_update(args, update), update)
                
                # Environment metrics
                for k, v in metric_mean.items():
                    writer.add_scalar(f"env/{k}", v, update)
                
                # Action distribution
                for i, name in enumerate(["STOP", "LEFT", "RIGHT", "STRAIGHT"]):
                    writer.add_scalar(f"action_frac/{name}", fracs[i], update)
                
                # Training speed
                writer.add_scalar("train/rollout_time_s", rollout_time, update)
                writer.add_scalar("train/steps_per_second", T * B / max(rollout_time, 1e-6), update)

        if args.save_every > 0 and update % args.save_every == 0:
            payload = _checkpoint_payload(actor, critic, actor_opt, critic_opt, args, update, scaler_actor, scaler_critic)
            save_checkpoint(out_dir / f"ckpt_{update:06d}.pt", payload)

    payload = _checkpoint_payload(actor, critic, actor_opt, critic_opt, args, args.total_updates, scaler_actor, scaler_critic)
    save_checkpoint(out_dir / "ckpt_final.pt", payload)

    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {out_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
