import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from vmas import make_env

from VMAS.scenarios.triangle_fill import Scenario


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MAPPO-style PPO training for VMAS triangle_fill (shared actor, centralized critic). "
            "All tensors stay on the chosen device; no numpy/CPU buffer conversions."
        )
    )

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

    # Task config
    p.add_argument(
        "--w-cover",
        type=float,
        default=0.0,
        help="Coverage reward weight. Start at 0.0; increase later to train filling.",
    )
    # Environment/scenario knobs (for scientific ablations; passed into Scenario.make_world via make_env kwargs).
    # If you don't set these, the scenario defaults are used.
    p.add_argument("--pile-center-x-mm", type=float, default=None)
    p.add_argument("--pile-center-y-mm", type=float, default=None)
    # Domain randomization: sample pile center Y uniformly in [min, max] each episode reset.
    p.add_argument("--pile-center-y-mm-min", type=float, default=None)
    p.add_argument("--pile-center-y-mm-max", type=float, default=None)
    p.add_argument("--pile-halfwidth-mm", type=float, default=None)
    p.add_argument("--w-in", type=float, default=None)
    p.add_argument("--w-out", type=float, default=None)
    p.add_argument("--w-action", type=float, default=None)
    p.add_argument("--w-collision", type=float, default=None)
    p.add_argument("--w-depth", type=float, default=None)
    p.add_argument("--depth-scale-mm", type=float, default=None)
    p.add_argument("--turn-v-frac", type=float, default=None)
    p.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override scenario observation normalization (default: scenario setting).",
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
    def __init__(self, global_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


@dataclass
class Rollout:
    obs: torch.Tensor  # [T,B,N,obs_dim]
    actions: torch.Tensor  # [T,B,N]
    logprobs: torch.Tensor  # [T,B,N]
    rewards: torch.Tensor  # [T,B]
    dones: torch.Tensor  # [T,B]
    values: torch.Tensor  # [T,B]


def global_state_from_obs(obs_list: List[torch.Tensor]) -> torch.Tensor:
    """Permutation-invariant global state for the centralized critic.

    Instead of concatenating agents in a fixed order (which is not permutation-invariant),
    we build a pooled representation across agents.

    Returns: [B, 2*obs_dim] = concat(mean_pool, max_pool)
    """
    x = torch.stack(obs_list, dim=1)  # [B, N, obs_dim]
    mean_pool = x.mean(dim=1)  # [B, obs_dim]
    max_pool = x.max(dim=1).values  # [B, obs_dim]
    return torch.cat([mean_pool, max_pool], dim=-1)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    t, b = rewards.shape
    adv = torch.zeros((t, b), device=rewards.device)
    lastgaelam = torch.zeros((b,), device=rewards.device)

    for step in reversed(range(t)):
        nextnonterminal = (~dones[step]).float()
        nextvalues = last_value if step == t - 1 else values[step + 1]
        delta = rewards[step] + gamma * nextvalues * nextnonterminal - values[step]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[step] = lastgaelam

    returns = adv + values
    return adv, returns


def mean_metrics(infos: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    if not infos:
        return {}
    info0 = infos[0]
    out = {}
    for k in ["inside_frac", "outside_mean", "collisions_mean", "cover_error", "speed_mean"]:
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
    scenario_kwargs = {"w_cover": args.w_cover, "n_agents": args.n_agents}
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
    if args.w_in is not None:
        scenario_kwargs["w_in"] = float(args.w_in)
    if args.w_out is not None:
        scenario_kwargs["w_out"] = float(args.w_out)
    if args.w_action is not None:
        scenario_kwargs["w_action"] = float(args.w_action)
    if args.w_collision is not None:
        scenario_kwargs["w_collision"] = float(args.w_collision)
    if args.w_depth is not None:
        scenario_kwargs["w_depth"] = float(args.w_depth)
    if args.depth_scale_mm is not None:
        scenario_kwargs["depth_scale_mm"] = float(args.depth_scale_mm)
    if args.turn_v_frac is not None:
        scenario_kwargs["turn_v_frac"] = float(args.turn_v_frac)
    if args.normalize_obs is not None:
        scenario_kwargs["normalize_obs"] = bool(args.normalize_obs)

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
    # Store resolved obs_dim in args for future evaluation / reproducibility.
    # (Older checkpoints may not have this; evaluate.py can infer from actor_state.)
    try:
        setattr(args, "obs_dim", int(obs_dim))
        # Observation layout metadata (used to make resume safe when new features are appended).
        # We append goal_rel (2 dims) after the original 12-dim base observation.
        setattr(args, "obs_include_goal_rel", True)
        setattr(args, "goal_rel_start", 12)
    except Exception:
        pass
    global_dim = 2 * obs_dim  # mean+max pooled critic input

    actor = Actor(obs_dim=obs_dim, hidden=args.actor_hidden).to(env.device)
    critic = Critic(global_dim=global_dim, hidden=args.critic_hidden).to(env.device)
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr_critic)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_update = 1
    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = load_checkpoint(ckpt_path, device=env.device)

        actor.load_state_dict(ckpt["actor_state"])
        critic.load_state_dict(ckpt["critic_state"])
        actor_opt.load_state_dict(ckpt["actor_opt_state"])
        critic_opt.load_state_dict(ckpt["critic_opt_state"])

        # Resume safety: if the checkpoint was trained before goal_rel existed, those input dims were always zero
        # (padding). Turning them on suddenly would inject random behavior because the corresponding input weights
        # were never trained. We therefore zero those weight columns unless the checkpoint explicitly says it
        # already included goal_rel.
        ckpt_args = ckpt.get("args", {}) or {}
        had_goal_rel = bool(ckpt_args.get("obs_include_goal_rel", False))
        goal_rel_start = int(ckpt_args.get("goal_rel_start", 12))
        if (not had_goal_rel) and obs_dim >= goal_rel_start + 2:
            try:
                # Actor first layer: [hidden, obs_dim]
                actor.net[0].weight.data[:, goal_rel_start : goal_rel_start + 2].zero_()
                # Critic first layer sees [mean_pool, max_pool] concatenated => 2*obs_dim
                critic.net[0].weight.data[:, goal_rel_start : goal_rel_start + 2].zero_()
                critic.net[0].weight.data[:, obs_dim + goal_rel_start : obs_dim + goal_rel_start + 2].zero_()
                print(f"NOTE: resume-compat: zeroed goal_rel input weights at dims [{goal_rel_start}:{goal_rel_start+2}).")
            except Exception as e:
                print(f"WARNING: failed to apply resume-compat goal_rel weight zeroing: {e}")

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
        print(f"Resumed from {ckpt_path} at update={start_update-1}. Continuing with w_cover={args.w_cover}.")

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

    # Always write the current run config (may differ from checkpoint args, e.g., w_cover).
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

        # Rolling averages of env-provided diagnostics (Scenario.info()).
        # Keep defaults for missing keys so logging never crashes mid-training.
        metric_acc = {
            "inside_frac": 0.0,
            "outside_mean": 0.0,
            "collisions_mean": 0.0,
            "cover_error": 0.0,
            "speed_mean": 0.0,
        }
        metric_steps = 0

        rollout_start = time.time()
        for t in range(T):
            with torch.no_grad():
                global_state = global_state_from_obs(obs_list)
                val_buf[t] = critic(global_state)

                actions_for_env = []
                for i in range(n_agents):
                    obs_i = obs_list[i]
                    logits = actor(obs_i)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)

                    obs_buf[t, :, i, :] = obs_i
                    act_buf[t, :, i] = a
                    logp_buf[t, :, i] = logp
                    actions_for_env.append(a.unsqueeze(-1))

            obs_list, rews, dones, infos = env.step(actions_for_env)
            rew_buf[t] = rews[0]
            done_buf[t] = dones

            m = mean_metrics(infos)
            for k in metric_acc.keys():
                metric_acc[k] += float(m.get(k, 0.0))
            metric_steps += 1

            obs_list = reset_done_envs(env, obs_list, dones)

        with torch.no_grad():
            last_value = critic(global_state_from_obs(obs_list))
            adv, ret = compute_gae(rew_buf, done_buf, val_buf, last_value, args.gamma, args.gae_lambda)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten policy batch: (T,B,N)
        obs_flat = obs_buf.reshape(T * B * n_agents, obs_dim)
        act_flat = act_buf.reshape(T * B * n_agents)
        old_logp_flat = logp_buf.reshape(T * B * n_agents)
        adv_flat = adv.unsqueeze(-1).expand(T, B, n_agents).reshape(T * B * n_agents)  # shared-return -> same advantage for all agents

        # Flatten value batch: (T,B)
        # Build permutation-invariant critic inputs from obs_buf by pooling over agents.
        # obs_buf: [T, B, N, obs_dim]
        mean_pool = obs_buf.mean(dim=2)  # [T, B, obs_dim]
        max_pool = obs_buf.max(dim=2).values  # [T, B, obs_dim]
        global_states = torch.cat([mean_pool, max_pool], dim=-1)  # [T, B, 2*obs_dim]
        global_flat = global_states.reshape(T * B, global_dim)
        ret_flat = ret.reshape(T * B)
        old_val_flat = val_buf.reshape(T * B)

        batch_policy = obs_flat.shape[0]
        batch_value = global_flat.shape[0]
        mb_policy = min(args.minibatch_size_policy, batch_policy)
        mb_value = min(args.minibatch_size_value, batch_value)

        for _epoch in range(args.update_epochs):
            ent_coef_now = _ent_coef_at_update(args, update)

            # Actor update
            perm_p = torch.randperm(batch_policy, device=env.device)
            for start_i in range(0, batch_policy, mb_policy):
                mb = perm_p[start_i : start_i + mb_policy]

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
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                actor_opt.step()

            # Critic update
            perm_v = torch.randperm(batch_value, device=env.device)
            for start_i in range(0, batch_value, mb_value):
                mb = perm_v[start_i : start_i + mb_value]
                values = critic(global_flat[mb])

                v_old = old_val_flat[mb]
                v_clipped = v_old + torch.clamp(values - v_old, -args.value_clip, args.value_clip)
                v_loss_unclipped = (values - ret_flat[mb]).pow(2)
                v_loss_clipped = (v_clipped - ret_flat[mb]).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                critic_loss = args.vf_coef * value_loss
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_opt.step()

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
                + " ".join([f"{k}={metric_mean.get(k, 0.0):.3f}" for k in ["inside_frac", "outside_mean", "collisions_mean", "cover_error", "speed_mean"]])
                + f" ent_coef={_ent_coef_at_update(args, update):.4f}"
                + f" act_frac=[STOP {fracs[0]:.2f}, LEFT {fracs[1]:.2f}, RIGHT {fracs[2]:.2f}, STRAIGHT {fracs[3]:.2f}]"
            )

        if args.save_every > 0 and update % args.save_every == 0:
            payload = _checkpoint_payload(actor, critic, actor_opt, critic_opt, args, update)
            save_checkpoint(out_dir / f"ckpt_{update:06d}.pt", payload)

    payload = _checkpoint_payload(actor, critic, actor_opt, critic_opt, args, args.total_updates)
    save_checkpoint(out_dir / "ckpt_final.pt", payload)


if __name__ == "__main__":
    main()
