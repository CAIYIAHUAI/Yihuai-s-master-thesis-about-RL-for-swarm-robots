import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vmas import make_env
from vmas.simulator.utils import save_video

try:
    # Run-friendly import when executing from the repo root.
    # 从仓库根目录直接运行脚本时使用的导入方式。
    from scenarios.triangle_fill import Scenario
except ImportError:  # pragma: no cover
    # Package-style import fallback.
    # 包形式导入的兜底写法。
    from VMAS.scenarios.triangle_fill import Scenario


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained triangle_fill policy (fixed seeds, mean/variance metrics).")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (from train.py)")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=128)
    # Default None so we can fall back to the training config stored in the checkpoint for reproducibility.
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use argmax actions instead of sampling (default: true).",
    )

    # Video (optional)
    p.add_argument("--render", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-path", default=None)
    p.add_argument("--video-env-index", type=int, default=0)

    p.add_argument("--out-json", default=None, help="Optional path to write evaluation summary JSON")
    p.add_argument(
        "--metric-mode",
        choices=["last", "mean", "max"],
        default="mean",
        help="How to aggregate per-step metrics over an episode. 'mean' matches training logs; 'last' is final-state; 'max' is best-so-far.",
    )
    p.add_argument(
        "--debug-scenario",
        action="store_true",
        help="Print scenario kwargs and resolved runtime values (normalize_obs, world_semidim, turn_v_frac, etc.).",
    )

    # Optional overrides (by default we *reuse the training config stored in the checkpoint*).
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
        help="Override scenario observation normalization (default: checkpoint/scenario setting).",
    )
    p.add_argument("--formation-w", type=float, default=None)
    p.add_argument("--spacing-w", type=float, default=None)
    p.add_argument("--lattice-w", type=float, default=None)
    p.add_argument("--lattice-k", type=int, default=None)
    p.add_argument("--formation-sinkhorn-tau", type=float, default=None)
    p.add_argument("--formation-sinkhorn-iters", type=int, default=None)
    p.add_argument("--formation-eps", type=float, default=None)
    p.add_argument("--formation-template-seed", type=int, default=None)
    p.add_argument("--progress-reward", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--per-agent-reward", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--safe-collision-w", type=float, default=None)
    p.add_argument("--safe-action-w", type=float, default=None)
    p.add_argument(
        "--fast-collisions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the runtime VMAS sphere collision fast path.",
    )
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


def metric_keys_for_eval() -> List[str]:
    return [
        "formation_loss",
        "spacing_loss",
        "lattice_loss",
        "local_spacing_progress_mean",
        "local_lattice_progress_mean",
        "global_shape_progress_mean",
        "formation_score",
        "collision_mean",
        "action_mean",
        "sinkhorn_entropy",
        "speed_mean",
    ]


def _stack_metric(info0: dict, key: str) -> Optional[torch.Tensor]:
    v = info0.get(key, None)
    if isinstance(v, torch.Tensor):
        return v.float().detach().clone()
    return None


def summarize(values: torch.Tensor) -> Dict[str, float]:
    # values: [E]
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=True).item() if values.numel() > 1 else 0.0,
        "min": values.min().item(),
        "max": values.max().item(),
    }


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=args.device)

    ckpt_args = ckpt.get("args", {})
    n_agents = int(ckpt_args.get("n_agents", 30))
    actor_hidden = int(ckpt_args.get("actor_hidden", 256))
    # Prefer obs_dim stored in checkpoint args; otherwise infer from the first linear layer.
    obs_dim_expected = ckpt_args.get("obs_dim", None)
    if obs_dim_expected is None:
        w = ckpt.get("actor_state", {}).get("net.0.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            obs_dim_expected = int(w.shape[1])

    # Keep evaluation horizon consistent with training unless explicitly overridden.
    if args.max_episode_steps is None:
        args.max_episode_steps = int(ckpt_args.get("max_episode_steps", 500))

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Reuse training scenario config from checkpoint by default (for scientific reproducibility).
    # You can override any field via CLI flags for domain-shift evaluation.
    scenario_kwargs = {"n_agents": n_agents}
    for key in [
        "pile_center_mm",
        "pile_center_y_mm_range",
        "pile_halfwidth_mm",
        "obs_top_k_neighbors",
        "turn_v_frac",
        "normalize_obs",
        "formation_w",
        "spacing_w",
        "lattice_w",
        "lattice_k",
        "formation_sinkhorn_tau",
        "formation_sinkhorn_iters",
        "formation_eps",
        "formation_template_seed",
        "progress_reward",
        "share_reward",
        "safe_collision_w",
        "safe_action_w",
        "fast_collisions",
    ]:
        v = ckpt_args.get(key, None)
        if v is not None:
            scenario_kwargs[key] = v

    # Backward-compat: older checkpoints store pile_center as CLI args (pile_center_x_mm / pile_center_y_mm)
    # rather than the derived tuple (pile_center_mm). Reconstruct it so evaluation matches training.
    if "pile_center_mm" not in scenario_kwargs:
        pcx = ckpt_args.get("pile_center_x_mm", None)
        pcy = ckpt_args.get("pile_center_y_mm", None)
        if pcx is not None or pcy is not None:
            scenario_kwargs["pile_center_mm"] = (float(pcx or 0.0), float(pcy or 0.0))
    if "pile_halfwidth_mm" not in scenario_kwargs:
        phw = ckpt_args.get("pile_halfwidth_mm", None)
        if phw is not None:
            scenario_kwargs["pile_halfwidth_mm"] = float(phw)
    if "turn_v_frac" not in scenario_kwargs:
        tvf = ckpt_args.get("turn_v_frac", None)
        if tvf is not None:
            scenario_kwargs["turn_v_frac"] = float(tvf)
    if "share_reward" not in scenario_kwargs:
        per_agent_reward = ckpt_args.get("per_agent_reward", None)
        if per_agent_reward is not None:
            scenario_kwargs["share_reward"] = not bool(per_agent_reward)
    # Backward-compat: some runs store the Y-range as CLI args instead of the derived tuple.
    if "pile_center_y_mm_range" not in scenario_kwargs:
        y0 = ckpt_args.get("pile_center_y_mm_min", None)
        y1 = ckpt_args.get("pile_center_y_mm_max", None)
        if y0 is not None and y1 is not None:
            scenario_kwargs["pile_center_y_mm_range"] = (float(y0), float(y1))

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
    if args.formation_w is not None:
        scenario_kwargs["formation_w"] = float(args.formation_w)
    if args.spacing_w is not None:
        scenario_kwargs["spacing_w"] = float(args.spacing_w)
    if args.lattice_w is not None:
        scenario_kwargs["lattice_w"] = float(args.lattice_w)
    if args.lattice_k is not None:
        scenario_kwargs["lattice_k"] = int(args.lattice_k)
    if args.formation_sinkhorn_tau is not None:
        scenario_kwargs["formation_sinkhorn_tau"] = float(args.formation_sinkhorn_tau)
    if args.formation_sinkhorn_iters is not None:
        scenario_kwargs["formation_sinkhorn_iters"] = int(args.formation_sinkhorn_iters)
    if args.formation_eps is not None:
        scenario_kwargs["formation_eps"] = float(args.formation_eps)
    if args.formation_template_seed is not None:
        scenario_kwargs["formation_template_seed"] = int(args.formation_template_seed)
    if args.progress_reward is not None:
        scenario_kwargs["progress_reward"] = bool(args.progress_reward)
    if args.per_agent_reward is not None:
        scenario_kwargs["share_reward"] = not bool(args.per_agent_reward)
    if args.safe_collision_w is not None:
        scenario_kwargs["safe_collision_w"] = float(args.safe_collision_w)
    if args.safe_action_w is not None:
        scenario_kwargs["safe_action_w"] = float(args.safe_action_w)
    if args.fast_collisions is not None:
        scenario_kwargs["fast_collisions"] = bool(args.fast_collisions)

    # Backward-compat: if the checkpoint expects a larger obs_dim than the current scenario outputs,
    # ask the scenario to pad observations with zeros so the Actor can run.
    if obs_dim_expected is not None:
        scenario_kwargs["obs_pad_to_dim"] = int(obs_dim_expected)

    metric_keys = metric_keys_for_eval()

    env = make_env(
        scenario=Scenario(),
        num_envs=min(args.num_envs, args.episodes),
        device=args.device,
        continuous_actions=False,
        dict_spaces=False,
        max_steps=args.max_episode_steps,
        seed=args.seed,
        **scenario_kwargs,
    )

    obs_list = env.reset()
    obs_dim = obs_list[0].shape[-1]

    if args.debug_scenario:
        print("DEBUG scenario_kwargs:", json.dumps(scenario_kwargs, indent=2, default=str))
        s = env.scenario
        resolved = {
            "normalize_obs": getattr(s, "normalize_obs", None),
            "world_semidim": float(getattr(s, "world_semidim", float("nan"))),
            "comm_r": float(getattr(s, "comm_r", float("nan"))),
            "v0": float(getattr(s, "v0", float("nan"))),
            "turn_v_frac": getattr(s, "turn_v_frac", None),
            "pile_center_mm": getattr(s, "pile_center_mm", None),
            "pile_halfwidth_mm": getattr(s, "pile_halfwidth_mm", None),
            "obs_dim_expected": obs_dim_expected,
            "obs_dim_runtime": int(obs_dim),
            "task_mode": "formation",
            "metric_mode": args.metric_mode,
            "deterministic": args.deterministic,
            "max_episode_steps": int(args.max_episode_steps),
        }
        print("DEBUG scenario_resolved:", json.dumps(resolved, indent=2, default=str))
        sys.stdout.flush()

    # Build the actor with the expected obs_dim (after optional scenario padding).
    actor = Actor(obs_dim=int(obs_dim), hidden=actor_hidden).to(env.device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    episodes_done = 0
    metric_values: Dict[str, List[torch.Tensor]] = {k: [] for k in metric_keys}
    reward_all: List[torch.Tensor] = []

    frames: Optional[List] = [] if (args.render and args.save_video) else None

    while episodes_done < args.episodes:
        batch_envs = env.batch_dim
        remaining = args.episodes - episodes_done
        active = min(batch_envs, remaining)

        obs_list = env.reset()
        ep_return = torch.zeros((batch_envs,), device=env.device)

        # Aggregate per-step metrics to avoid the "last frame only" pitfall and match training logs when needed.
        last = {k: torch.zeros((batch_envs,), device=env.device) for k in metric_keys}
        maxv = {k: torch.full((batch_envs,), -float("inf"), device=env.device) for k in metric_keys}
        sumv = {k: torch.zeros((batch_envs,), device=env.device) for k in metric_keys}
        steps = 0

        for _t in range(args.max_episode_steps):
            actions_for_env = []
            with torch.no_grad():
                for i in range(n_agents):
                    logits = actor(obs_list[i])
                    if args.deterministic:
                        a = torch.argmax(logits, dim=-1)
                    else:
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                    actions_for_env.append(a.unsqueeze(-1))

            obs_list, rews, dones, infos = env.step(actions_for_env)
            ep_return += rews[0]
            info0 = infos[0]
            steps += 1

            for k in metric_keys:
                v = _stack_metric(info0, k)
                if v is None:
                    v = torch.zeros((batch_envs,), device=env.device)
                last[k] = v
                maxv[k] = torch.maximum(maxv[k], v)
                sumv[k] = sumv[k] + v

            if args.render:
                _ = env.render(
                    mode="rgb_array",
                    env_index=args.video_env_index,
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
                if frames is not None:
                    frames.append(_)

        if steps <= 0:
            raise RuntimeError("No steps collected during evaluation")

        if args.metric_mode == "last":
            final = last
        elif args.metric_mode == "max":
            final = maxv
        else:
            final = {k: (sumv[k] / float(steps)) for k in metric_keys}

        for k in metric_keys:
            metric_values[k].append(final[k][:active].cpu())
        reward_all.append(ep_return[:active].cpu())

        episodes_done += active

    aggregated_metrics = {k: torch.cat(v, dim=0) for k, v in metric_values.items()}
    ep_ret = torch.cat(reward_all, dim=0)

    summary = {
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "task_mode": "formation",
        "metric_mode": str(args.metric_mode),
        "checkpoint": str(ckpt_path),
        "episode_return": summarize(ep_ret),
    }
    for k in metric_keys:
        summary[k] = summarize(aggregated_metrics[k])

    print(json.dumps(summary, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if frames is not None and len(frames) > 0:
        fps = 1.0 / env.scenario.world.dt
        if args.video_path is None:
            base = "triangle_fill_eval"
        else:
            base = os.path.splitext(args.video_path)[0]
        save_video(base, frames, fps=int(round(fps)))
        print("saved video:", base + ".mp4")


if __name__ == "__main__":
    main()
