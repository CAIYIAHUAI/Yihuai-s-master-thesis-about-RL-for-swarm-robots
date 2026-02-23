import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vmas import make_env

try:
    # Run-friendly import when executing from the repo root.
    # 从仓库根目录直接运行脚本时使用的导入方式。
    from scenarios.triangle_fill import Scenario
except ImportError:  # pragma: no cover
    # Package-style import fallback.
    # 包形式导入的兜底写法。
    from VMAS.scenarios.triangle_fill import Scenario


class Actor(nn.Module):
    """Must match train.py/evaluate.py Actor exactly (so state_dict loads)."""

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


def _infer_obs_dim_expected(ckpt: Dict[str, Any]) -> Optional[int]:
    ckpt_args = ckpt.get("args", {}) or {}
    obs_dim = ckpt_args.get("obs_dim", None)
    if obs_dim is not None:
        return int(obs_dim)
    w = ckpt.get("actor_state", {}).get("net.0.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return None


def _scenario_kwargs_from_ckpt(ckpt: Dict[str, Any], overrides: argparse.Namespace) -> Dict[str, Any]:
    """Reconstruct scenario kwargs so visualization matches training as closely as possible."""

    ckpt_args = ckpt.get("args", {}) or {}
    n_agents = int(ckpt_args.get("n_agents", 30))

    scenario_kwargs: Dict[str, Any] = {"n_agents": n_agents}
    for key in [
        "pile_center_mm",
        "pile_center_y_mm_range",
        "pile_halfwidth_mm",
        "obs_top_k_neighbors",
        "turn_v_frac",
        "normalize_obs",
        "formation_w",
        "formation_sinkhorn_tau",
        "formation_sinkhorn_iters",
        "formation_eps",
        "formation_template_seed",
        "safe_collision_w",
        "safe_action_w",
    ]:
        v = ckpt_args.get(key, None)
        if v is not None:
            scenario_kwargs[key] = v

    # Backward compat: some runs store pile_center as CLI args instead of the derived tuple.
    if "pile_center_mm" not in scenario_kwargs:
        pcx = ckpt_args.get("pile_center_x_mm", None)
        pcy = ckpt_args.get("pile_center_y_mm", None)
        if pcx is not None or pcy is not None:
            scenario_kwargs["pile_center_mm"] = (float(pcx or 0.0), float(pcy or 0.0))
    if "pile_halfwidth_mm" not in scenario_kwargs:
        phw = ckpt_args.get("pile_halfwidth_mm", None)
        if phw is not None:
            scenario_kwargs["pile_halfwidth_mm"] = float(phw)

    # Backward compat: some runs store the Y-range as CLI args instead of the derived tuple.
    y0 = ckpt_args.get("pile_center_y_mm_min", None)
    y1 = ckpt_args.get("pile_center_y_mm_max", None)
    if y0 is not None and y1 is not None:
        scenario_kwargs["pile_center_y_mm_range"] = (float(y0), float(y1))

    # CLI overrides (useful for quick debugging).
    if (getattr(overrides, "pile_center_y_mm_min", None) is None) ^ (getattr(overrides, "pile_center_y_mm_max", None) is None):
        raise ValueError("--pile-center-y-mm-min and --pile-center-y-mm-max must be set together")
    if getattr(overrides, "pile_center_y_mm_min", None) is not None and getattr(overrides, "pile_center_y_mm_max", None) is not None:
        scenario_kwargs["pile_center_y_mm_range"] = (float(overrides.pile_center_y_mm_min), float(overrides.pile_center_y_mm_max))

    if overrides.pile_center_x_mm is not None or overrides.pile_center_y_mm is not None:
        scenario_kwargs["pile_center_mm"] = (float(overrides.pile_center_x_mm or 0.0), float(overrides.pile_center_y_mm or 0.0))
    if overrides.pile_halfwidth_mm is not None:
        scenario_kwargs["pile_halfwidth_mm"] = float(overrides.pile_halfwidth_mm)
    if overrides.normalize_obs is not None:
        scenario_kwargs["normalize_obs"] = bool(overrides.normalize_obs)
    if overrides.formation_w is not None:
        scenario_kwargs["formation_w"] = float(overrides.formation_w)
    if overrides.formation_sinkhorn_tau is not None:
        scenario_kwargs["formation_sinkhorn_tau"] = float(overrides.formation_sinkhorn_tau)
    if overrides.formation_sinkhorn_iters is not None:
        scenario_kwargs["formation_sinkhorn_iters"] = int(overrides.formation_sinkhorn_iters)
    if overrides.formation_eps is not None:
        scenario_kwargs["formation_eps"] = float(overrides.formation_eps)
    if overrides.formation_template_seed is not None:
        scenario_kwargs["formation_template_seed"] = int(overrides.formation_template_seed)
    if overrides.safe_collision_w is not None:
        scenario_kwargs["safe_collision_w"] = float(overrides.safe_collision_w)
    if overrides.safe_action_w is not None:
        scenario_kwargs["safe_action_w"] = float(overrides.safe_action_w)

    obs_dim_expected = _infer_obs_dim_expected(ckpt)
    if obs_dim_expected is not None:
        # Make scenario pad observations if needed so the Actor can run.
        scenario_kwargs["obs_pad_to_dim"] = int(obs_dim_expected)

    return scenario_kwargs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive visualization for a trained triangle_fill policy (runs until you stop it).")
    p.add_argument("--ckpt", default="/home/user/Yihuai/Code/VMAS/runs/triangle_fill/ckpt_final.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Use cpu for rendering (recommended).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-episode-steps", type=int, default=None, help="If None, uses training max_episode_steps (or 1000).")
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True, help="Sleep to match sim dt.")
    p.add_argument("--print-every", type=int, default=0, help="Print metrics every N steps (0 disables).")

    # Optional overrides (mainly for debugging).
    p.add_argument("--pile-center-x-mm", type=float, default=None)
    p.add_argument("--pile-center-y-mm", type=float, default=None)
    p.add_argument("--pile-center-y-mm-min", type=float, default=None)
    p.add_argument("--pile-center-y-mm-max", type=float, default=None)
    p.add_argument("--pile-halfwidth-mm", type=float, default=None)
    p.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--formation-w", type=float, default=None)
    p.add_argument("--formation-sinkhorn-tau", type=float, default=None)
    p.add_argument("--formation-sinkhorn-iters", type=int, default=None)
    p.add_argument("--formation-eps", type=float, default=None)
    p.add_argument("--formation-template-seed", type=int, default=None)
    p.add_argument("--safe-collision-w", type=float, default=None)
    p.add_argument("--safe-action-w", type=float, default=None)
    p.add_argument("--debug-scenario", action="store_true", help="Print scenario kwargs and resolved values.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    ckpt_args = ckpt.get("args", {}) or {}
    actor_hidden = int(ckpt_args.get("actor_hidden", 256))
    n_agents = int(ckpt_args.get("n_agents", 30))

    if args.max_episode_steps is None:
        args.max_episode_steps = int(ckpt_args.get("max_episode_steps", 1000))

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    scenario_kwargs = _scenario_kwargs_from_ckpt(ckpt, args)

    env = make_env(
        scenario=Scenario(),
        num_envs=1,
        device=args.device,
        continuous_actions=False,
        dict_spaces=False,
        max_steps=args.max_episode_steps,
        seed=args.seed,
        **scenario_kwargs,
    )

    obs_list = env.reset()
    obs_dim = int(obs_list[0].shape[-1])

    actor = Actor(obs_dim=obs_dim, hidden=actor_hidden).to(env.device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

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
                "obs_dim": int(obs_dim),
                "task_mode": "formation",
                "deterministic": bool(args.deterministic),
                "max_episode_steps": int(args.max_episode_steps),
            }
        print("DEBUG scenario_resolved:", json.dumps(resolved, indent=2, default=str))
        sys.stdout.flush()

    dt = float(getattr(env.scenario.world, "dt", 0.0) or 0.0)
    step = 0

    print("Running... close the window or press Ctrl+C to stop.")
    try:
        while True:
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

            # Render continuously.
            _ = env.render(
                mode="rgb_array",
                env_index=0,
                agent_index_focus=None,
                visualize_when_rgb=True,
            )

            # Keep running across episode boundaries.
            if bool(dones.item()):
                obs_list = env.reset()

            step += 1
            if args.print_every and step % int(args.print_every) == 0:
                info0 = infos[0] if infos else {}
                parts = [f"step={step:6d}"]
                for key in ("formation_score", "formation_loss", "collision_mean", "action_mean"):
                    v = info0.get(key, None)
                    if isinstance(v, torch.Tensor) and v.numel() > 0:
                        parts.append(f"{key}={v.float().mean().item():.3f}")
                print(" ".join(parts))

            if args.realtime and dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
