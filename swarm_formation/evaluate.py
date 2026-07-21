"""Multi-seed multi-shape eval of a trained actor.

For each shape and each seed, runs n_envs envs for T steps with the actor
forcing this shape (override env.slot_pair_dist and env.target_map).
Reports per-shape mean ± std across seeds, plus all-shape aggregate.

Determinism: this script enables PyTorch deterministic algorithms so that
two runs with the same seed produce identical numbers, suppressing the
~2% drift introduced by non-deterministic GPU scatter kernels in GATv2Conv.
Expect 2-3x slower eval; the per-cell mean shifts <0.01 vs the legacy path.

Usage (from the repository root):
    python swarm_formation/evaluate.py --ckpt path/to/best.pt
    python swarm_formation/evaluate.py --ckpt path/to/best.pt --n_seeds 5 --n_envs 64 --T 300
    python swarm_formation/evaluate.py --controller --config configs/n21_all_shapes.yaml --shapes t6 hex21 ring21 row7x3
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Required by torch.use_deterministic_algorithms when CUDA matmul is in play.
# Must be set before any cuBLAS context is created (i.e., before torch import).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from expert_controller import compute_controller_action
from environment import MeanShiftEnv, load_config
from shape_metrics import pairwise_dist, per_agent_match_distance_iterated
from templates import SHAPES, make_template, make_target_map
from train_mappo import _build_actor, actor_dist, build_selfshape_obs


def _force_shape(eval_env: MeanShiftEnv, eval_cfg: dict, shape_name: str,
                 spacing: float, n_envs: int, device: str) -> torch.Tensor:
    tmpl = make_template(shape_name, spacing, device=device)
    pair_dist = pairwise_dist(tmpl).to(device=device)
    if eval_env.slot_pair_dist.ndim == 3:
        eval_env.slot_pair_dist[:] = pair_dist.unsqueeze(0).expand(n_envs, -1, -1)
    else:
        eval_env.slot_pair_dist = pair_dist
    if shape_name in eval_env._shape_names:
        eval_env.shape_id[:] = eval_env._shape_names.index(shape_name)
    eval_env.target_map[:] = make_target_map(
        shape_name, eval_env.positions, spacing=spacing,
        theta=float(eval_cfg.get("target_theta", 0.0)),
        center_mode=str(eval_cfg.get("target_center_mode", "initial_swarm_centroid")),
    )
    return tmpl


def _final_metrics(eval_env: MeanShiftEnv, eval_cfg: dict, tmpl: torch.Tensor,
                   spacing: float) -> dict:
    threshold_loose = float(eval_cfg.get("success_shape_distance", 0.4)) * spacing
    threshold_strict = float(eval_cfg.get("success_shape_distance_strict", 0.25)) * spacing

    prof, match_d, iso, bp = eval_env.compute_shape_metrics(
        tmpl, compute_match_distance=True, theta_bins=72,
    )
    per_agent_match_d = per_agent_match_distance_iterated(eval_env.positions, tmpl)
    succ_loose = (match_d < threshold_loose).float()
    succ_strict = (match_d < threshold_strict).float()
    agent_match_strict = (per_agent_match_d < threshold_strict).float().mean()
    coll = float(eval_env._cum_collision.float().mean().item())
    with torch.no_grad():
        stress_env, _ = eval_env.compute_stress_reward()
    return {
        "succ_loose":      succ_loose.mean().item(),
        "succ_strict":     succ_strict.mean().item(),
        "agent_match":     float(agent_match_strict.item()),
        "match_d":         float((match_d.mean() / spacing).item()),
        "match_d_max":     float((match_d.max()  / spacing).item()),
        "stress":          float(stress_env.mean().item()),
        "coll":            coll,
        "coll_per_agent":  coll / max(int(eval_env.N), 1),
        "bond":            float(bp.mean().item()),
    }


def eval_shape(actor, cfg: dict, shape_name: str, n_envs: int, T: int,
                device: str, seed: int) -> dict:
    """Force all envs to one shape, run rollout, return aggregate metrics."""
    eval_cfg = dict(cfg)
    eval_cfg["episode_length"] = T
    eval_env = MeanShiftEnv(eval_cfg, n_envs=n_envs, device=device, seed=seed)
    spacing = float(eval_cfg["target_spacing"])

    eval_env.reset_all()
    eval_env.set_cue_present_prob(0.0)

    tmpl = _force_shape(eval_env, eval_cfg, shape_name, spacing, n_envs, device)

    noise_gen = torch.Generator(device=device); noise_gen.manual_seed(seed + 1)
    h = torch.zeros(n_envs, eval_env.N, actor.hidden, device=device)
    with torch.no_grad():
        for _t in range(T):
            exp_out = eval_env.expert_action(noise_gen=noise_gen, build_obs=False)
            obs, ei, ea = build_selfshape_obs(eval_env, exp_out,
                                               cue_present=eval_env.cue_present,
                                               latent_z=eval_env.latent_z,
                                               heading_locked=True)
            logits, vel_mu, h = actor(obs, ei, ea, h)
            action = torch.tanh(vel_mu).clamp(-1.0, 1.0)         # deterministic mu
            eval_env.step(action)

    return _final_metrics(eval_env, eval_cfg, tmpl, spacing)


def eval_controller_shape(cfg: dict, shape_name: str, n_envs: int, T: int,
                          device: str, seed: int) -> dict:
    """Force one shape and evaluate the hand-coded local spring controller."""
    eval_cfg = dict(cfg)
    eval_cfg["episode_length"] = T
    eval_env = MeanShiftEnv(eval_cfg, n_envs=n_envs, device=device, seed=seed)
    spacing = float(eval_cfg["target_spacing"])

    eval_env.reset_all()
    eval_env.set_cue_present_prob(0.0)
    tmpl = _force_shape(eval_env, eval_cfg, shape_name, spacing, n_envs, device)

    comm_radius = float(eval_cfg["comm_radius"])
    k_spring = float(eval_cfg.get("controller_k_spring", 1.0))
    linear_max_speed = float(eval_cfg.get("controller_linear_max_speed_d", 0.05)) * spacing
    with torch.no_grad():
        for _t in range(T):
            action = compute_controller_action(
                eval_env, eval_env.slot_pair_dist,
                comm_radius=comm_radius,
                k_spring=k_spring,
                linear_max_speed=linear_max_speed,
            )
            eval_env.step(action)

    return _final_metrics(eval_env, eval_cfg, tmpl, spacing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--controller", action="store_true",
                   help="Evaluate the hand-coded spring controller instead of a checkpoint.")
    p.add_argument("--config", type=Path, default=None,
                   help="Config to use with --controller.")
    p.add_argument("--shapes", nargs="+", default=["t4", "hex4", "ring10", "row5x2"])
    p.add_argument("--actor_type", choices=["gat", "meanagg", "tolstaya_edge", "tolstaya_pure"],
                   default=None, help="Override actor architecture; defaults to checkpoint config.")
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seed_start", type=int, default=100)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.controller:
        if args.config is None:
            p.error("--controller requires --config")
        cfg = load_config(args.config)
        cfg.pop("arch_version", None)
        actor = None
        print(f"[multi-seed-eval] controller config={args.config} shapes={args.shapes}")
    else:
        if args.ckpt is None:
            p.error("--ckpt is required unless --controller is used")
        print(f"[multi-seed-eval] ckpt={args.ckpt} shapes={args.shapes}")
        state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        cfg = dict(state["config"])
        cfg.pop("arch_version", None)
        ckpt_n_agents = int(state.get("n_agents", cfg.get("n_agents", 10)))
        cfg.setdefault("n_agents", ckpt_n_agents)
        if args.actor_type is not None:
            cfg["actor_type"] = args.actor_type
        else:
            cfg.setdefault("actor_type", state.get("actor_type", "gat"))
        actor = _build_actor(cfg, args.device, selfshape=True)
        actor.load_state_dict(state["actor_state"], strict=False)
        actor.eval()
    print(f"[multi-seed-eval] {args.n_seeds} seeds x {args.n_envs} envs x {args.T} steps "
          f"({'controller' if args.controller else 'deterministic mu'}, cue=off)")

    for shape in args.shapes:
        n_shape = make_template(shape, float(cfg["target_spacing"]), device="cpu").shape[0]
        if n_shape != int(cfg["n_agents"]):
            raise SystemExit(
                f"shape {shape!r} has {n_shape} points but checkpoint/config n_agents="
                f"{cfg['n_agents']}"
            )

    per_shape_seed: dict[str, list[dict]] = {s: [] for s in args.shapes}
    for k, shape in enumerate(args.shapes):
        for s in range(args.n_seeds):
            seed = args.seed_start + s * 1000 + k
            if args.controller:
                m = eval_controller_shape(cfg, shape, args.n_envs, args.T,
                                          args.device, seed)
            else:
                m = eval_shape(actor, cfg, shape, args.n_envs, args.T,
                               args.device, seed)
            per_shape_seed[shape].append(m)
            print(f"  [{shape}] seed={seed}  "
                  f"succ@0.4d={m['succ_loose']:.3f}  succ@0.25d={m['succ_strict']:.3f}  "
                  f"agent_match={m['agent_match']:.3f}  match={m['match_d']:.3f}d  "
                  f"max={m['match_d_max']:.3f}d  coll={m['coll']:.1f}  "
                  f"coll/N={m['coll_per_agent']:.3f}  stress={m['stress']:+.4f}")

    def mean_std(vals: list[float]) -> tuple[float, float]:
        n = len(vals)
        mu = sum(vals) / n
        var = sum((x - mu) ** 2 for x in vals) / max(n - 1, 1)
        return mu, math.sqrt(var)

    print("\n=== Per-shape mean ± std ===")
    print(f"{'shape':<8} {'succ@0.4d':>14} {'succ@0.25d':>15} {'agent_match':>15} "
          f"{'match (d)':>14} {'max (d)':>12} {'coll':>9} {'coll/N':>9} {'stress':>10}")
    metric_keys = [
        "succ_loose", "succ_strict", "agent_match", "match_d", "match_d_max",
        "coll", "coll_per_agent", "stress",
    ]
    agg: dict[str, list[float]] = {k: [] for k in metric_keys}
    for shape in args.shapes:
        seeds = per_shape_seed[shape]
        line = f"{shape:<8}"
        for k in metric_keys:
            mu, std = mean_std([m[k] for m in seeds])
            agg[k].append(mu)
            line += f"  {mu:.3f}±{std:.3f}"
        print(line)

    print("\n=== All-shape aggregate (mean across shapes of per-shape means) ===")
    for k in metric_keys:
        mu, std = mean_std(agg[k])
        print(f"  {k:<14} {mu:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
