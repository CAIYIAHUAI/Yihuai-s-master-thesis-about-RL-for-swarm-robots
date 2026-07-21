"""v17a: hand-coded local distance-spring controller probe (no NN).

Tests the hypothesis: can a purely local rule with role-pair desired distances
produce a clear T4 triangle? If yes, it's a teacher for BC. If no, pure local
role-interaction is insufficient and we need explicit slot/frame cue.

Controller:
  Each agent i has slot_i ∈ [0, N-1] (from rank of z[i, 0]).
  For each neighbor j within comm_radius:
    desired_dist = ||template[slot_i] - template[slot_j]||
    actual_dist  = ||p_j - p_i||
    F_ij = k_spring * (actual_dist - desired_dist) * unit(p_j - p_i)
           # positive (pull together) when too far, negative (push apart) when too close
  F_i = sum F_ij   (net spring force)
  → continuous action (linear, angular) that points heading toward F_i and moves forward.

Outputs:
  - per-seed eval metrics (succ@0.4d, density, m3, spread, coll)
  - render mp4 at seed=42 (if --render)
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import yaml

from environment import MeanShiftEnv, load_config
from shape_metrics import make_centered_t4_template, pairwise_dist


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]."""
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def compute_controller_action(env: MeanShiftEnv, slot_to_pair_dist: torch.Tensor,
                               comm_radius: float, k_spring: float,
                               linear_max_speed: float) -> torch.Tensor:
    """Spring-force controller. Returns continuous action [B, N, 2] in [-1, 1].

    Args:
      slot_to_pair_dist: [N, N] desired pairwise distances from the template
      comm_radius: only neighbors within this range contribute force
      k_spring: spring gain (per-d)
      linear_max_speed: scale for converting force magnitude → linear action
    """
    pos = env.positions                                                     # [B, N, 2]
    rot = env.rotations                                                     # [B, N]
    slots = env.slot_assignment                                             # [B, N] long
    B, N, _ = pos.shape

    # Pairwise vectors: rel[b, i, j] = p_j - p_i, shape [B, N, N, 2]
    rel = pos.unsqueeze(1) - pos.unsqueeze(2)
    actual = rel.norm(dim=-1)                                                # [B, N, N]

    # Desired pairwise distances per (slot_i, slot_j): [B, N, N]
    # slots[b, i] = slot_i; gather D[slot_i, slot_j]
    # slot_to_pair_dist may be [N, N] (single-shape) or [B, N, N]
    # (multi-shape, one shape per env).
    s_i = slots.unsqueeze(2).expand(B, N, N)                                 # [B, N, N]
    s_j = slots.unsqueeze(1).expand(B, N, N)
    if slot_to_pair_dist.ndim == 3:
        B_idx = torch.arange(B, device=pos.device).view(B, 1, 1)
        desired = slot_to_pair_dist[B_idx, s_i, s_j]                         # [B, N, N]
    else:
        desired = slot_to_pair_dist[s_i, s_j]                                # [B, N, N]

    # Mask: in-comm-radius and j != i
    eye = torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)     # [1, N, N]
    in_range = (actual < comm_radius) & ~eye                                 # [B, N, N]

    # Spring force on i from j: k * (actual - desired) * unit(p_j - p_i)
    # Direction = rel / actual (unit vector pointing from i toward j)
    actual_safe = actual.clamp_min(1e-6)
    direction = rel / actual_safe.unsqueeze(-1)                              # [B, N, N, 2]
    spring_mag = k_spring * (actual - desired)                               # [B, N, N], +pull -push
    F_ij = spring_mag.unsqueeze(-1) * direction                              # [B, N, N, 2]
    F_ij = F_ij * in_range.unsqueeze(-1).float()                             # zero out non-neighbors and self

    F = F_ij.sum(dim=2)                                                      # [B, N, 2] net force on each agent

    # Convert force → continuous action
    F_mag = F.norm(dim=-1)                                                   # [B, N]
    desired_heading = torch.atan2(F[..., 1], F[..., 0])                      # [B, N]
    heading_err = wrap_angle(desired_heading - rot)                          # [B, N] ∈ (-pi, pi]

    # angular: turn toward desired heading, clamped to [-1, 1] in units of rotate_step
    rotate_step = float(env.rotate_step)
    angular = (heading_err / rotate_step).clamp(-1.0, 1.0)                   # [B, N]

    # linear: speed proportional to F_mag, but only when roughly facing target
    # cos(err) damps when facing wrong way
    aim_factor = heading_err.cos().clamp_min(0.0)                            # [B, N], 1 when aligned, 0 when 90° off
    speed = (F_mag / linear_max_speed).clamp(0.0, 1.0) * aim_factor          # [B, N]
    # If F is essentially zero (well-positioned), output zero linear
    speed = torch.where(F_mag < 1e-4, torch.zeros_like(speed), speed)

    action = torch.stack([speed, angular], dim=-1)                            # [B, N, 2]
    return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=REPO.parent / "configs" / "n10_triangle.yaml")
    p.add_argument("--n_envs", type=int, default=256)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k_spring", type=float, default=1.0,
                   help="Spring gain (force per unit d-mismatch)")
    p.add_argument("--linear_max_speed", type=float, default=0.05,
                   help="Force magnitude that maps to action linear=1.0 (in d units)")
    p.add_argument("--render_n", type=int, default=0,
                   help="If > 0, also render this many envs to mp4 (uses seed=42)")
    p.add_argument("--render_out", type=Path, default=REPO.parent / "runs/mappo/v17a_controller.mp4")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["episode_length"] = args.T
    spacing = float(cfg["target_spacing"])
    comm_radius = float(cfg["comm_radius"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build T4 pairwise distance table [10, 10]
    template = make_centered_t4_template(spacing, device=device)             # [10, 2]
    template_pair_dist = pairwise_dist(template)                              # [10, 10]

    # linear_max_speed cfg arg is in d-units; convert to absolute force magnitude
    F_norm = args.linear_max_speed * spacing

    print(f"[v17a] spacing={spacing:.4f}  comm_radius={comm_radius:.4f} (={comm_radius/spacing:.2f}d)  "
          f"k_spring={args.k_spring}  F_norm={F_norm:.5f}")
    print(f"[v17a] template pairwise distances (in d units):")
    print((template_pair_dist / spacing).cpu().numpy().round(2))

    import statistics
    KEYS = ["shape_succ", "shape_succ_strict", "shape_match_d", "shape_match_d_max",
            "shape_profile", "bond_penalty", "coll"]
    agg = {k: [] for k in KEYS}

    for s in range(args.n_seeds):
        env = MeanShiftEnv(cfg, n_envs=args.n_envs, device=str(device), seed=args.seed + s)
        env.reset_all()
        env.set_cue_present_prob(0.0)
        with torch.no_grad():
            for _t in range(args.T):
                action = compute_controller_action(env, template_pair_dist,
                                                    comm_radius=comm_radius,
                                                    k_spring=args.k_spring,
                                                    linear_max_speed=F_norm)
                # env.expert_action populates internal state; we bypass and call step directly
                _info = env.step(action)
        # Final-state metrics
        prof, match_d, iso, bp = env.compute_shape_metrics(template, compute_match_distance=True, theta_bins=72)
        succ_loose = (match_d < 0.4 * spacing).float().mean().item()
        succ_strict = (match_d < 0.25 * spacing).float().mean().item()
        m = {
            "shape_succ": succ_loose, "shape_succ_strict": succ_strict,
            "shape_match_d": (match_d.mean() / spacing).item(),
            "shape_match_d_max": (match_d.max() / spacing).item(),
            "shape_profile": prof.mean().item(),
            "bond_penalty": bp.mean().item(),
            "coll": env._cum_collision.float().mean().item(),
        }
        for k in KEYS: agg[k].append(m.get(k, 0.0))
        print(f"[seed {args.seed+s}] succ@0.4d={m['shape_succ']:.3f} succ@0.25d={m['shape_succ_strict']:.3f} "
              f"match={m['shape_match_d']:.3f}d max={m['shape_match_d_max']:.3f}d "
              f"prof={m['shape_profile']:.3f} bond={m['bond_penalty']:.3f} coll={m['coll']:.0f}")

    print(f"\n=== v17a controller multi-seed ({args.n_seeds} × {args.n_envs} × {args.T}) ===")
    def stat(name, key, fmt=".3f"):
        vals = agg[key]
        mean = statistics.mean(vals); std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {name:18s} {mean:{fmt}} ± {std:{fmt}}")
    for k in ["shape_succ", "shape_succ_strict", "shape_match_d", "shape_match_d_max",
              "shape_profile", "bond_penalty"]: stat(k, k)
    stat("coll", "coll", ".1f")

    # Optional render
    if args.render_n > 0:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        env = MeanShiftEnv(cfg, n_envs=args.render_n, device=str(device), seed=42)
        env.reset_all(); env.set_cue_present_prob(0.0)
        traj = []
        with torch.no_grad():
            for _t in range(args.T):
                action = compute_controller_action(env, template_pair_dist,
                                                    comm_radius=comm_radius,
                                                    k_spring=args.k_spring,
                                                    linear_max_speed=F_norm)
                env.step(action)
                traj.append(env.positions.detach().cpu().numpy().copy())
        import numpy as np
        traj = np.stack(traj, axis=0)                                         # [T, B, N, 2]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        scatters = []
        for b, ax in enumerate(axes[:args.render_n]):
            ax.set_xlim(-0.18, 0.18); ax.set_ylim(-0.18, 0.18); ax.set_aspect("equal")
            ax.set_title(f"env {b}")
            sc = ax.scatter(traj[0, b, :, 0], traj[0, b, :, 1], s=40, c=range(env.N), cmap="tab10")
            scatters.append(sc)

        def update(t):
            for b, sc in enumerate(scatters):
                sc.set_offsets(traj[t, b])
            return scatters

        anim = animation.FuncAnimation(fig, update, frames=args.T, interval=50, blit=True)
        args.render_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[render] writing {args.render_out} ...")
        anim.save(str(args.render_out), writer="ffmpeg", fps=20, dpi=80)
        plt.close(fig)
        print(f"[render] done.")


if __name__ == "__main__":
    main()
