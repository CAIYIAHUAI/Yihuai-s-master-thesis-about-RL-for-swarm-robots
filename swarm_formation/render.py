"""Render an MP4 of a self-shape checkpoint forming any registered shape.

Accepts --shape and uses templates.make_template
+ env.slot_pair_dist override so the actor is evaluated on the forced shape.
The final title metrics are computed after exactly T environment steps, matching
evaluate.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_conda_ffmpeg = Path(sys.executable).parent / "ffmpeg"
if _conda_ffmpeg.exists():
    import matplotlib
    matplotlib.rcParams["animation.ffmpeg_path"] = str(_conda_ffmpeg)

from environment import MeanShiftEnv
from shape_metrics import pairwise_dist
from templates import make_template, make_target_map
from train_mappo import _build_actor, actor_dist, build_selfshape_obs


def best_fit_template(positions: torch.Tensor, template: torch.Tensor,
                      theta_bins: int = 36):
    B, N, _ = positions.shape
    centroid = positions.mean(dim=1, keepdim=True)
    pos_c = (positions - centroid).detach().cpu().numpy()
    tpl = template.detach().cpu().numpy()
    thetas = np.linspace(-np.pi, np.pi, theta_bins + 1)[:-1]
    md = np.zeros(B, dtype=np.float32)
    best_template = np.zeros((B, N, 2), dtype=np.float32)
    assignment = np.zeros((B, N), dtype=np.int64)
    for b in range(B):
        best_d = float("inf"); best_R = None; best_col = None
        for t in thetas:
            c, s = math.cos(t), math.sin(t)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            rotated = tpl @ R.T
            diff = pos_c[b][:, None, :] - rotated[None, :, :]
            cost = np.sqrt((diff ** 2).sum(-1))
            row, col = linear_sum_assignment(cost)
            d_mean = float(cost[row, col].mean())
            if d_mean < best_d:
                best_d = d_mean; best_R = R; best_col = col
        md[b] = best_d
        best_template[b] = tpl @ best_R.T + centroid[b].cpu().numpy()
        assignment[b] = best_col
    return (torch.from_numpy(md).to(positions.device),
            torch.from_numpy(best_template).to(positions.device),
            torch.from_numpy(assignment).to(positions.device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--shape", required=True,
                   choices=["t4", "hex4", "ring10", "row5x2",
                            "t6", "hex21", "ring21", "row7x3",
                            "t7", "hex28", "ring28", "row7x4"])
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--T", type=int, default=600)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metric_every", type=int, default=5)
    p.add_argument("--theta_bins", type=int, default=72,
                   help="Rotation grid for the best-fit metric; 72 matches evaluate.py")
    p.add_argument("--comm_radius_mm", type=float, default=None,
                   help="Override the trained comm_radius (mm). Used to study degradation "
                        "under tighter sensing; density branch is masked under cue_off.")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = dict(state["config"]); cfg.pop("arch_version", None)
    cfg["episode_length"] = args.T

    if args.comm_radius_mm is not None:
        new_cr = args.comm_radius_mm * 1e-3
        # Shrink the density-branch radii so the constructor invariants pass.
        # Density features are masked when cue_off=True, so this does not affect
        # the actor's strict-local observation.
        cfg["comm_radius"]         = new_cr
        cfg["density_radius"]      = min(cfg.get("density_radius", 0.0675), new_cr / 2.0)
        cfg["kernel_sigma"]        = min(cfg.get("kernel_sigma", 0.03375),
                                          cfg["density_radius"] / 2.0)
        cfg["target_sense_radius"] = max(new_cr, cfg["density_radius"])
        print(f"[override] comm_radius = {new_cr*1000:.1f}mm "
              f"= {new_cr/cfg['target_spacing']:.2f}d "
              f"(originally {state['config'].get('comm_radius', 0.180)*1000:.0f}mm)")
    spacing = float(cfg["target_spacing"])

    actor = _build_actor(cfg, args.device, selfshape=True)
    actor.load_state_dict(state["actor_state"], strict=False)
    actor.eval()

    env = MeanShiftEnv(cfg, n_envs=args.n_envs, device=args.device, seed=args.seed)
    # Match evaluate.py: re-sample latent_z / positions / target_map once
    # after construction. MeanShiftEnv.__init__ already calls reset_all(), so this
    # is a second reset that advances the RNG by one draw. Without it, the same
    # --seed would land on a different latent_z / slot_assignment than the eval
    # script reports, producing systematically different match_d numbers.
    env.reset_all()
    env.set_cue_present_prob(0.0)

    # Force shape
    tmpl = make_template(args.shape, spacing, device=args.device)
    if tmpl.shape[0] != env.N:
        raise SystemExit(
            f"shape {args.shape!r} has {tmpl.shape[0]} points but checkpoint env has "
            f"n_agents={env.N}"
        )
    pair_dist = pairwise_dist(tmpl)
    if env.slot_pair_dist.ndim == 3:
        env.slot_pair_dist[:] = pair_dist.unsqueeze(0).expand(args.n_envs, -1, -1)
    if args.shape in env._shape_names:
        env.shape_id[:] = env._shape_names.index(args.shape)
    env.target_map[:] = make_target_map(
        args.shape, env.positions, spacing=spacing,
        theta=float(cfg.get("target_theta", 0.0)),
        center_mode=str(cfg.get("target_center_mode", "initial_swarm_centroid")),
    )

    success_loose = float(cfg.get("success_shape_distance", 0.4)) * spacing
    success_strict = float(cfg.get("success_shape_distance_strict", 0.25)) * spacing

    h = torch.zeros(args.n_envs, env.N, actor.hidden, device=args.device)
    noise_gen = torch.Generator(device=args.device); noise_gen.manual_seed(args.seed + 1)

    # Store the initial state plus T post-step states. The final frame therefore
    # matches evaluate.py, which computes metrics after T env.step calls.
    n_frames = args.T + 1
    positions = np.zeros((n_frames, args.n_envs, env.N, 2), dtype=np.float32)
    md_hist = np.zeros((n_frames, args.n_envs), dtype=np.float32)
    tpl_hist = np.zeros((n_frames, args.n_envs, env.N, 2), dtype=np.float32)
    last_md = None; last_tpl = None

    with torch.no_grad():
        for t in range(n_frames):
            positions[t] = env.positions.cpu().numpy()
            if t % args.metric_every == 0 or t == args.T:
                md, btpl, _ = best_fit_template(env.positions, tmpl,
                                                theta_bins=args.theta_bins)
                last_md = md.cpu().numpy()
                last_tpl = btpl.cpu().numpy()
            md_hist[t] = last_md
            tpl_hist[t] = last_tpl
            if t == args.T:
                break
            exp_out = env.expert_action(noise_gen=noise_gen, build_obs=False)
            obs, ei, ea = build_selfshape_obs(env, exp_out,
                                              cue_present=env.cue_present,
                                              latent_z=env.latent_z,
                                              heading_locked=True)
            logits, vel_mu, h = actor(obs, ei, ea, h)
            action = torch.tanh(vel_mu).clamp(-1.0, 1.0)
            env.step(action)

    final_md_d = md_hist[-1] / spacing
    print(f"[{args.shape}] final match_d (d-units): {[f'{x:.3f}' for x in final_md_d]}")
    print(f"[{args.shape}] succ@0.4d  per_env={[1 if x<success_loose else 0 for x in md_hist[-1]]}")
    print(f"[{args.shape}] succ@0.25d per_env={[1 if x<success_strict else 0 for x in md_hist[-1]]}")

    cols = int(math.ceil(math.sqrt(args.n_envs)))
    rows = int(math.ceil(args.n_envs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    extents = []
    for b in range(args.n_envs):
        all_pts = np.concatenate([positions[:, b].reshape(-1, 2),
                                  tpl_hist[:, b].reshape(-1, 2)], axis=0)
        c = all_pts.mean(0)
        r = max(np.abs(all_pts - c).max() * 1.15, 0.08)
        extents.append((c, r))
    s_pos, s_tpl, titles = [], [], []
    for i, ax in enumerate(axes):
        if i >= args.n_envs:
            ax.axis("off"); continue
        c, r = extents[i]
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_aspect("equal")
        s_tpl.append(ax.scatter(tpl_hist[0, i, :, 0], tpl_hist[0, i, :, 1],
                                marker="x", c="gray", s=120, alpha=0.55,
                                linewidths=1.8))
        s_pos.append(ax.scatter(positions[0, i, :, 0], positions[0, i, :, 1],
                                c=np.arange(env.N), cmap="tab20", s=70,
                                edgecolors="black", linewidths=0.6))
        titles.append(ax.set_title(f"env {i} t=0 [{args.shape}]"))

    def update(t):
        for i in range(args.n_envs):
            s_pos[i].set_offsets(positions[t, i])
            s_tpl[i].set_offsets(tpl_hist[t, i])
            md_d = md_hist[t, i] / spacing
            sl = "✓" if md_hist[t, i] < success_loose else "·"
            ss = "✓" if md_hist[t, i] < success_strict else "·"
            titles[i].set_text(
                f"env {i} t={t} [{args.shape}]  match={md_d:.3f}d  0.4d{sl} 0.25d{ss}"
            )
        return [*s_pos, *s_tpl, *titles]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if animation.writers.is_available("ffmpeg"):
        print(f"[{args.shape}] writing {args.out} (mp4)...")
        anim.save(str(args.out), writer=animation.FFMpegWriter(fps=30, bitrate=2400),
                  dpi=100)
    else:
        gif_path = args.out.with_suffix(".gif")
        print(f"[{args.shape}] ffmpeg not found, writing {gif_path}...")
        anim.save(str(gif_path), writer=animation.PillowWriter(fps=20), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
