"""
Test 2: Per-agent row loss direction verification.

Verifies that the proposed per-agent row_loss:
  (a) Is near zero when agents match the template perfectly.
  (b) Increases mainly for the displaced agent when one agent is moved.
  (c) Decreases when that agent moves BACK toward correct position.
  (d) Increases when that agent moves FURTHER away.

If all checks pass, the per-agent reward signal points in the correct
direction for individual agents — the mathematical prerequisite for
solving the credit assignment problem.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.triangle_reward import (
    build_triangle_template,
    squared_distance_matrix_batched,
    row_signature_cost_matrix,
    sinkhorn,
    scale_invariant_distance_loss,
)

TAU = 0.001
ITERS = 100
EPS = 1e-8
N = 30
SCALE = 0.045  # 45mm target spacing → sim units


def compute_per_agent_row_loss(pos, tmpl_dist2):
    """Row-wise scale-invariant loss. pos: [B,N,2] → row_loss: [B,N]."""
    dist2 = squared_distance_matrix_batched(pos)
    cost = row_signature_cost_matrix(dist2, tmpl_dist2)
    P = sinkhorn(cost, tau=TAU, iters=ITERS, eps=EPS)
    tmpl = tmpl_dist2.unsqueeze(0).expand(pos.shape[0], -1, -1)
    dist2_soft = P @ tmpl @ P.transpose(1, 2)

    a = dist2 - dist2.mean(dim=-1, keepdim=True)
    b = dist2_soft - dist2_soft.mean(dim=-1, keepdim=True)
    dot = (a * b).sum(dim=-1)
    norm = a.norm(dim=-1) * b.norm(dim=-1) + EPS
    return 1.0 - dot / norm


def compute_global_loss(pos, tmpl_dist2):
    """Global scale-invariant formation loss. pos: [B,N,2] → loss: [B]."""
    dist2 = squared_distance_matrix_batched(pos)
    cost = row_signature_cost_matrix(dist2, tmpl_dist2)
    P = sinkhorn(cost, tau=TAU, iters=ITERS, eps=EPS)
    tmpl = tmpl_dist2.unsqueeze(0).expand(pos.shape[0], -1, -1)
    dist2_soft = P @ tmpl @ P.transpose(1, 2)
    return scale_invariant_distance_loss(dist2, dist2_soft, eps=EPS)


def main():
    tmpl = build_triangle_template(N, side_length=1.0)
    tmpl_dist2 = squared_distance_matrix_batched(tmpl.unsqueeze(0))[0]

    pos_perfect = tmpl.unsqueeze(0) * SCALE

    print("=" * 60)
    print("CHECK A: Perfect match → row_loss ≈ 0 for every agent")
    print("=" * 60)
    rl_perfect = compute_per_agent_row_loss(pos_perfect, tmpl_dist2)
    gl_perfect = compute_global_loss(pos_perfect, tmpl_dist2)
    print(f"  Global shape_loss : {gl_perfect.item():.6f}")
    print(f"  Row_loss mean     : {rl_perfect.mean().item():.6f}")
    print(f"  Row_loss max      : {rl_perfect.max().item():.6f}")
    ok_a = rl_perfect.max().item() < 0.01
    print(f"  PASS: {ok_a}\n")

    TARGET_AGENT = 5
    DISPLACE = 0.01  # 10mm displacement

    print("=" * 60)
    print(f"CHECK B: Displace agent {TARGET_AGENT} by {DISPLACE*1000:.0f}mm")
    print("         → its row_loss should increase much more than others")
    print("=" * 60)
    pos_disp = pos_perfect.clone()
    pos_disp[0, TARGET_AGENT, 0] += DISPLACE

    rl_disp = compute_per_agent_row_loss(pos_disp, tmpl_dist2)
    gl_disp = compute_global_loss(pos_disp, tmpl_dist2)

    others_mask = torch.arange(N) != TARGET_AGENT
    rl_target = rl_disp[0, TARGET_AGENT].item()
    rl_others_mean = rl_disp[0, others_mask].mean().item()
    rl_others_max = rl_disp[0, others_mask].max().item()

    print(f"  Global shape_loss       : {gl_disp.item():.6f}")
    print(f"  Agent {TARGET_AGENT} row_loss       : {rl_target:.6f}")
    print(f"  Other agents mean       : {rl_others_mean:.6f}")
    print(f"  Other agents max        : {rl_others_max:.6f}")
    print(f"  Ratio (target/others)   : {rl_target / max(rl_others_mean, 1e-10):.1f}x")
    ok_b = rl_target > rl_others_mean * 2.0
    print(f"  PASS (target > 2x others): {ok_b}\n")

    STEP = 0.003  # 3mm step

    print("=" * 60)
    print(f"CHECK C: Move agent {TARGET_AGENT} TOWARD correct position")
    print(f"         → its row_loss should DECREASE")
    print("=" * 60)
    pos_good = pos_disp.clone()
    pos_good[0, TARGET_AGENT, 0] -= STEP  # back toward correct

    rl_good = compute_per_agent_row_loss(pos_good, tmpl_dist2)
    delta_good = rl_good[0, TARGET_AGENT].item() - rl_target
    print(f"  row_loss change: {delta_good:+.6f} (should be negative)")
    ok_c = delta_good < 0
    print(f"  PASS: {ok_c}\n")

    print("=" * 60)
    print(f"CHECK D: Move agent {TARGET_AGENT} AWAY from correct position")
    print(f"         → its row_loss should INCREASE")
    print("=" * 60)
    pos_bad = pos_disp.clone()
    pos_bad[0, TARGET_AGENT, 0] += STEP  # further away

    rl_bad = compute_per_agent_row_loss(pos_bad, tmpl_dist2)
    delta_bad = rl_bad[0, TARGET_AGENT].item() - rl_target
    print(f"  row_loss change: {delta_bad:+.6f} (should be positive)")
    ok_d = delta_bad > 0
    print(f"  PASS: {ok_d}\n")

    print("=" * 60)
    print(f"CHECK E: Repeat for multiple agents and directions")
    print("=" * 60)
    n_pass = 0
    n_total = 0
    for agent_id in [0, 7, 14, 21, 29]:
        for dim in [0, 1]:
            pos_base = pos_perfect.clone()
            pos_base[0, agent_id, dim] += DISPLACE

            rl_base = compute_per_agent_row_loss(pos_base, tmpl_dist2)[0, agent_id].item()

            pos_toward = pos_base.clone()
            pos_toward[0, agent_id, dim] -= STEP
            rl_toward = compute_per_agent_row_loss(pos_toward, tmpl_dist2)[0, agent_id].item()

            pos_away = pos_base.clone()
            pos_away[0, agent_id, dim] += STEP
            rl_away = compute_per_agent_row_loss(pos_away, tmpl_dist2)[0, agent_id].item()

            toward_ok = rl_toward < rl_base
            away_ok = rl_away > rl_base
            n_total += 2
            n_pass += int(toward_ok) + int(away_ok)
            tag = "OK" if (toward_ok and away_ok) else "FAIL"
            dim_name = "x" if dim == 0 else "y"
            print(f"  Agent {agent_id:2d} dim={dim_name}: toward {rl_toward-rl_base:+.6f}  "
                  f"away {rl_away-rl_base:+.6f}  [{tag}]")

    print(f"\n  Direction checks passed: {n_pass}/{n_total}")
    ok_e = n_pass == n_total
    print(f"  ALL PASS: {ok_e}\n")

    print("=" * 60)
    all_ok = ok_a and ok_b and ok_c and ok_d and ok_e
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
