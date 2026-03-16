"""
Test 3: Credit assignment SNR comparison.

Tests three aspects of per-agent vs shared reward:

Part A - Single agent isolation:
  Only ONE agent moves, others fixed.
  Measures how sensitive each reward scheme is to individual movement.

Part B - Cross-agent differentiation:
  ALL agents move simultaneously.
  Measures whether per-agent rewards can DISTINGUISH which agents
  moved well vs poorly within the same timestep.

Part C - Original correlation test:
  ALL agents move, correlation between agent's move and its reward.
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
SCALE = 0.045


def compute_per_agent_row_loss(pos, tmpl_dist2):
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
    dist2 = squared_distance_matrix_batched(pos)
    cost = row_signature_cost_matrix(dist2, tmpl_dist2)
    P = sinkhorn(cost, tau=TAU, iters=ITERS, eps=EPS)
    tmpl = tmpl_dist2.unsqueeze(0).expand(pos.shape[0], -1, -1)
    dist2_soft = P @ tmpl @ P.transpose(1, 2)
    return scale_invariant_distance_loss(dist2, dist2_soft, eps=EPS)


def corrcoef(x, y):
    x_c = x - x.mean()
    y_c = y - y.mean()
    return (x_c * y_c).sum() / (x_c.norm() * y_c.norm() + 1e-12)


def main():
    tmpl = build_triangle_template(N, side_length=1.0)
    tmpl_dist2 = squared_distance_matrix_batched(tmpl.unsqueeze(0))[0]
    pos_perfect = tmpl.unsqueeze(0) * SCALE

    torch.manual_seed(42)

    # ================================================================
    # PART A: Single-agent isolation
    # ================================================================
    print("=" * 70)
    print("PART A: Single agent moves, others fixed")
    print("        How sensitive is each reward to ONE agent's movement?")
    print("=" * 70)

    STEP = 0.003  # 3mm
    print(f"\n{'Agent':>6}  {'Global Δloss':>14}  {'row_loss_i Δ':>14}  {'Ratio':>8}")
    print("-" * 50)

    ratios = []
    for agent_id in range(0, N, 3):
        pos_moved = pos_perfect.clone()
        pos_moved[0, agent_id, 0] += STEP

        gl_base = compute_global_loss(pos_perfect, tmpl_dist2).item()
        gl_moved = compute_global_loss(pos_moved, tmpl_dist2).item()
        d_global = gl_moved - gl_base

        rl_base = compute_per_agent_row_loss(pos_perfect, tmpl_dist2)[0, agent_id].item()
        rl_moved = compute_per_agent_row_loss(pos_moved, tmpl_dist2)[0, agent_id].item()
        d_row = rl_moved - rl_base

        ratio = abs(d_row) / max(abs(d_global), 1e-12)
        ratios.append(ratio)
        print(f"{agent_id:6d}  {d_global:+14.6f}  {d_row:+14.6f}  {ratio:8.1f}x")

    mean_ratio = sum(ratios) / len(ratios)
    print(f"{'MEAN':>6}  {'':>14}  {'':>14}  {mean_ratio:8.1f}x")
    print(f"\n  Per-agent row_loss is on average {mean_ratio:.1f}x more sensitive")
    print(f"  to single-agent movement than global loss.")

    # ================================================================
    # PART B: Cross-agent differentiation
    # ================================================================
    print("\n" + "=" * 70)
    print("PART B: All agents move simultaneously")
    print("        Can per-agent reward distinguish good vs bad movers?")
    print("=" * 70)

    N_TRIALS = 300
    PERTURB = 0.002

    cross_agent_corrs_peragent = []
    cross_agent_corrs_shared = []
    per_agent_reward_stds = []

    for t in range(N_TRIALS):
        perturbation = torch.randn(N, 2) * PERTURB
        pos = pos_perfect.clone()
        pos[0] += perturbation

        # "Movement quality" per agent: how much closer to correct position?
        # Negative displacement magnitude = closer (smaller is better).
        # Since we start at perfect, ANY perturbation is bad.
        # movement_quality_i = -|perturbation_i| (bigger move = worse)
        move_quality = -perturbation.norm(dim=-1)  # [N]

        # Shared reward: same for all agents
        shared_rew = -compute_global_loss(pos, tmpl_dist2).item()
        # Per-agent reward: different for each agent
        per_agent_rew = -compute_per_agent_row_loss(pos, tmpl_dist2)[0]  # [N]

        # Cross-agent correlation: within this timestep,
        # do agents with better movement_quality get better reward?
        c_pa = corrcoef(move_quality, per_agent_rew).item()
        cross_agent_corrs_peragent.append(c_pa)

        per_agent_reward_stds.append(per_agent_rew.std().item())

    cross_agent_corrs_peragent = torch.tensor(cross_agent_corrs_peragent)
    per_agent_reward_stds = torch.tensor(per_agent_reward_stds)

    print(f"\n  Across {N_TRIALS} timesteps:")
    print(f"  Per-agent reward std across agents: {per_agent_reward_stds.mean():.6f}")
    print(f"    (shared reward std = 0 by definition)")
    print(f"\n  Cross-agent correlation")
    print(f"    (does 'moved less = better reward' hold across agents?)")
    print(f"    Per-agent: mean={cross_agent_corrs_peragent.mean():.4f}, "
          f"std={cross_agent_corrs_peragent.std():.4f}")
    print(f"    Shared:    correlation is undefined (all agents get same reward)")

    positive_frac = (cross_agent_corrs_peragent > 0).float().mean().item()
    print(f"\n  Fraction of timesteps with positive correlation: {positive_frac:.1%}")
    print(f"  (>50% means per-agent reward correctly differentiates agents)")

    # ================================================================
    # PART C: Displaced starting state (more realistic)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART C: Start from DISPLACED state (not perfect)")
    print("        More realistic: agents are not at template positions")
    print("=" * 70)

    torch.manual_seed(123)
    pos_displaced = pos_perfect.clone()
    pos_displaced[0] += torch.randn(N, 2) * 0.01  # 10mm random displacement

    N_TRIALS_C = 300
    STEP_C = 0.001  # 1mm step

    correct_direction_shared = 0
    correct_direction_peragent = 0
    total_checks = 0

    for t in range(N_TRIALS_C):
        agent_id = t % N
        dim = t % 2

        # Correct direction: move TOWARD perfect position
        sign_toward = 1.0 if pos_perfect[0, agent_id, dim] > pos_displaced[0, agent_id, dim] else -1.0

        pos_toward = pos_displaced.clone()
        pos_toward[0, agent_id, dim] += sign_toward * STEP_C

        pos_away = pos_displaced.clone()
        pos_away[0, agent_id, dim] -= sign_toward * STEP_C

        # Shared reward
        gl_toward = -compute_global_loss(pos_toward, tmpl_dist2).item()
        gl_away = -compute_global_loss(pos_away, tmpl_dist2).item()

        # Per-agent reward
        pa_toward = -compute_per_agent_row_loss(pos_toward, tmpl_dist2)[0, agent_id].item()
        pa_away = -compute_per_agent_row_loss(pos_away, tmpl_dist2)[0, agent_id].item()

        if gl_toward > gl_away:
            correct_direction_shared += 1
        if pa_toward > pa_away:
            correct_direction_peragent += 1
        total_checks += 1

    print(f"\n  Moving one agent toward/away from correct position ({total_checks} trials):")
    print(f"  Shared reward gives correct direction:     "
          f"{correct_direction_shared}/{total_checks} = {correct_direction_shared/total_checks:.1%}")
    print(f"  Per-agent reward gives correct direction:   "
          f"{correct_direction_peragent}/{total_checks} = {correct_direction_peragent/total_checks:.1%}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Part A - Single-agent sensitivity:   per-agent is {mean_ratio:.1f}x more sensitive")
    print(f"  Part B - Cross-agent differentiation: per-agent reward std = {per_agent_reward_stds.mean():.6f}")
    print(f"           Positive cross-agent correlation: {positive_frac:.1%} of timesteps")
    print(f"  Part C - Direction accuracy:          shared {correct_direction_shared/total_checks:.1%} "
          f"vs per-agent {correct_direction_peragent/total_checks:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
