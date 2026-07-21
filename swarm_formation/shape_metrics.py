"""Shape-only metrics for self-organization T4 (no target slots used).

All functions consume robot positions [B, N, 2] and a centered T4 template
[N, 2]. Used by self-shape mode for reward shaping (per-step) and
success/eval (periodic).

Risks addressed:
- R2: bond_length_penalty (displacement-based control toward {d, √3 d, 2d, √7 d, 3d})
- R4: shape_profile_error uses sorted distances + Laplacian eigvalsh, avoiding
      isospectral non-isomorphic local minima
- R6b: isolated_agent_count for per-agent local connectivity penalty
"""
from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment

from target import make_t4_template


def make_centered_t4_template(spacing: float, device: torch.device | str = "cpu",
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Wrapper around target.make_t4_template (already centered). Shape [10, 2]."""
    return make_t4_template(spacing, theta=0.0, device=device, dtype=dtype)


def pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """Self-pairwise distances. x: [..., N, 2] -> [..., N, N]."""
    return (x.unsqueeze(-2) - x.unsqueeze(-3)).norm(dim=-1)


def shape_profile_error(positions: torch.Tensor, template: torch.Tensor,
                        alpha: float = 0.5) -> torch.Tensor:
    """Sorted-pairwise-distance error + alpha * sorted-Laplacian-eigvalsh error.

    Both terms are normalized by template scale to be dimensionless. Used
    per-step as shape reward signal. (Risk 4 fix vs naive sorted-distance
    only, which has isospectral non-isomorphic counterexamples.)

    positions: [B, N, 2]
    template:  [N, 2] centered
    Returns:   [B] dimensionless error >= 0 (zero iff positions ~ template up to
               translation/rotation/permutation).
    """
    B, N, _ = positions.shape
    d_ij = pairwise_dist(positions)                                  # [B, N, N]
    e_ij = pairwise_dist(template)                                   # [N, N]

    # --- Sorted upper-triangle pairwise distances ---
    upper = torch.triu_indices(N, N, offset=1, device=positions.device)
    sd = d_ij[..., upper[0], upper[1]].sort(-1).values               # [B, K]
    se = e_ij[upper[0], upper[1]].sort(-1).values                    # [K]
    err_d = (sd - se.unsqueeze(0)).abs().mean(-1)                    # [B]

    # --- Sorted Laplacian eigenvalues of Gaussian-weighted adjacency ---
    sigma = se.median().clamp_min(1e-6)
    eye = torch.eye(N, device=positions.device, dtype=positions.dtype)
    A   = torch.exp(-(d_ij ** 2) / (2 * sigma ** 2)) * (1 - eye).unsqueeze(0)
    A_t = torch.exp(-(e_ij ** 2) / (2 * sigma ** 2)) * (1 - eye)
    L   = torch.diag_embed(A.sum(dim=-1)) - A                        # [B, N, N]
    L_t = torch.diag(A_t.sum(dim=-1)) - A_t                          # [N, N]
    # eigvalsh occasionally fails to converge on degenerate (collapsed) swarms;
    # fall back per-batch with a small jitter so eval continues for failing actors.
    try:
        eig = torch.linalg.eigvalsh(L).sort(-1).values                # [B, N]
    except torch._C._LinAlgError:
        eig = torch.zeros(B, N, device=positions.device, dtype=positions.dtype)
        for b in range(B):
            try:
                eig[b] = torch.linalg.eigvalsh(L[b]).sort(-1).values
            except torch._C._LinAlgError:
                jitter = 1e-4 * torch.eye(N, device=positions.device, dtype=positions.dtype)
                eig[b] = torch.linalg.eigvalsh(L[b] + jitter).sort(-1).values
    eig_t = torch.linalg.eigvalsh(L_t).sort(-1).values               # [N]
    err_e = (eig - eig_t.unsqueeze(0)).abs().mean(-1)                # [B]

    # --- Normalize and combine ---
    err_d_norm = err_d / se.mean().clamp_min(1e-6)                   # dimensionless
    err_e_norm = err_e / eig_t.abs().mean().clamp_min(1e-6)          # dimensionless
    return err_d_norm + alpha * err_e_norm                           # [B]


def shape_match_distance_grid(positions: torch.Tensor, template: torch.Tensor,
                              theta_bins: int = 72) -> torch.Tensor:
    """Min over rotation grid of mean per-robot Hungarian-matched distance.

    Used for success/eval/checkpoint selection (slow: B * theta_bins
    Hungarians on N x N cost matrices). For periodic eval only — not per-step
    reward.

    positions: [B, N, 2]
    template:  [N, 2] centered
    Returns:   [B] minimum mean per-robot distance to template under best
               rotation + best assignment.
    """
    B, N, _ = positions.shape
    centroid = positions.mean(dim=1, keepdim=True)
    pos_c = (positions - centroid).detach().cpu()                    # [B, N, 2]

    # Rotate template by each theta: [theta_bins, N, 2]
    thetas = torch.linspace(-math.pi, math.pi, theta_bins + 1,
                             dtype=template.dtype)[:-1]              # [theta_bins]
    c, s = thetas.cos(), thetas.sin()
    rot = torch.stack([torch.stack([c, -s], -1),
                       torch.stack([s, c], -1)], -2)                 # [theta_bins, 2, 2]
    rotated_t = torch.einsum("tij,kj->tki", rot, template.detach().cpu())  # [theta_bins, N, 2]

    rotated_np = rotated_t.numpy()
    pos_np = pos_c.numpy()
    out = torch.empty(B, dtype=positions.dtype)
    for b in range(B):
        best = float("inf")
        for t in range(theta_bins):
            diff = pos_np[b][:, None, :] - rotated_np[t][None, :, :]
            cost = (diff ** 2).sum(-1) ** 0.5
            row, col = linear_sum_assignment(cost)
            mean_d = float(cost[row, col].mean())
            if mean_d < best:
                best = mean_d
        out[b] = best
    return out.to(positions.device)


def shape_success(shape_match_distance: torch.Tensor, spacing: float,
                  threshold_d: float = 0.4) -> torch.Tensor:
    """Bool [B]: shape_match_distance < threshold_d * spacing."""
    return shape_match_distance < (threshold_d * spacing)


def bond_length_penalty(positions: torch.Tensor, comm_radius: float,
                        spacing: float,
                        bond_lengths_d: list[float] | None = None,
                        template: torch.Tensor | None = None) -> torch.Tensor:
    """Mean per-edge deviation from nearest valid bond length, in d-units.

    For each pair of robots within comm_radius, compute |distance - nearest
    valid bond length| / spacing and average over active edges.

    Bond lengths can be specified two ways:
      - bond_lengths_d: explicit list of allowed bond ratios in d-units (legacy
        single-shape; T4 uses [1.0, √3, 2.0, √7, 3.0]).
      - template: [10, 2] template tensor. Bond lengths are auto-derived as the
        sorted unique values of pairwise_dist(template) / spacing (rounded for
        clustering). Used for multi-shape where bond constraints are per-shape.

    Exactly one of `bond_lengths_d` or `template` must be provided.

    positions:      [B, N, 2]
    Returns:        [B] mean per-edge deviation in d-units, >= 0.
    """
    if (bond_lengths_d is None) == (template is None):
        raise ValueError("Provide exactly one of bond_lengths_d or template.")
    B, N, _ = positions.shape
    d_ij = pairwise_dist(positions)                                  # [B, N, N]
    upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=positions.device),
                       diagonal=1).unsqueeze(0).expand(B, -1, -1)    # [B, N, N]
    in_comm = d_ij < comm_radius
    edge_mask = (upper & in_comm).float()                            # [B, N, N]

    if template is not None:
        # Auto-derive bond lengths from template pairwise distances. Cluster
        # near-equal values by rounding to 1e-3 of a d-unit to avoid spurious
        # bond targets from floating-point noise.
        tpl_d = pairwise_dist(template)                               # [N, N]
        upper_tpl = torch.triu(torch.ones_like(tpl_d, dtype=torch.bool),
                                diagonal=1)
        tpl_d_flat = tpl_d[upper_tpl] / spacing                       # [45]
        keys = (tpl_d_flat * 1000.0).round().long()
        uniq_keys = torch.unique(keys)
        bond_abs = (uniq_keys.to(d_ij.dtype) / 1000.0) * spacing      # [K]
        bond_abs = bond_abs.to(d_ij.device)
    else:
        bond_abs = torch.tensor(bond_lengths_d, dtype=d_ij.dtype,
                                 device=d_ij.device) * spacing        # [K]
    diff = (d_ij.unsqueeze(-1) - bond_abs.view(1, 1, 1, -1)).abs()   # [B, N, N, K]
    nearest_bond = diff.min(dim=-1).values                            # [B, N, N]

    total = (nearest_bond * edge_mask).sum(dim=(1, 2)) / spacing      # in d-units
    edge_count = edge_mask.sum(dim=(1, 2)).clamp_min(1.0)
    return total / edge_count                                         # [B]


def per_agent_bond_penalty(positions: torch.Tensor, comm_radius: float,
                           spacing: float, bond_lengths_d: list[float]) -> torch.Tensor:
    """v8: per-agent bond penalty (asymmetric reward signal — breaks R4).

    For each agent i: mean over edges INCIDENT TO i (within comm_radius) of
    |distance - nearest valid bond length| / spacing. Different agents in the
    same env get different penalty values based on their own neighbor geometry.

    positions:      [B, N, 2]
    Returns:        [B, N] mean per-incident-edge deviation in d-units, ≥ 0.
                    Agents with zero in-comm neighbors return 0 (no penalty).
    """
    B, N, _ = positions.shape
    d_ij = pairwise_dist(positions)                                  # [B, N, N]
    diag = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    in_comm = (d_ij < comm_radius) & ~diag                           # [B, N, N]
    edge_mask = in_comm.float()                                       # [B, N, N]

    bond_abs = torch.tensor(bond_lengths_d, dtype=d_ij.dtype,
                             device=d_ij.device) * spacing
    diff = (d_ij.unsqueeze(-1) - bond_abs.view(1, 1, 1, -1)).abs()
    nearest_bond = diff.min(dim=-1).values                            # [B, N, N]

    # Sum over j (per agent i): edges incident to i
    per_agent_total = (nearest_bond * edge_mask).sum(dim=2) / spacing  # [B, N], d-units
    per_agent_count = edge_mask.sum(dim=2).clamp_min(1.0)              # [B, N]
    return per_agent_total / per_agent_count                          # [B, N]


def per_agent_collision_count(positions: torch.Tensor,
                               collision_hard: float) -> torch.Tensor:
    """v8: per-agent collision count (asymmetric reward signal).

    For each agent i: count of neighbors j with |x_i - x_j| < collision_hard.
    positions:        [B, N, 2]
    collision_hard:   scalar
    Returns:          [B, N] int → cast to float
    """
    B, N, _ = positions.shape
    d_ij = pairwise_dist(positions)
    diag = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    too_close = (d_ij < collision_hard) & ~diag
    return too_close.sum(dim=-1).float()                              # [B, N]


def per_agent_match_distance_iterated(positions: torch.Tensor, template: torch.Tensor,
                                       n_iters: int = 3,
                                       init_theta_bins: int = 12,
                                       prev_theta: torch.Tensor | None = None,
                                       return_theta: bool = False
                                       ):
    """v8: per-agent distance to its Hungarian-matched T4 slot, with template
    centered+rotated to best-fit current positions (Procrustes-iterated).

    Algorithm:
    1. Center positions on swarm centroid.
    2. Coarse search over `init_theta_bins` rotations of template, Hungarian
       assign at each, pick lowest-cost assignment as init.
    3. Iterate Procrustes (closed-form optimal rotation given assignment) +
       Hungarian re-assignment for `n_iters` rounds.
    4. Return per-agent distance from assigned slot in best-fit T4 frame.

    Used as v8 per-agent reward signal: agents close to their assigned slot
    get less penalty; encourages each agent to claim a unique slot.

    v9 FIX #11: optionally seed coarse init from `prev_theta` [B] (last step's
    best-fit theta) instead of full grid. With prev_theta given, only check
    a small window around it (±30°) to prevent step-to-step assignment flipping.

    positions:  [B, N, 2]
    template:   [N, 2] centered T4 template
    prev_theta: optional [B] previous step's best-fit theta (for stability)
    return_theta: if True, also return [B] this step's best-fit theta
    Returns:    [B, N] per-agent distance to assigned slot (absolute units, NOT d-units).
                If return_theta: ([B, N], [B] theta) tuple.
    """
    import math
    B, N, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    centroid = positions.mean(dim=1, keepdim=True)
    pos_c = (positions - centroid).detach().cpu().numpy()             # [B, N, 2]
    template_np = template.detach().cpu().numpy()                     # [N, 2]
    prev_theta_np = prev_theta.detach().cpu().numpy() if prev_theta is not None else None

    np = __import__("numpy")
    out = np.zeros((B, N), dtype=np.float32)
    out_theta = np.zeros(B, dtype=np.float32)

    for b in range(B):
        # Step 1: coarse init — narrow window around prev_theta if available
        if prev_theta_np is not None:
            # ±30° window, 7 bins (5° steps) — much narrower than full grid
            base = float(prev_theta_np[b])
            window_thetas = [base + math.radians(d) for d in range(-30, 31, 10)]
        else:
            window_thetas = [-math.pi + 2 * math.pi * k / init_theta_bins
                              for k in range(init_theta_bins)]

        best_cost_total = float("inf")
        best_R = None
        best_col = None
        for t in window_thetas:
            c, s = math.cos(t), math.sin(t)
            R = np.array([[c, -s], [s, c]], dtype=template_np.dtype)
            rotated = template_np @ R.T
            cost = np.linalg.norm(pos_c[b][:, None, :] - rotated[None, :, :], axis=-1)
            row, col = linear_sum_assignment(cost)
            tot = cost[row, col].sum()
            if tot < best_cost_total:
                best_cost_total = tot
                best_R = R
                best_col = col

        # Step 2: Procrustes-iterate
        col = best_col
        R = best_R
        for _ in range(n_iters):
            X = pos_c[b]
            Y = template_np[col]
            H = Y.T @ X
            U, _S, Vt = np.linalg.svd(H)
            det = np.linalg.det(Vt.T @ U.T)
            S_corr = np.array([[1.0, 0.0], [0.0, det]], dtype=template_np.dtype)
            R = Vt.T @ S_corr @ U.T
            rotated = template_np @ R.T
            cost = np.linalg.norm(pos_c[b][:, None, :] - rotated[None, :, :], axis=-1)
            row, col = linear_sum_assignment(cost)

        out[b] = cost[row, col]
        out_theta[b] = math.atan2(R[1, 0], R[0, 0])

    out_t = torch.from_numpy(out).to(device=device, dtype=dtype)
    if return_theta:
        theta_t = torch.from_numpy(out_theta).to(device=device, dtype=dtype)
        return out_t, theta_t
    return out_t


def isolated_agent_count(positions: torch.Tensor, comm_radius: float,
                         min_neighbors: int = 1) -> torch.Tensor:
    """Count agents with fewer than min_neighbors comm-neighbors. (R6b: per-agent
    local connectivity penalty, replaces global all-connected penalty.)

    positions: [B, N, 2]
    Returns:   [B] float count of isolated agents (0 = all agents have >=
               min_neighbors neighbors in comm range).
    """
    B, N, _ = positions.shape
    d_ij = pairwise_dist(positions)                                  # [B, N, N]
    diag = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    in_comm = (d_ij < comm_radius) & ~diag
    nbr_count = in_comm.sum(dim=-1)                                  # [B, N]
    return (nbr_count < min_neighbors).sum(dim=-1).float()           # [B]

# ---------------------------------------------------------------------------
# v18: Formation Stress Reward — strictly local pairwise distance potential
# ---------------------------------------------------------------------------
def formation_stress_reward(positions: torch.Tensor,
                             slot_assignment: torch.Tensor,
                             slot_pair_dist: torch.Tensor,
                             comm_radius: float,
                             spacing: float,
                             stress_norm: str = "fixed_n_minus_1",
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """v18 main reward (theoretically aligned with v17a controller).

    Computes the standard distance-based formation control stress potential
        V(p) = Σ_{(i,j)∈ E_loc} (||p_i - p_j|| - d_{ij})²
    where E_loc = {(i,j): ||p_i - p_j|| < comm_radius and i != j} is the local
    communication graph and d_{ij} = slot_pair_dist[slot_i, slot_j] is the
    desired pairwise distance from the active template.

    Strictly local: each agent's per_agent[b, i] depends only on its own slot,
    its neighbors within comm_radius, and pairwise distances. No global
    centroid, no global θ, no template overlay — exactly what the v17a
    hand-coded controller uses for its spring force.

    Returned reward is NEGATIVE stress, normalized so that values stay in a
    bounded range across init scale (per-agent err in d²-units, divided by
    max neighbor count).

    Args:
        positions:        [B, N, 2] current robot positions.
        slot_assignment:  [B, N] long, slot in 0..N-1 (rank z[..., 0]).
        slot_pair_dist:   [N_slots, N_slots] desired pairwise distances from
                          the active template (in absolute units, not d-units).
        comm_radius:      float, edges with d > this are dropped.
        spacing:          float, d. Used to normalize stress to d²-units.
        stress_norm:      "fixed_n_minus_1" restores the v18/V1 scale;
                          "active_neighbors" preserves the V2 large-N ablation.

    Returns:
        env_score [B]:    -mean per-agent stress (≤ 0; 0 iff agents lie on
                          template up to SE(2)+reflection in their local graph).
        per_agent [B, N]: -per-agent local stress (≤ 0).
    """
    B, N, _ = positions.shape
    device = positions.device

    # Actual pairwise distances [B, N, N]
    actual = pairwise_dist(positions)
    # Desired pairwise distances per (slot_i, slot_j) [B, N, N].
    # slot_pair_dist may be [N, N] (single-shape) or [B, N, N]
    # (multi-shape, one shape per env).
    if slot_pair_dist.ndim == 3:
        B_idx = torch.arange(B, device=device).view(B, 1, 1)
        s_i = slot_assignment.unsqueeze(2)                          # [B, N, 1]
        s_j = slot_assignment.unsqueeze(1)                          # [B, 1, N]
        desired = slot_pair_dist[B_idx, s_i, s_j]                   # [B, N, N]
    else:
        desired = slot_pair_dist[slot_assignment.unsqueeze(2),
                                  slot_assignment.unsqueeze(1)]
    # Squared error in d²-units
    err_sq = ((actual - desired) / spacing) ** 2                                # [B, N, N]
    # Locality mask: only neighbors within comm_radius and j != i
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)            # [1, N, N]
    in_range = (actual < comm_radius) & ~eye                                    # [B, N, N]
    in_range_f = in_range.float()
    err_sq = err_sq * in_range_f
    if stress_norm == "fixed_n_minus_1":
        denom = torch.full((B, N), float(max(N - 1, 1)),
                           device=device, dtype=err_sq.dtype)
    elif stress_norm == "active_neighbors":
        # V2 ablation: normalize each agent by its actual local degree.
        denom = in_range_f.sum(dim=-1).clamp_min(1.0)
    else:
        raise ValueError(
            f"unknown stress_norm={stress_norm!r}; expected "
            "'fixed_n_minus_1' or 'active_neighbors'"
        )
    per_agent_stress = err_sq.sum(dim=-1) / denom                               # [B, N]
    env_stress = per_agent_stress.mean(dim=-1)                                  # [B]
    # Negate so higher = better (PPO maximizes reward)
    return -env_stress, -per_agent_stress



def triangularity_moment(positions: torch.Tensor) -> torch.Tensor:
    """v14: rotation-invariant triangle-shape signature.

    For complex z_i = x_i + i*y_i (centered), the 3rd-order moment scaled by
    individual mass:
        m3 = |mean(z_i^3)| / mean(|z_i|^3)
    has m3 ≈ 0.72 for T4 template, m3 ≈ 0 for circular blob, m3 ≈ 0 for
    any rotationally symmetric distribution. Lying on a line gives high m3 too,
    but combined with spread/edge constraints, this peaks at triangle.

    positions: [B, N, 2] (must be centered; this fn does NOT center)
    Returns: [B] m3 ∈ [0, 1+]
    """
    x = positions[..., 0]
    y = positions[..., 1]
    # z^3 = x^3 - 3xy^2 + i(3x^2 y - y^3)
    real_z3 = x ** 3 - 3 * x * y * y
    imag_z3 = 3 * x * x * y - y ** 3
    mean_real = real_z3.mean(dim=-1)                                      # [B]
    mean_imag = imag_z3.mean(dim=-1)                                      # [B]
    abs_mean_z3 = (mean_real * mean_real + mean_imag * mean_imag).sqrt()  # [B]
    abs_z_cubed = ((x * x + y * y) ** 1.5).mean(dim=-1)                   # [B]
    return abs_mean_z3 / abs_z_cubed.clamp_min(1e-12)
