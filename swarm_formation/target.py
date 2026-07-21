"""T4 target map: 10 agents, 4 rows, equilateral triangle.

Target locations
    q_{r,k} = ((k - r/2) * d, -r * sqrt(3)/2 * d)   for r=0..3, k=0..r

The map is a *shared task* — assignment-free formation control means each robot
makes its own claim about which slot to occupy, but every robot sees the same
target set. Optimal matching distance and coverage are computed only as
diagnostics / reward-shaping signals; never used to assign robots to slots.
"""
from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment

T4_ROWS = 4                          # 1 + 2 + 3 + 4 = 10
T4_N_AGENTS = sum(range(T4_ROWS + 1))


def make_t4_template(spacing: float, theta: float = 0.0,
                     device: torch.device | str = "cpu",
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return T4 target locations centered at origin, rotated by theta. Shape [10, 2]."""
    pts = []
    for r in range(T4_ROWS):
        for k in range(r + 1):
            x = (k - r / 2.0) * spacing
            y = -r * (math.sqrt(3.0) / 2.0) * spacing
            pts.append((x, y))
    p = torch.tensor(pts, dtype=dtype, device=device)
    if theta != 0.0:
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)
        p = p @ rot.T
    # Recenter on the centroid so `target_center_mode` controls placement uniquely.
    p = p - p.mean(dim=0, keepdim=True)
    return p


def make_t4_target_map(positions: torch.Tensor, spacing: float, theta: float = 0.0,
                       center_mode: str = "initial_swarm_centroid",
                       centers: torch.Tensor | None = None,
                       thetas: torch.Tensor | None = None) -> torch.Tensor:
    """Build the shared T4 target map.

    Args:
        positions: [B, N, 2] initial robot positions.
        spacing: side length d.
        theta: rotation in rad, used when `thetas` is not provided.
        center_mode: "initial_swarm_centroid" places the template at the
            per-batch centroid of `positions`. Anything else places at origin.
        centers: optional [B, 2] target centroids. Overrides `center_mode`.
        thetas: optional [B] per-batch rotations. Overrides `theta`.
    Returns:
        targets: [B, M=10, 2] target positions per batch.
    """
    B = positions.shape[0]
    if thetas is not None:
        thetas = thetas.to(device=positions.device, dtype=positions.dtype)
        if thetas.shape != (B,):
            raise ValueError(f"thetas must have shape ({B},), got {tuple(thetas.shape)}")
        template = make_t4_template(spacing, 0.0, device=positions.device,
                                    dtype=positions.dtype)        # [10, 2]
        c, s = thetas.cos(), thetas.sin()
        rot = torch.stack(
            [
                torch.stack([c, -s], dim=-1),
                torch.stack([s, c], dim=-1),
            ],
            dim=-2,
        )                                                        # [B, 2, 2]
        rotated = torch.einsum("bij,kj->bki", rot, template)     # [B, 10, 2]
    else:
        template = make_t4_template(spacing, theta, device=positions.device,
                                    dtype=positions.dtype)        # [10, 2]
        rotated = template.unsqueeze(0).expand(B, -1, -1)

    if centers is not None:
        centers = centers.to(device=positions.device, dtype=positions.dtype)
        if centers.shape != (B, 2):
            raise ValueError(f"centers must have shape ({B}, 2), got {tuple(centers.shape)}")
        center = centers.unsqueeze(1)                             # [B, 1, 2]
    elif center_mode == "initial_swarm_centroid":
        center = positions.mean(dim=1, keepdim=True)              # [B, 1, 2]
    else:
        center = torch.zeros(B, 1, 2, device=positions.device, dtype=positions.dtype)
    return rotated + center                                      # [B, 10, 2]


def pairwise_distance(positions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[B, N, 2], [B, M, 2] -> [B, N, M] euclidean distances."""
    diff = positions.unsqueeze(2) - targets.unsqueeze(1)
    return diff.norm(dim=-1)


def compute_matching(positions: torch.Tensor, targets: torch.Tensor,
                     return_per_agent: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal robot↔target assignment per batch (Hungarian).

    Args:
        positions: [B, N, 2]
        targets:   [B, N, 2] (must have N == M)
        return_per_agent: if True, second return is per-agent matched distance
            tensor of shape [B, N]. If False (default), returns *mean* per-robot
            distance [B] for backward compat (thresholds like "match < 0.4*d"
            are naturally per-robot).

    Note: matching is a *diagnostic* / reward signal only — never used to assign
    robots to slots in the policy.
    """
    B, N, _ = positions.shape
    M = targets.shape[1]
    assert N == M, f"matching expects N == M, got N={N} M={M}"
    d = pairwise_distance(positions, targets).cpu().numpy()
    perm = torch.empty(B, N, dtype=torch.long, device=positions.device)
    if return_per_agent:
        per_agent = torch.empty(B, N, dtype=positions.dtype, device=positions.device)
        for b in range(B):
            row, col = linear_sum_assignment(d[b])
            perm[b] = torch.from_numpy(col).to(perm.device)
            per_agent[b] = torch.from_numpy(d[b, row, col]).to(per_agent.device).to(per_agent.dtype)
        return perm, per_agent
    dist = torch.empty(B, dtype=positions.dtype, device=positions.device)
    for b in range(B):
        row, col = linear_sum_assignment(d[b])
        perm[b] = torch.from_numpy(col).to(perm.device)
        dist[b] = float(d[b, row, col].mean())
    return perm, dist


def compute_coverage(positions: torch.Tensor, targets: torch.Tensor,
                     radius: float) -> torch.Tensor:
    """Fraction of target slots that have at least one robot within `radius`.
    Returns [B] float in [0, 1].
    """
    d = pairwise_distance(positions, targets)              # [B, N, M]
    covered = (d.min(dim=1).values < radius).float()        # [B, M]
    return covered.mean(dim=-1)
