from __future__ import annotations

import torch


def squared_distance_matrix_batched(points: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, d]
    returns: [B, N, N] with zero diagonal
    """
    sq_norm = (points * points).sum(dim=-1, keepdim=True)  # [B,N,1]
    dist2 = sq_norm + sq_norm.transpose(1, 2) - 2.0 * torch.matmul(points, points.transpose(1, 2))
    dist2 = torch.clamp_min(dist2, 0.0)
    dist2 = dist2 - torch.diag_embed(torch.diagonal(dist2, dim1=1, dim2=2))
    return dist2


def upper_triangle_indices(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(n, n, offset=1, device=device)
