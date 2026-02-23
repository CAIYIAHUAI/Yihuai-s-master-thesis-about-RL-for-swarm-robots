from __future__ import annotations

import torch


def scale_invariant_distance_loss(dist_a: torch.Tensor, dist_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant structural loss on upper-triangle distance vectors.

    dist_a: [B,N,N]
    dist_b: [B,N,N]
    returns: [B]
    """
    n = dist_a.shape[1]
    iu = torch.triu_indices(n, n, offset=1, device=dist_a.device)
    vec_a = dist_a[:, iu[0], iu[1]]
    vec_b = dist_b[:, iu[0], iu[1]]

    vec_a = vec_a - vec_a.mean(dim=-1, keepdim=True)
    vec_b = vec_b - vec_b.mean(dim=-1, keepdim=True)

    dot = (vec_a * vec_b).sum(dim=-1)
    norm = torch.linalg.vector_norm(vec_a, dim=-1) * torch.linalg.vector_norm(vec_b, dim=-1)
    cosine = dot / (norm + eps)
    return 1.0 - cosine
