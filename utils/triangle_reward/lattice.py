from __future__ import annotations

import torch


def _normalized_knn_signatures(dist_mat: torch.Tensor, k: int, eps: float) -> torch.Tensor:
    n = dist_mat.shape[-1]
    actual_k = min(int(k), n - 1)
    if actual_k <= 0:
        raise ValueError("k must be at least 1 when n >= 2")

    diag = torch.eye(n, device=dist_mat.device, dtype=torch.bool)
    masked = dist_mat.masked_fill(diag, float("inf"))
    knn = torch.topk(masked, k=actual_k, dim=-1, largest=False, sorted=True).values
    return knn / (knn.mean(dim=-1, keepdim=True) + eps)


def build_template_knn_signatures(
    template_points: torch.Tensor,
    k: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pre-compute normalized KNN distance signatures for template points."""
    if template_points.ndim != 2 or template_points.shape[-1] != 2:
        raise ValueError("template_points must have shape [N, 2]")
    dist_mat = torch.cdist(template_points, template_points)
    return _normalized_knn_signatures(dist_mat, k=k, eps=eps)


def per_agent_lattice_loss(
    dist_mat: torch.Tensor,
    template_sigs: torch.Tensor,
    k: int = 6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Each agent's local lattice loss as the best-matching template signature."""
    if dist_mat.ndim != 3 or dist_mat.shape[-1] != dist_mat.shape[-2]:
        raise ValueError("dist_mat must have shape [B, N, N]")
    if template_sigs.ndim != 2:
        raise ValueError("template_sigs must have shape [N, K]")

    agent_sigs = _normalized_knn_signatures(dist_mat, k=k, eps=eps)
    if agent_sigs.shape[-1] != template_sigs.shape[-1]:
        raise ValueError("template_sigs K does not match dist_mat K")

    diff = agent_sigs.unsqueeze(2) - template_sigs.unsqueeze(0).unsqueeze(0)
    match_costs = diff.square().mean(dim=-1)
    return match_costs.min(dim=-1).values
