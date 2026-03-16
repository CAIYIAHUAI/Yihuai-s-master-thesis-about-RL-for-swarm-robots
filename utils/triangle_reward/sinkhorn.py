from __future__ import annotations

import torch


def row_signature_cost_matrix(dist_a: torch.Tensor, dist_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Build pairwise row-signature cost matrix (scale-invariant).

    Each row's sorted distance signature is normalized by its mean so that
    the comparison depends only on relative distance profiles, not absolute scale.

    dist_a: [B, N, N]
    dist_b: [N, N]
    returns: [B, N, N]
    """
    bsz, n, _ = dist_a.shape
    diag_mask = torch.eye(n, device=dist_a.device, dtype=torch.bool).unsqueeze(0)

    a_masked = dist_a.masked_fill(diag_mask, float("inf"))
    a_sig = torch.sort(a_masked, dim=-1).values[:, :, : n - 1]  # [B,N,N-1]

    dist_b_batch = dist_b.unsqueeze(0).expand(bsz, -1, -1)
    b_masked = dist_b_batch.masked_fill(diag_mask, float("inf"))
    b_sig = torch.sort(b_masked, dim=-1).values[:, :, : n - 1]  # [B,N,N-1]

    a_sig = a_sig / (a_sig.mean(dim=-1, keepdim=True) + eps)
    b_sig = b_sig / (b_sig.mean(dim=-1, keepdim=True) + eps)

    diff = a_sig.unsqueeze(2) - b_sig.unsqueeze(1)  # [B,N,N,N-1]
    return (diff * diff).mean(dim=-1)  # [B,N,N]


def sinkhorn(cost: torch.Tensor, tau: float = 0.1, iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a cost matrix into a soft permutation (doubly stochastic matrix).

    cost: [B, N, N]
    returns: [B, N, N]
    """
    temperature = max(float(tau), float(eps))
    logits = -cost / temperature
    logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
    prob = torch.exp(logits)

    for _ in range(int(iters)):
        prob = prob / (prob.sum(dim=-1, keepdim=True) + eps)
        prob = prob / (prob.sum(dim=-2, keepdim=True) + eps)

    return prob
