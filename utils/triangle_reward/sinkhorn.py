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
    log_prob = -cost / temperature
    log_prob = log_prob - log_prob.amax(dim=(-2, -1), keepdim=True)

    for _ in range(int(iters)):
        log_prob = log_prob - torch.logsumexp(log_prob, dim=-1, keepdim=True)
        log_prob = log_prob - torch.logsumexp(log_prob, dim=-2, keepdim=True)

    return torch.exp(log_prob)


def formation_soft_permutation(
    dist2: torch.Tensor,
    template_dist2: torch.Tensor,
    tau: float = 0.1,
    iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    cost = row_signature_cost_matrix(dist2, template_dist2, eps=eps)
    return sinkhorn(cost, tau=tau, iters=iters, eps=eps)


def make_formation_soft_permutation_fn(
    tau: float,
    iters: int,
    eps: float,
    compile_enabled: bool = False,
):
    def soft_perm_fn(dist2: torch.Tensor, template_dist2: torch.Tensor) -> torch.Tensor:
        return formation_soft_permutation(
            dist2,
            template_dist2,
            tau=tau,
            iters=iters,
            eps=eps,
        )

    if not compile_enabled:
        return soft_perm_fn

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("WARNING: torch.compile() is unavailable; leaving formation matching uncompiled.")
        return soft_perm_fn

    try:
        return compile_fn(soft_perm_fn)
    except Exception as e:
        print(f"WARNING: failed to compile formation matching; continuing without compile. Reason: {e}")
        return soft_perm_fn
