from __future__ import annotations

import math
import torch


def _point_to_segment_dist(pts: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ap = pts - a
    t = (ap @ ab) / (ab @ ab + 1e-12)
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return (pts - closest).norm(dim=-1)


def _inside_triangle(pts: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    d = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    a = ((v1[1] - v2[1]) * (pts[:, 0] - v2[0]) + (v2[0] - v1[0]) * (pts[:, 1] - v2[1])) / d
    b = ((v2[1] - v0[1]) * (pts[:, 0] - v2[0]) + (v0[0] - v2[0]) * (pts[:, 1] - v2[1])) / d
    c = 1.0 - a - b
    margin = -1e-6
    return (a >= margin) & (b >= margin) & (c >= margin)


def _generate_hex_in_triangle(
    d0: float, side_length: float, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
) -> torch.Tensor:
    row_h = d0 * math.sqrt(3.0) / 2.0
    pts = []
    j = 0
    ymax = side_length * math.sqrt(3.0) / 2.0 + d0
    while j * row_h <= ymax:
        y = j * row_h
        x_off = d0 / 2.0 if j % 2 == 1 else 0.0
        x = x_off
        while x <= side_length + d0:
            pts.append([x, y])
            x += d0
        j += 1
    if not pts:
        return torch.zeros((0, 2), dtype=torch.float32)
    pts_t = torch.tensor(pts, dtype=torch.float32)
    return pts_t[_inside_triangle(pts_t, v0, v1, v2)]


def build_triangle_template(n_agents: int, seed: int = 0, side_length: float = 1.0) -> torch.Tensor:
    del seed  # Kept for API compatibility; template is deterministic.
    if n_agents < 3:
        raise ValueError("n_agents must be >= 3")

    side = float(side_length)
    v0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([side, 0.0], dtype=torch.float32)
    v2 = torch.tensor([side / 2.0, side * math.sqrt(3.0) / 2.0], dtype=torch.float32)
    edges = [(v0, v1), (v1, v2), (v2, v0)]

    lo, hi = side / (2.0 * n_agents), side / 2.0
    best_pts = None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        pts = _generate_hex_in_triangle(mid, side, v0, v1, v2)
        if pts.shape[0] >= n_agents:
            best_pts = pts
            lo = mid
        else:
            hi = mid

    if best_pts is None or best_pts.shape[0] < n_agents:
        raise RuntimeError(f"could not generate {n_agents} points in triangle")

    if best_pts.shape[0] > n_agents:
        min_edge_dist = torch.full((best_pts.shape[0],), float("inf"), dtype=best_pts.dtype)
        for a, b in edges:
            d = _point_to_segment_dist(best_pts, a, b)
            min_edge_dist = torch.minimum(min_edge_dist, d)
        keep = min_edge_dist.argsort()[:n_agents]
        best_pts = best_pts[keep]

    pts = best_pts - best_pts.mean(dim=0, keepdim=True)
    max_ext = pts.abs().max()
    if max_ext > 1e-8:
        pts = pts / max_ext * (side / 2.0)
    return pts
