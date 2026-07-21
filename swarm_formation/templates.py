"""Rigid formation templates for self-shape training and evaluation.

The default four names preserve the cleaned v18d N=10 experiments. Additional
N=21/N=28 names support the scaling ladder without changing the rest of the
environment code.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from target import make_t4_template


def _rotate_center(p: torch.Tensor, theta: float) -> torch.Tensor:
    if theta != 0.0:
        c, s = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], dtype=p.dtype, device=p.device)
        p = p @ rot.T
    return p - p.mean(dim=0, keepdim=True)


def make_triangle_rows(rows: int, spacing: float, theta: float = 0.0,
                       device: torch.device | str = "cpu",
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    pts = []
    for r in range(rows):
        for k in range(r + 1):
            pts.append([(k - r / 2.0) * spacing,
                        -r * (math.sqrt(3.0) / 2.0) * spacing])
    return _rotate_center(torch.tensor(pts, dtype=dtype, device=device), theta)


def make_centered_t4_template(spacing: float, theta: float = 0.0,
                              device: torch.device | str = "cpu",
                              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return make_t4_template(spacing, theta=theta, device=device, dtype=dtype)


def make_hex_plus_4_inner(spacing: float, theta: float = 0.0,
                          device: torch.device | str = "cpu",
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    pts = []
    for k in range(6):
        a = k * math.pi / 3.0
        pts.append([spacing * math.cos(a), spacing * math.sin(a)])
    for k in range(4):
        a = k * math.pi / 2.0 + math.pi / 4.0
        pts.append([0.5 * spacing * math.cos(a), 0.5 * spacing * math.sin(a)])
    return _rotate_center(torch.tensor(pts, dtype=dtype, device=device), theta)


def make_hexlike(n: int, spacing: float, theta: float = 0.0,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Hex-like disk with one center and expanding rings, trimmed to n points."""
    pts = [[0.0, 0.0]]
    ring = 1
    while len(pts) < n:
        count = 6 * ring
        radius = ring * spacing
        for k in range(count):
            if len(pts) >= n:
                break
            a = k * 2.0 * math.pi / count
            pts.append([radius * math.cos(a), radius * math.sin(a)])
        ring += 1
    return _rotate_center(torch.tensor(pts, dtype=dtype, device=device), theta)


def make_ring(n: int, spacing: float, theta: float = 0.0,
              device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    radius = spacing / (2.0 * math.sin(math.pi / n))
    pts = [[radius * math.cos(k * 2.0 * math.pi / n),
            radius * math.sin(k * 2.0 * math.pi / n)] for k in range(n)]
    return _rotate_center(torch.tensor(pts, dtype=dtype, device=device), theta)


def make_grid(rows: int, cols: int, spacing: float, theta: float = 0.0,
              device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    pts = []
    for r in range(rows):
        for c in range(cols):
            pts.append([(c - (cols - 1) / 2.0) * spacing,
                        (r - (rows - 1) / 2.0) * spacing])
    return _rotate_center(torch.tensor(pts, dtype=dtype, device=device), theta)


def make_two_row_5(spacing: float, theta: float = 0.0,
                   device: torch.device | str = "cpu",
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return make_grid(2, 5, spacing, theta=theta, device=device, dtype=dtype)


SHAPES: dict[str, Callable[..., torch.Tensor]] = {
    "t4": make_centered_t4_template,
    "hex4": make_hex_plus_4_inner,
    "ring10": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_ring(10, spacing, theta=theta, device=device, dtype=dtype),
    "row5x2": make_two_row_5,
    "t6": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_triangle_rows(6, spacing, theta=theta, device=device, dtype=dtype),
    "hex21": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_hexlike(21, spacing, theta=theta, device=device, dtype=dtype),
    "ring21": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_ring(21, spacing, theta=theta, device=device, dtype=dtype),
    "row7x3": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_grid(3, 7, spacing, theta=theta, device=device, dtype=dtype),
    "t7": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_triangle_rows(7, spacing, theta=theta, device=device, dtype=dtype),
    "hex28": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_hexlike(28, spacing, theta=theta, device=device, dtype=dtype),
    "ring28": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_ring(28, spacing, theta=theta, device=device, dtype=dtype),
    "row7x4": lambda spacing, theta=0.0, device="cpu", dtype=torch.float32:
        make_grid(4, 7, spacing, theta=theta, device=device, dtype=dtype),
}


def make_template(name: str, spacing: float, theta: float = 0.0,
                  device: torch.device | str = "cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if name not in SHAPES:
        raise ValueError(f"Unknown shape '{name}'. Available: {list(SHAPES.keys())}")
    return SHAPES[name](spacing, theta=theta, device=device, dtype=dtype)


def make_target_map(name: str, positions: torch.Tensor, spacing: float,
                    theta: float = 0.0,
                    center_mode: str = "initial_swarm_centroid",
                    centers: torch.Tensor | None = None,
                    thetas: torch.Tensor | None = None) -> torch.Tensor:
    """Build target map for any registered shape."""
    B = positions.shape[0]
    if thetas is not None:
        thetas = thetas.to(device=positions.device, dtype=positions.dtype)
        if thetas.shape != (B,):
            raise ValueError(f"thetas must have shape ({B},), got {tuple(thetas.shape)}")
        template = make_template(name, spacing, theta=0.0,
                                 device=positions.device, dtype=positions.dtype)
        c, s = thetas.cos(), thetas.sin()
        rot = torch.stack(
            [torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)],
            dim=-2,
        )
        rotated = torch.einsum("bij,kj->bki", rot, template)
    else:
        template = make_template(name, spacing, theta=theta,
                                 device=positions.device, dtype=positions.dtype)
        rotated = template.unsqueeze(0).expand(B, -1, -1)

    if centers is not None:
        centers = centers.to(device=positions.device, dtype=positions.dtype)
        if centers.shape != (B, 2):
            raise ValueError(f"centers must have shape ({B}, 2), got {tuple(centers.shape)}")
        center = centers.unsqueeze(1)
    elif center_mode == "initial_swarm_centroid":
        center = positions.mean(dim=1, keepdim=True)
    else:
        center = torch.zeros(B, 1, 2, device=positions.device, dtype=positions.dtype)
    return rotated + center
