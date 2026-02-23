from __future__ import annotations

import math

import torch


def _van_der_corput(n: int, base: int) -> float:
    value = 0.0
    denom = 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        value += rem / denom
    return value


def build_triangle_template(n_agents: int, seed: int = 0, side_length: float = 1.0) -> torch.Tensor:
    if n_agents < 3:
        raise ValueError("Triangle template requires n_agents >= 3.")

    side = float(side_length)
    v0 = torch.tensor([0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([side, 0.0], dtype=torch.float32)
    v2 = torch.tensor([0.5 * side, (math.sqrt(3.0) / 2.0) * side], dtype=torch.float32)

    points = [v0, v1, v2]
    if n_agents == 3:
        return torch.stack(points, dim=0)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    shift = torch.rand((2,), generator=rng, dtype=torch.float32)

    k = 1
    while len(points) < n_agents:
        u = (_van_der_corput(k, 2) + float(shift[0])) % 1.0
        v = (_van_der_corput(k, 3) + float(shift[1])) % 1.0
        r1 = math.sqrt(u)
        r2 = v
        p = (1.0 - r1) * v0 + r1 * (1.0 - r2) * v1 + r1 * r2 * v2
        points.append(p)
        k += 1

    return torch.stack(points, dim=0)
