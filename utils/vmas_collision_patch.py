from __future__ import annotations

from types import MethodType

import torch
from vmas.simulator.core import Sphere


def patch_world_for_fast_sphere_collisions(world) -> tuple[bool, str]:
    """Install a sphere-only collision fast path on a VMAS World instance.

    The patch is intentionally narrow: it only activates when every collidable
    entity is a sphere and the world has no joints. In that case we can replace
    VMAS's Python pair enumeration + per-pair force updates with one batched
    gather/scatter pass on the simulator device.
    """

    if getattr(world, "_fast_sphere_collision_enabled", False):
        return True, "already enabled"

    entities = list(world.entities)
    if not entities:
        return False, "world has no entities"
    if getattr(world, "_joints", None):
        return False, "joints present"
    if any(not isinstance(entity.shape, Sphere) for entity in entities):
        return False, "non-sphere entity present"

    n_entities = len(entities)
    if n_entities < 2:
        return False, "need at least two entities"

    pair_i = []
    pair_j = []
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            pair_i.append(i)
            pair_j.append(j)

    pair_i_t = torch.tensor(pair_i, device=world.device, dtype=torch.long)
    pair_j_t = torch.tensor(pair_j, device=world.device, dtype=torch.long)
    radii_t = torch.tensor(
        [float(entity.shape.radius) for entity in entities],
        device=world.device,
        dtype=torch.float32,
    )

    def _fast_apply_vectorized_environment_force(self):
        # This fast path only applies to the no-joint sphere-only case.
        self._vectorized_joint_constraints([])

        pos = torch.stack([entity.state.pos for entity in entities], dim=1)  # [B, N, 2]
        pos_a = pos[:, pair_i_t]  # [B, P, 2]
        pos_b = pos[:, pair_j_t]  # [B, P, 2]
        dist_min = radii_t[pair_i_t] + radii_t[pair_j_t]  # [P]
        dist_min = dist_min.unsqueeze(0).expand(self.batch_dim, -1)  # [B, P]

        force_a, force_b = self._get_constraint_forces(
            pos_a,
            pos_b,
            dist_min=dist_min,
            force_multiplier=self._collision_force,
        )

        total_force = torch.zeros(
            self.batch_dim,
            n_entities,
            pos.shape[-1],
            device=self.device,
            dtype=pos.dtype,
        )
        pair_i_index = pair_i_t.view(1, -1, 1).expand(self.batch_dim, -1, pos.shape[-1])
        pair_j_index = pair_j_t.view(1, -1, 1).expand(self.batch_dim, -1, pos.shape[-1])
        total_force.scatter_add_(1, pair_i_index, force_a)
        total_force.scatter_add_(1, pair_j_index, force_b)

        for idx, entity in enumerate(entities):
            if entity.movable:
                self.forces_dict[entity] = self.forces_dict[entity] + total_force[:, idx]

    world._apply_vectorized_enviornment_force = MethodType(
        _fast_apply_vectorized_environment_force,
        world,
    )
    world._fast_sphere_collision_enabled = True
    world._fast_sphere_collision_pair_count = len(pair_i)
    return True, f"sphere-only fast path enabled for {len(pair_i)} pairs"
