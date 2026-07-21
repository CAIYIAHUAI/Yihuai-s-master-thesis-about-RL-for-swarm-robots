"""Mean-shift environment + expert sanity rollout + sharded demo generation.

Env: vectorized 2D unicycle over B parallel batches. Actions are
    STOP=0    no change
    LEFT=1    heading += rotate_step
    RIGHT=2   heading -= rotate_step
    FORWARD=3 position += v_max * dt * [cos(heading), sin(heading)]

CLI
    --sanity            run expert for N episodes and report metrics
    --gen --out PATH    generate demos to PATH (creates `PATH.shards/` directory)

The shard / manifest framework is borrowed from
`yihuai-thesis-v2/scripts/generate_demos.py` (training infrastructure only).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from observations import (
    ACT_FORWARD, ACT_LEFT, ACT_RIGHT, ACT_STOP,
    EDGE_FEATURE_DIM, NODE_FEATURE_DIM,
    ExpertOutput, ExpertState, assert_time_scales, expert_step,
)
from target import compute_coverage, compute_matching


ARCH_VERSION = "triangle_ms_v1"


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    return cfg


# ---------------------------------------------------------------------------
# Vectorized env
# ---------------------------------------------------------------------------
@dataclass
class StepInfo:
    matching_distance: torch.Tensor    # [B] mean per-robot optimal-matching distance
    coverage: torch.Tensor             # [B] in [0, 1]
    collision_count: torch.Tensor      # [B] int — pairs in collision_hard radius this step (state-based)
    swap_event_count: torch.Tensor     # [B] int — accumulated true 2-cycle nearest-target swaps
    success: torch.Tensor              # [B] bool

    # Self-shape mode (R1-R17) — Optional, populated by env.compute_shape_metrics()
    shape_error: torch.Tensor | None = None             # [B] per-step profile error (R4)
    shape_match_distance: torch.Tensor | None = None    # [B] best-rotation Hungarian dist (slow, periodic)
    isolated_count: torch.Tensor | None = None          # [B] # agents with < min_neighbors (R6b)
    bond_penalty: torch.Tensor | None = None            # [B] mean per-edge deviation from T4 bonds (R2)
    success_shape: torch.Tensor | None = None           # [B] bool, shape_match_distance < threshold


class MeanShiftEnv:
    """B parallel N-agent unicycle envs sharing one target map per env."""

    def __init__(self, cfg: dict, n_envs: int, device: torch.device | str = "cpu",
                 init_radius: float | None = None, seed: int | None = None):
        n_agents = int(cfg["n_agents"])
        assert n_agents >= 4, f"swarm_formation expects n_agents >= 4, cfg has {n_agents}"
        assert_time_scales(cfg)
        self.cfg = cfg
        self.B = int(n_envs)
        self.N = n_agents
        self.M = n_agents
        self.device = torch.device(device)
        self.init_radius = float(init_radius if init_radius is not None
                                 else 1.5 * float(cfg["target_spacing"]))
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        self.dt = float(cfg["dt"])
        self.v_max = float(cfg["v_max"])
        self.rotate_step = float(cfg["rotate_step"])
        self.spacing = float(cfg["target_spacing"])
        self.success_dist = float(cfg["success_matching_distance"]) * self.spacing
        self.collision_hard = float(cfg["collision_hard"])
        self.episode_length = int(cfg["episode_length"])

        # state
        self.positions = torch.zeros(self.B, self.N, 2, device=self.device)
        self.rotations = torch.zeros(self.B, self.N, device=self.device)
        self.velocity = torch.zeros(self.B, self.N, 2, device=self.device)
        self.target_map = torch.zeros(self.B, self.M, 2, device=self.device)
        self.last_action = torch.full((self.B, self.N), float(ACT_STOP),
                                      dtype=torch.long, device=self.device)
        self.expert_state = ExpertState.zeros(self.B, self.N, self.M, device=self.device)
        self.stuck_counter = torch.zeros(self.B, self.N, dtype=torch.long, device=self.device)

        # R1: per-episode persistent latent. Resampled at every reset, constant
        # within an episode. Always allocated (zero in target/legacy mode); only
        # actually consumed when caller passes it through build_node_features /
        # build_edges with cue_off / latent_z paths.
        self.latent_z_dim = int(cfg.get("latent_z_dim", 4))
        self.latent_z = torch.zeros(self.B, self.N, self.latent_z_dim, device=self.device)

        # R1+R12: per-episode binary cue dropout decision. True = actor sees target
        # cue this episode; False = strict target-blind. Resampled at every reset.
        # Driven by curriculum (cue_prob), set externally via set_cue_present().
        self.cue_present = torch.ones(self.B, dtype=torch.bool, device=self.device)

        # v9 FIX #11: per-env EMA Procrustes theta (for per-agent match reward stability).
        # Reset to 0 at episode start; updated by compute_per_agent_metrics.
        self.match_prev_theta = torch.zeros(self.B, device=self.device)

        # v18d: per-episode z-driven slot assignment via rank(z[..., 0]). Computed
        # once at every reset_envs(idx). Used by formation_stress_reward.
        self.slot_assignment = torch.zeros(self.B, self.N, dtype=torch.long,
                                            device=self.device)

        # diagnostics
        self.t = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._prev_argmin = torch.zeros(self.B, self.N, dtype=torch.long, device=self.device)
        self._cum_swap = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._cum_collision = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Multi-shape: shape_pool drives per-env per-reset shape sampling.
        # Default ["t4"] preserves single-shape T4 behavior (backward compat).
        from shape_metrics import pairwise_dist as _pdist
        from templates import make_template
        shape_pool = list(self.cfg.get("shape_pool", ["t4"]))
        self._shape_names = shape_pool
        templates = [make_template(name, self.spacing, device=self.device)
                     for name in shape_pool]
        for name, template in zip(shape_pool, templates):
            if template.shape[0] != self.N:
                raise ValueError(
                    f"shape {name!r} has {template.shape[0]} points, "
                    f"but cfg.n_agents={self.N}"
                )
        self._cached_pair_dists = torch.stack([
            _pdist(template) for template in templates
        ], dim=0)                                                            # [K, N, N]
        # v19: also cache centered template coordinates normalized by spacing.
        # Used by build_selfshape_obs when slot_repr=template_coords to embed
        # each agent's slot as a 2-D position (N-independent obs schema).
        self._cached_templates_norm = torch.stack([
            (template - template.mean(dim=0, keepdim=True)) / self.spacing
            for template in templates
        ], dim=0)                                                            # [K, N, 2]
        self.slot_pair_dist = torch.zeros(self.B, self.M, self.M,
                                           device=self.device)
        # v19: per-env normalized template coords [B, N, 2], set on reset to
        # match shape_id. Slot embed = gather(_cached_templates_norm, slot).
        self.slot_template_norm = torch.zeros(self.B, self.M, 2,
                                              device=self.device)
        self.shape_id = torch.zeros(self.B, dtype=torch.long, device=self.device)

        self.reset_all()

    # -- init --
    def _sample_init(self, idx: torch.Tensor) -> None:
        """Sample positions/rotations for envs in `idx`."""
        n = idx.numel()
        init_mode = str(self.cfg.get("robot_init_mode", "gaussian_recentered"))
        if init_mode == "origin_jitter":
            jitter = float(self.cfg.get("robot_init_jitter", 0.015))
            positions = self._sample_origin_jitter(n, jitter)
        else:
            # Legacy: sample a tight Gaussian blob and recenter it at its own centroid.
            positions = torch.randn(n, self.N, 2, device=self.device,
                                    generator=self.gen) * self.init_radius
            positions = positions - positions.mean(dim=1, keepdim=True)
        rotations = (torch.rand(n, self.N, device=self.device, generator=self.gen) * 2 - 1) * math.pi
        self.positions[idx] = positions
        self.rotations[idx] = rotations
        self.velocity[idx] = 0.0
        self.last_action[idx] = ACT_STOP
        self.t[idx] = 0
        self.expert_state.claim_count[idx] = 0.0
        self.expert_state.blocked_steps[idx] = 0
        self.stuck_counter[idx] = 0

        centers = None
        thetas = None
        center_mode = str(self.cfg.get("target_center_mode", "initial_swarm_centroid"))
        if center_mode == "random_disc":
            offset_max = float(self.cfg.get("target_offset_max", 0.0))
            r = offset_max * torch.rand(n, device=self.device,
                                        generator=self.gen).sqrt()
            phi = 2.0 * math.pi * torch.rand(n, device=self.device,
                                             generator=self.gen)
            centers = torch.stack([r * phi.cos(), r * phi.sin()], dim=-1)
        if bool(self.cfg.get("target_theta_random", False)):
            thetas = (torch.rand(n, device=self.device, generator=self.gen) * 2 - 1) * math.pi

        # Sample shape for each env being reset.
        shape_idx = torch.randint(0, len(self._shape_names), (n,),
                                   device=self.device, generator=self.gen)
        self.shape_id[idx] = shape_idx
        self.slot_pair_dist[idx] = self._cached_pair_dists[shape_idx]        # [n, N, N]
        # v19: pick per-env template coords for slot_repr=template_coords.
        self.slot_template_norm[idx] = self._cached_templates_norm[shape_idx]  # [n, N, 2]

        # Build target_map per-shape. positions/centers/thetas are sized [n, ...]
        # so all per-shape sub-batches must be sliced with `mask` consistently.
        from templates import make_target_map
        tm = torch.empty(n, self.M, 2, device=self.device, dtype=positions.dtype)
        for k, name in enumerate(self._shape_names):
            mask = (shape_idx == k)
            if not mask.any():
                continue
            centers_k = centers[mask] if centers is not None else None
            thetas_k  = thetas[mask]  if thetas  is not None else None
            tm[mask] = make_target_map(
                name, positions[mask], spacing=self.spacing,
                theta=float(self.cfg.get("target_theta", 0.0)),
                center_mode=center_mode,
                centers=centers_k, thetas=thetas_k,
            )
        self.target_map[idx] = tm
        # Reset swap tracker by recording the initial argmin.
        d = (positions.unsqueeze(2) - tm.unsqueeze(1)).norm(dim=-1)
        self._prev_argmin[idx] = d.argmin(dim=-1)
        self._cum_swap[idx] = 0
        self._cum_collision[idx] = 0

        # R1: resample per-episode latent. iid Gaussian across agents, no
        # cross-episode identity (each reset draws fresh).
        self.latent_z[idx] = torch.randn(n, self.N, self.latent_z_dim,
                                          device=self.device, generator=self.gen)
        # v9 #11: reset EMA Procrustes theta to 0 — first step uses full grid search
        self.match_prev_theta[idx] = 0.0
        # v12: recompute slot_assignment from fresh z (rank or frozen ψ + Hungarian).
        # Frozen for the entire episode — drives role_pair_distance_reward.
        self.slot_assignment[idx] = self._compute_slot_assignment(self.latent_z[idx])
        # R3+R12: cue_present is NOT touched here; caller controls it via
        # set_cue_present_prob() after reset (otherwise we'd clobber the
        # curriculum-driven dropout decision).

    def _compute_slot_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """v18d: z -> slot index per agent via rank(z[..., 0]). Returns [B, N] long.

        Double argsort gives the rank: for each agent i, slot[i] =
        #(j: z[j, 0] < z[i, 0]). Guarantees a valid permutation (each slot gets
        exactly one agent).
        """
        return z[..., 0].argsort(dim=-1).argsort(dim=-1)

    def _sample_origin_jitter(self, n: int, jitter: float) -> torch.Tensor:
        """Sample a compact origin spawn while avoiding initial hard collisions."""
        positions = torch.empty(n, self.N, 2, device=self.device)
        min_sep = max(1.05 * self.collision_hard, 1e-6)
        max_radius = max(5.0 * jitter, min_sep)
        for b in range(n):
            placed = 0
            attempts = 0
            while placed < self.N:
                cand = torch.randn(2, device=self.device, generator=self.gen) * jitter
                cand_norm = cand.norm()
                if cand_norm > max_radius:
                    cand = cand * (max_radius / cand_norm.clamp_min(1e-9))
                if placed == 0:
                    ok = True
                else:
                    sep = (positions[b, :placed] - cand).norm(dim=-1)
                    ok = bool((sep >= min_sep).all().item())
                if ok:
                    positions[b, placed] = cand
                    placed += 1
                attempts += 1
                if attempts > 1000:
                    positions[b] = self._origin_jitter_fallback(jitter, min_sep)
                    break
        return positions

    def _origin_jitter_fallback(self, jitter: float, min_sep: float) -> torch.Tensor:
        side = math.ceil(math.sqrt(self.N))
        axis = torch.arange(side, device=self.device, dtype=self.positions.dtype)
        axis = axis - axis.mean()
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:self.N]
        grid = grid * (1.1 * min_sep)
        angle = 2.0 * math.pi * torch.rand((), device=self.device, generator=self.gen)
        c, s = angle.cos(), angle.sin()
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        return grid @ rot.T

    def reset_all(self) -> None:
        idx = torch.arange(self.B, device=self.device)
        self._sample_init(idx)

    def reset_envs(self, mask: torch.Tensor) -> None:
        if not mask.any():
            return
        idx = mask.nonzero(as_tuple=True)[0]
        self._sample_init(idx)

    # -- step --
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, StepInfo, torch.Tensor]:
        """action:
          - [B, N] long: discrete (legacy) — STOP/LEFT/RIGHT/FORWARD
          - [B, N, 2] float: continuous — (linear_signal, angular_signal) ∈ [-1, 1]
        Returns (velocity_world [B, N, 2], info, done [B] bool).
        """
        if action.dtype.is_floating_point and action.dim() == 3 and action.shape[-1] == 2:
            # v16: continuous action path
            action_clamped = action.clamp(-1.0, 1.0)
            linear_sig = action_clamped[..., 0]                            # [B, N] in [-1, 1]
            angular_sig = action_clamped[..., 1]                           # [B, N] in [-1, 1]
            # Apply angular: scale to actual rotation rate
            new_rot = self.rotations + angular_sig * self.rotate_step
            new_rot = (new_rot + math.pi) % (2 * math.pi) - math.pi
            # Apply linear: forward speed in NEW heading
            v_forward = linear_sig * self.v_max                            # signed (allow backward)
            v_world = torch.zeros_like(self.positions)
            v_world[..., 0] = v_forward * torch.cos(new_rot)
            v_world[..., 1] = v_forward * torch.sin(new_rot)
            new_pos = self.positions + v_world * self.dt
            # Cache pseudo-discrete last_action for legacy diagnostic compatibility:
            # use FORWARD if linear>0.3, else STOP (rough mapping).
            pseudo_action = torch.where(linear_sig > 0.3,
                                          torch.full_like(linear_sig, float(ACT_FORWARD)),
                                          torch.full_like(linear_sig, float(ACT_STOP))).long()
            self.last_action = pseudo_action
            is_forward = linear_sig > 0.3                                   # for stuck_counter logic
        else:
            # Discrete action path (legacy)
            is_left = action == ACT_LEFT
            is_right = action == ACT_RIGHT
            is_forward = action == ACT_FORWARD
            new_rot = self.rotations.clone()
            new_rot[is_left] = self.rotations[is_left] + self.rotate_step
            new_rot[is_right] = self.rotations[is_right] - self.rotate_step
            new_rot = (new_rot + math.pi) % (2 * math.pi) - math.pi
            v_world = torch.zeros_like(self.positions)
            v_world[..., 0] = is_forward.float() * self.v_max * torch.cos(new_rot)
            v_world[..., 1] = is_forward.float() * self.v_max * torch.sin(new_rot)
            new_pos = self.positions + v_world * self.dt
            self.last_action = action

        self.positions = new_pos
        self.rotations = new_rot
        self.velocity = v_world
        self.t = self.t + 1

        # diagnostics ---------------------------------------------------------
        d_im = (self.positions.unsqueeze(2) - self.target_map.unsqueeze(1)).norm(dim=-1)
        argmin = d_im.argmin(dim=-1)                                # [B, N]
        # True swap event: a pair (i, j) such that argmin_t[i]==argmin_{t-1}[j]
        # AND argmin_t[j]==argmin_{t-1}[i] AND argmin_t[i] != argmin_{t-1}[i].
        # Counts each unordered swap once per env.
        a_now = argmin.unsqueeze(2)                                 # [B, N, 1]
        a_prev = self._prev_argmin.unsqueeze(1)                     # [B, 1, N]
        swap_pair = (a_now == a_prev) & \
                    (argmin.unsqueeze(1) == self._prev_argmin.unsqueeze(2)) & \
                    (argmin.unsqueeze(2) != self._prev_argmin.unsqueeze(2))
        # zero out diagonal i==j
        swap_pair = swap_pair & (~torch.eye(self.N, dtype=torch.bool,
                                            device=self.device).unsqueeze(0))
        swap_now = swap_pair.sum(dim=(-2, -1)) // 2                # unordered pairs
        self._cum_swap = self._cum_swap + swap_now
        self._prev_argmin = argmin

        d_ij = (self.positions.unsqueeze(2) - self.positions.unsqueeze(1)).norm(dim=-1)
        n_self = ~torch.eye(self.N, dtype=torch.bool, device=self.device).unsqueeze(0)
        coll_now = ((d_ij < self.collision_hard) & n_self).sum(dim=(-2, -1)) // 2
        self._cum_collision = self._cum_collision + coll_now

        # stuck counter: bumped when this step is non-FORWARD or near-zero translation.
        moved = is_forward & (v_world.norm(dim=-1) > 0.5 * self.v_max)
        self.stuck_counter = torch.where(moved, torch.zeros_like(self.stuck_counter),
                                         self.stuck_counter + 1)

        # matching for reward / success
        _perm, match_dist = compute_matching(self.positions, self.target_map)
        coverage = compute_coverage(self.positions, self.target_map,
                                    radius=float(self.cfg["arrival_radius"]))
        success = match_dist < self.success_dist
        info = StepInfo(matching_distance=match_dist, coverage=coverage,
                        collision_count=coll_now, swap_event_count=self._cum_swap,
                        success=success)

        done = self.t >= self.episode_length
        return v_world, info, done

    # -- self-shape helpers (R1, R2, R3, R4, R6b) --
    def set_cue_present_prob(self, cue_prob: float, mask: torch.Tensor | None = None) -> None:
        """Set per-env cue_present flag based on cue_prob.

        Call this AFTER reset_envs/reset_all (reset preserves the previous flag
        deliberately so the caller has full control over curriculum dropout).
        mask: bool [B] of which envs to update; default = all envs.
        """
        if mask is None:
            mask = torch.ones(self.B, dtype=torch.bool, device=self.device)
        if not mask.any():
            return
        idx = mask.nonzero(as_tuple=True)[0]
        n = idx.numel()
        if cue_prob >= 1.0:
            self.cue_present[idx] = True
        elif cue_prob <= 0.0:
            self.cue_present[idx] = False
        else:
            r = torch.rand(n, device=self.device, generator=self.gen)
            self.cue_present[idx] = r < cue_prob

    def compute_shape_metrics(self, template: torch.Tensor,
                              alpha: float = 0.5,
                              theta_bins: int | None = None,
                              compute_match_distance: bool = False
                              ) -> tuple[torch.Tensor, torch.Tensor | None,
                                         torch.Tensor, torch.Tensor]:
        """Self-shape metrics for current env state.

        Returns (profile_error[B], match_dist[B] or None, isolated_count[B],
                 bond_penalty[B]).
        match_distance is slow (B*theta_bins Hungarians) — only compute when
        explicitly requested (eval/episode-end), not per step.
        """
        from shape_metrics import (shape_profile_error, shape_match_distance_grid,
                                    bond_length_penalty, isolated_agent_count)
        prof = shape_profile_error(self.positions, template, alpha=alpha)
        match = None
        if compute_match_distance:
            tb = int(theta_bins or self.cfg.get("shape_match_theta_bins", 72))
            match = shape_match_distance_grid(self.positions, template, theta_bins=tb)
        iso = isolated_agent_count(self.positions,
                                    comm_radius=float(self.cfg["comm_radius"]),
                                    min_neighbors=int(self.cfg.get("connectivity_min_neighbors", 1)))
        bp = bond_length_penalty(self.positions,
                                  comm_radius=float(self.cfg["comm_radius"]),
                                  spacing=self.spacing,
                                  bond_lengths_d=list(self.cfg.get(
                                      "bond_length_ratios",
                                      [1.0, math.sqrt(3), 2.0, math.sqrt(7), 3.0])))
        return prof, match, iso, bp

    def compute_shape_metrics_multi(self, templates: dict,
                                     alpha: float = 0.5,
                                     theta_bins: int | None = None,
                                     compute_match_distance: bool = False
                                     ) -> tuple[torch.Tensor, torch.Tensor | None,
                                                torch.Tensor, torch.Tensor]:
        """Multi-shape version of compute_shape_metrics. Groups envs by self.shape_id
        and uses the corresponding template for each group. `templates` keys must
        match self._shape_names order.

        Returns same tuple shapes as compute_shape_metrics, computed per-shape.
        bond_penalty here uses per-shape pairwise distances (auto-derived from
        slot_pair_dist) — see shape_metrics.bond_length_penalty docs.
        """
        from shape_metrics import (shape_profile_error, shape_match_distance_grid,
                                    bond_length_penalty, isolated_agent_count)
        B = self.B
        device = self.device
        dtype = self.positions.dtype
        prof = torch.zeros(B, device=device, dtype=dtype)
        bp = torch.zeros(B, device=device, dtype=dtype)
        match = (torch.zeros(B, device=device, dtype=dtype)
                 if compute_match_distance else None)
        # isolated_agent_count is shape-agnostic — compute on full batch.
        iso = isolated_agent_count(
            self.positions,
            comm_radius=float(self.cfg["comm_radius"]),
            min_neighbors=int(self.cfg.get("connectivity_min_neighbors", 1)),
        )
        tb = int(theta_bins or self.cfg.get("shape_match_theta_bins", 72))
        comm_r = float(self.cfg["comm_radius"])
        for k, name in enumerate(self._shape_names):
            mask = (self.shape_id == k)
            if not mask.any():
                continue
            if name not in templates:
                raise KeyError(f"templates missing entry for shape '{name}'")
            tmpl = templates[name]
            pos_k = self.positions[mask]
            prof[mask] = shape_profile_error(pos_k, tmpl, alpha=alpha)
            # bond_length_penalty uses the shape's own pairwise distances as
            # the desired bond lengths (auto-derived; see shape_metrics).
            bp[mask] = bond_length_penalty(
                pos_k, comm_radius=comm_r, spacing=self.spacing,
                template=tmpl,
            )
            if compute_match_distance:
                match[mask] = shape_match_distance_grid(pos_k, tmpl, theta_bins=tb)
        return prof, match, iso, bp

    def compute_per_agent_metrics(self, template: torch.Tensor,
                                   match_n_iters: int = 3,
                                   match_init_theta_bins: int = 12,
                                   use_theta_ema: bool = True,
                                   theta_ema_alpha: float = 0.7,
                                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """v8 per-agent metrics for asymmetric reward signals (R4 fix).

        v9 FIX #11: when `use_theta_ema=True`, seed Procrustes init from
        `self.match_prev_theta` (last step's best-fit). After computation,
        EMA-update self.match_prev_theta to stabilize cross-step assignment.
        Combats Procrustes oscillation (P1 finding: 21-38% steps had agent
        reassignment, max 161° rotation jumps).

        Returns:
          per_agent_bond [B, N]:  mean per-incident-edge deviation in d-units
          per_agent_coll [B, N]:  count of close (< collision_hard) neighbors
          per_agent_match [B, N]: distance to Procrustes-Hungarian-assigned slot,
                                   in absolute units (NOT d-units).
        """
        from shape_metrics import (per_agent_bond_penalty, per_agent_collision_count,
                                    per_agent_match_distance_iterated)
        bp = per_agent_bond_penalty(
            self.positions,
            comm_radius=float(self.cfg["comm_radius"]),
            spacing=self.spacing,
            bond_lengths_d=list(self.cfg.get(
                "bond_length_ratios",
                [1.0, math.sqrt(3), 2.0, math.sqrt(7), 3.0])),
        )
        cc = per_agent_collision_count(self.positions, collision_hard=self.collision_hard)
        if use_theta_ema:
            # First step of episode (prev_theta=0): use full grid; subsequent: narrow window
            is_first = (self.match_prev_theta == 0).all().item()
            md, new_theta = per_agent_match_distance_iterated(
                self.positions, template,
                n_iters=match_n_iters,
                init_theta_bins=match_init_theta_bins,
                prev_theta=None if is_first else self.match_prev_theta,
                return_theta=True,
            )
            # EMA update: smooth jumps (alpha low → smoother)
            if is_first:
                self.match_prev_theta = new_theta
            else:
                self.match_prev_theta = (theta_ema_alpha * self.match_prev_theta
                                         + (1 - theta_ema_alpha) * new_theta)
        else:
            md = per_agent_match_distance_iterated(
                self.positions, template,
                n_iters=match_n_iters,
                init_theta_bins=match_init_theta_bins,
            )
        return bp, cc, md


    def compute_stress_reward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """v18: strictly local formation stress reward.

        Negated stress potential V(p) = Σ_{(i,j)∈E_loc} (||p_i-p_j||-d_ij)²
        with E_loc = neighbors within comm_radius. Uses cached
        `self.slot_pair_dist` (active-template pairwise distances). This is the
        same potential the v17a hand-coded controller minimizes via spring
        forces — so PPO gradient on this reward ≈ controller's gradient.

        Returns:
            env_score [B]:    -mean per-agent stress (≤ 0; 0 = perfect shape in
                              the local graph, up to SE(2)+reflection).
            per_agent [B, N]: -per-agent local stress (≤ 0).
        """
        from shape_metrics import formation_stress_reward
        return formation_stress_reward(
            self.positions, self.slot_assignment, self.slot_pair_dist,
            comm_radius=float(self.cfg["comm_radius"]),
            spacing=self.spacing,
            stress_norm=str(self.cfg.get("stress_norm", "fixed_n_minus_1")),
        )


    # -- expert convenience --
    def expert_action(self, noise_gen: torch.Generator | None,
                      build_obs: bool = True) -> ExpertOutput:
        out = expert_step(
            positions=self.positions, rotations=self.rotations,
            last_velocity=self.velocity, last_action=self.last_action,
            target_map=self.target_map, state=self.expert_state, cfg=self.cfg,
            stuck_counter=self.stuck_counter, noise_gen=noise_gen, build_obs=build_obs,
        )
        # Update expert state externally so the env owns it.
        self.expert_state = out.state_next
        return out


# ---------------------------------------------------------------------------
# Sanity rollout
# ---------------------------------------------------------------------------
def sanity_rollout(cfg: dict, n_envs: int, episode_length: int, device: str,
                   seed: int = 0, verbose: bool = True) -> dict[str, float]:
    cfg = dict(cfg)
    cfg["episode_length"] = episode_length
    env = MeanShiftEnv(cfg, n_envs=n_envs, device=device, seed=seed)
    noise_gen = torch.Generator(device=env.device); noise_gen.manual_seed(seed + 1)

    final_match = []
    final_cov = []
    swap = []
    collisions = []
    success = []
    for t in range(episode_length):
        out = env.expert_action(noise_gen=noise_gen, build_obs=False)
        _, info, done = env.step(out.action)
        if done.all():
            break
    final_match = info.matching_distance.cpu().tolist()
    final_cov = info.coverage.cpu().tolist()
    swap = env._cum_swap.cpu().tolist()
    collisions = env._cum_collision.cpu().tolist()
    success = info.success.cpu().tolist()

    md = sum(final_match) / len(final_match)
    cv = sum(final_cov) / len(final_cov)
    sr = sum(success) / len(success)
    sw = sum(swap) / len(swap)
    cl = sum(collisions) / len(collisions)
    if verbose:
        print(f"[sanity] envs={n_envs}  T={episode_length}  "
              f"success_rate={sr:.2f}  matching_distance={md:.4f} ({md/env.spacing:.2f}d)  "
              f"coverage={cv:.2f}  swap/ep={sw:.1f}  collisions/ep={cl:.1f}")
    return dict(success_rate=sr, matching_distance=md, coverage=cv,
                mean_swap=sw, mean_collisions=cl)



def _scatter_dense_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor,
                         B: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a PyG batched (edge_index, edge_attr) for B graphs of N nodes into a
    dense (mask [B, N, N], full attr [B, N, N, edge_dim]) representation.

    `edge_dim` is read from edge_attr.shape[-1] so this works for both legacy
    (9-dim) and self-shape (13-dim) edges.
    """
    src, dst = edge_index
    batch = src // N
    src_local = src % N
    dst_local = dst % N
    edge_dim = int(edge_attr.shape[-1]) if edge_attr.numel() > 0 else EDGE_FEATURE_DIM
    mask = torch.zeros(B, N, N, dtype=torch.bool, device=edge_index.device)
    full = torch.zeros(B, N, N, edge_dim, device=edge_index.device,
                       dtype=edge_attr.dtype)
    mask[batch, src_local, dst_local] = True
    full[batch, src_local, dst_local] = edge_attr
    return mask, full

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs" / "n10_triangle.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sanity")
    s1.add_argument("--n_envs", type=int, default=8)
    s1.add_argument("--episode_length", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.cmd == "sanity":
        sanity_rollout(cfg, n_envs=args.n_envs, episode_length=args.episode_length,
                       device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
