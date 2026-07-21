"""v18d self-shape MAPPO with formation stress reward and BC anchor.

The active path is strict-local and continuous-action. A frozen BC actor can be
loaded as an action-space MSE anchor, while PPO optimizes the local formation
stress potential reported by `formation_stress_reward`.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from observations import (
    ACT_STOP, EDGE_FEATURE_DIM, EDGE_FEATURE_DIM_SELFSHAPE,
    LATENT_Z_DIM, NODE_FEATURE_DIM, NODE_FEATURE_DIM_SELFSHAPE, N_ACTIONS,
    build_edges, build_node_features, selfshape_node_dim,
)
from model import GATGRUActor, MeanAggActor, CentralCritic
from environment import ARCH_VERSION, MeanShiftEnv, _scatter_dense_edges, load_config
from shape_metrics import make_centered_t4_template
from templates import make_template, make_target_map
from target import compute_coverage, compute_matching


def reconstruct_pyg(edge_mask: torch.Tensor, edge_attr_full: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """edge_mask[mb, N, N] bool, edge_attr_full[mb, N, N, F_e] -> PyG (edge_index, edge_attr)."""
    mb, N, _ = edge_mask.shape
    nz = edge_mask.nonzero(as_tuple=False)
    batch, src_local, dst_local = nz[:, 0], nz[:, 1], nz[:, 2]
    src = batch * N + src_local
    dst = batch * N + dst_local
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = edge_attr_full[batch, src_local, dst_local]
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Borrowed: GAE + PPO helpers (yihuai-thesis-v2/rl/{gae,ppo}.py).
# ---------------------------------------------------------------------------
def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(last_value)
    for step in reversed(range(t)):
        nextnonterminal = (~dones[step]).float()
        nextvalues = last_value if step == t - 1 else values[step + 1]
        delta = rewards[step] + gamma * nextvalues * nextnonterminal - values[step]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[step] = lastgaelam
    return adv, adv + values


def ppo_critic_loss(critic, features, log_extent, old_values, returns, value_clip=0.2):
    values = critic(features, log_extent)
    clipped = old_values + (values - old_values).clamp(-value_clip, value_clip)
    return 0.5 * torch.max((values - returns).square(), (clipped - returns).square()).mean(), values


def clip_grad_step(loss, module, opt: torch.optim.Optimizer, max_grad_norm=1.0):
    """Compute grad, clip param norm, step. `module` may be a single nn.Module
    OR a list/tuple of modules — clipping spans all of them so the norm
    reflects the joint update."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if isinstance(module, (list, tuple)):
        params = []
        for m in module:
            if m is not None:
                params.extend(p for p in m.parameters() if p.requires_grad)
    else:
        params = list(module.parameters())
    norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
    opt.step()
    return norm


def cosine_lr(start: float, end: float, step: int, total: int) -> float:
    if total <= 1:
        return end
    t = min(max(step, 0), total) / float(total)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# Self-shape curriculum (R3: continuous anneal driven by single param s)
# ---------------------------------------------------------------------------
def compute_curriculum(update_idx: int, T_warmup: int, T_total: int,
                       cfg: dict, frozen_s: float | None = None) -> dict:
    """Continuous curriculum schedule (R3).

    s = max(0, (update - T_warmup) / (T_total - T_warmup)). During warmup s=0:
    cue always on, shape reward off, KL anchor at full strength. Post-warmup s
    rises monotonically to 1 (unless self-paced freeze pins it).

    cue_prob = max(0, 1 - s^cue_anneal_exp): concave anneal (slow start, fast
        end) — cue stays mostly on for first half of post-warmup.
    shape_w  = lerp(0,    shape_w_end,    s)  — shape reward fades in
    target_w = lerp(target_w_start, 0,    s)  — legacy target reward fades out
    kl_old_w = lerp(kl_old_start, kl_old_end, s)  — KL anchor weakens

    frozen_s: if provided (self-paced safeguard), override the computed s.
    """
    cur = cfg.get("curriculum", {})
    s_max = float(cur.get("s_max", 1.0))                  # FIX 4: cap s at empirical best
    if T_total <= T_warmup:
        s_raw = s_max
    else:
        s_raw = max(0.0, (float(update_idx) - float(T_warmup)) / float(T_total - T_warmup))
        s_raw = min(s_max, s_raw)
    s = float(frozen_s) if frozen_s is not None else s_raw
    s = max(0.0, min(s_max, s))

    cue_anneal_exp = float(cur.get("cue_anneal_exp", 0.7))
    cue_prob = max(0.0, 1.0 - s ** cue_anneal_exp)

    shape_w_end    = float(cur.get("shape_w_end", 8.0))
    target_w_start = float(cur.get("target_w_start", 5.0))
    kl_start       = float(cur.get("kl_old_start", 1.0))
    kl_end         = float(cur.get("kl_old_end", 0.05))

    return {
        "s": s, "s_raw": s_raw,
        "cue_prob": cue_prob,
        "shape_w":  lerp(0.0, shape_w_end, s),
        "target_w": lerp(target_w_start, 0.0, s),
        "kl_old_w": lerp(kl_start, kl_end, s),
        "in_warmup": update_idx < T_warmup,
    }


def self_paced_threshold(s: float, cfg: dict) -> float:
    cur = cfg.get("curriculum", {})
    base = float(cur.get("self_paced_base", 0.3))
    slope = float(cur.get("self_paced_slope", 0.2))
    return base + slope * (1.0 - s)


# ---------------------------------------------------------------------------
# Self-shape obs / edge builders (wrap expert.build_node_features + build_edges)
# ---------------------------------------------------------------------------
def build_selfshape_obs(env: MeanShiftEnv, exp_out, cue_present: torch.Tensor,
                        latent_z: torch.Tensor, heading_locked: bool = True
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict-local self-shape obs (target-blind, cue=0).

    `cue_present` and `heading_locked` are accepted for call-site compatibility
    but no longer change the output (target/heading channels were pruned).

    Returns (obs_nodes [B, N, node_dim_selfshape], edge_index, edge_attr).
    """
    # Build the slot one-hot encoding the actor's own role.
    sector_oh = None
    expected_node_dim = int(env.cfg.get("node_dim_selfshape", NODE_FEATURE_DIM_SELFSHAPE))
    slot_dim = max(0, expected_node_dim - NODE_FEATURE_DIM - LATENT_Z_DIM)
    slot_repr = str(env.cfg.get("slot_repr", "one_hot"))
    if slot_dim > 0 and hasattr(env, "slot_assignment") and env.slot_assignment is not None:
        if slot_repr == "one_hot":
            # v16+: N-dim slot one-hot. slot_dim must equal env.N.
            sector_oh = F.one_hot(env.slot_assignment.long(),
                                    num_classes=slot_dim).to(env.positions.dtype)
        elif slot_repr == "template_coords":
            # v19: 2-dim normalized template coord. slot_dim must equal 2.
            # gather the template position for each agent's assigned slot.
            idx = env.slot_assignment.long().unsqueeze(-1).expand(-1, -1, 2)  # [B, N, 2]
            sector_oh = env.slot_template_norm.gather(1, idx).to(env.positions.dtype)
        else:
            raise ValueError(f"unknown slot_repr={slot_repr!r}; "
                             "expected 'one_hot' or 'template_coords'")
    nodes = build_node_features(
        rotations=env.rotations,
        last_velocity=env.velocity, last_action=env.last_action,
        stuck_counter=env.stuck_counter, blocked_flag=exp_out.blocked_flag,
        hop=exp_out.hop_count,
        min_neighbor_dist=_min_neighbor_dist(env.positions),
        collision_risk=exp_out.collision_risk, cfg=env.cfg,
        latent_z=latent_z, sector_one_hot=sector_oh,
    )

    # Edges always carry raw z_src in self-shape (R7+R15) plus, when env exposes
    # slot info AND cfg's edge_dim_selfshape is 14, the desired pair distance
    # D[slot_src, slot_dst]/spacing (v17b'). Legacy v9-v16c ckpts have edge_dim=13
    # and must NOT receive the extra dim — respect ckpt schema.
    adj = _comm_adj(env.positions, float(env.cfg["comm_radius"]))
    expected_edge_dim = int(env.cfg.get("edge_dim_selfshape", 13))
    if expected_edge_dim >= 14:
        slot_assign = getattr(env, "slot_assignment", None)
        slot_pair_dist = getattr(env, "slot_pair_dist", None)
    else:
        slot_assign = None
        slot_pair_dist = None
    ei, ea = build_edges(env.positions, env.rotations, env.velocity, adj,
                          env.cfg, latent_z=latent_z,
                          slot_assignment=slot_assign,
                          slot_pair_dist=slot_pair_dist)
    return nodes, ei, ea


def _comm_adj(positions: torch.Tensor, comm_radius: float) -> torch.Tensor:
    """[B, N, N] bool adjacency in comm range (no self-loops)."""
    N = positions.shape[1]
    d = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
    diag = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    return (d < comm_radius) & ~diag


def _min_neighbor_dist(positions: torch.Tensor) -> torch.Tensor:
    """[B, N] nearest-neighbor distance (excluding self)."""
    N = positions.shape[1]
    d = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
    big = torch.full((1, N, N), float("inf"),
                     device=positions.device, dtype=positions.dtype)
    eye = torch.eye(N, dtype=torch.bool, device=positions.device).unsqueeze(0)
    d_off = torch.where(eye, big, d)
    return d_off.min(dim=-1).values


# ---------------------------------------------------------------------------
# Critic features (per-agent global)
# ---------------------------------------------------------------------------
CRITIC_FEATURE_DIM = 12
CRITIC_FEATURE_DIM_SELFSHAPE = 20    # 12 target-aware (R11 keep) + 4 z_i + 4 self-shape globals


def build_critic_features(env: MeanShiftEnv, t_global: int, episode_length: int
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-agent global feature stack for the centralized critic. [B, N, F_c]."""
    pos = env.positions; rot = env.rotations; vel = env.velocity
    B, N, _ = pos.shape
    centroid = pos.mean(dim=1, keepdim=True)
    cent_dist = (pos - centroid).norm(dim=-1)
    extent = cent_dist.mean(dim=1).clamp_min(1e-9)
    progress = torch.full((B, N), float(t_global) / max(1, episode_length),
                          device=pos.device, dtype=pos.dtype)
    # Per-agent target diagnostics
    d_im = (pos.unsqueeze(2) - env.target_map.unsqueeze(1)).norm(dim=-1)
    nearest_dist = d_im.min(dim=-1).values
    visible_count = (d_im < float(env.cfg["density_radius"])).sum(dim=-1).float()
    spacing = float(env.cfg["target_spacing"])
    target_centroid = env.target_map.mean(dim=1)
    relative_extent = (pos - target_centroid.unsqueeze(1)).norm(dim=-1).max(dim=1).values
    absolute_extent = torch.cat([pos, env.target_map], dim=1).norm(dim=-1).max(dim=1).values
    halfsize = torch.maximum(relative_extent, absolute_extent).clamp_min(6.0 * spacing)
    halfsize = halfsize.view(B, 1).clamp_min(1e-6)
    feats = torch.stack(
        [
            pos[..., 0] / halfsize,
            pos[..., 1] / halfsize,
            torch.cos(rot), torch.sin(rot),
            vel[..., 0] / float(env.cfg["v_max"]),
            vel[..., 1] / float(env.cfg["v_max"]),
            cent_dist / halfsize,
            nearest_dist / spacing,
            visible_count / float(env.M),
            (env.expert_state.claim_count.max(dim=-1).values
             / float(env.cfg["claim_steps"])).clamp_max(1.0),
            env.expert_state.blocked_steps.float()
                / float(env.cfg["blocked_steps_threshold"]),
            progress,
        ],
        dim=-1,
    )
    log_extent = torch.log(extent)
    return feats, log_extent


def build_critic_features_selfshape(env: MeanShiftEnv, t_global: int,
                                     episode_length: int,
                                     latent_z: torch.Tensor,
                                     profile_error: torch.Tensor,
                                     isolated_count: torch.Tensor,
                                     bond_penalty: torch.Tensor
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, N, 20] critic features for self-shape mode (R11: ADD not REPLACE).

    Layout: 12 legacy target-aware features + 4 z_i + 4 self-shape globals
    (broadcast to per-agent: profile_error, isolated_count/N, bond_penalty,
    success_distance_proxy).

    The 12 target-aware dims are kept verbatim because warmup-phase reward is
    target-driven (target_w=5.0 at s=0); critic must see target to estimate
    value correctly.
    """
    base_feats, log_extent = build_critic_features(env, t_global, episode_length)  # [B, N, 12]
    B, N, _ = base_feats.shape

    # 4-dim per-agent latent (broadcast to per-agent shape; z_i is already per-agent)
    z_feats = latent_z                                                  # [B, N, 4]

    # 4-dim global self-shape signals (broadcast across agents within each env)
    prof_pa  = profile_error.view(B, 1).expand(B, N)
    iso_pa   = (isolated_count / float(env.N)).view(B, 1).expand(B, N)
    bond_pa  = bond_penalty.view(B, 1).expand(B, N)
    spacing  = float(env.cfg["target_spacing"])
    threshold = float(env.cfg.get("success_shape_distance", 0.4)) * spacing
    # Success-distance proxy: how close are we to forming T4 (ratio shape_match
    # to threshold). When match_distance is unavailable we fall back to a
    # cheap surrogate: bond_penalty*spacing.
    succ_proxy = (bond_penalty * spacing / max(threshold, 1e-9)).clamp_max(2.0)
    succ_pa  = succ_proxy.view(B, 1).expand(B, N)
    selfshape_globals = torch.stack([prof_pa, iso_pa, bond_pa, succ_pa], dim=-1)  # [B, N, 4]

    feats = torch.cat([base_feats, z_feats, selfshape_globals], dim=-1)  # [B, N, 20]
    return feats, log_extent


# ---------------------------------------------------------------------------
# Partial state_dict load (R7+R17 warm-start: 43→47 node, 9→13 edge, zero-init
# new dims so old behavior is preserved at update 0)
# ---------------------------------------------------------------------------
def load_partial_state_dict(model: nn.Module, old_sd: dict,
                            verbose: bool = True) -> tuple[int, int]:
    """Load `old_sd` into `model.state_dict()`, copying matched-shape tensors
    verbatim. For Linear weights that mismatch only in input dim (`in_features`
    grew), copy the old columns and zero-init the new tail.

    Specifically handles:
    - `node_encoder.0.weight`: [hidden, 43] -> [hidden, 47]
    - `gat1.lin_edge.weight` (or PyG edge MLP): [hidden, 9] -> [hidden, 13]

    Returns (n_copied_full, n_copied_partial).
    """
    new_sd = model.state_dict()
    n_full = 0
    n_partial = 0
    for key, new_w in new_sd.items():
        if key not in old_sd:
            if verbose:
                print(f"  [warm-start] no match in old ckpt: {key} (kept random init)")
            continue
        old_w = old_sd[key]
        if old_w.shape == new_w.shape:
            new_w.copy_(old_w)
            n_full += 1
        elif (old_w.dim() == new_w.dim() == 2
              and old_w.shape[0] == new_w.shape[0]
              and old_w.shape[1] < new_w.shape[1]):
            # Linear weight, in-features grew. Copy old cols, zero-init the new ones.
            new_w.zero_()
            new_w[:, :old_w.shape[1]].copy_(old_w)
            n_partial += 1
            if verbose:
                print(f"  [warm-start] partial: {key} {tuple(old_w.shape)} → "
                      f"{tuple(new_w.shape)} (cols 0..{old_w.shape[1]-1} copied, "
                      f"{old_w.shape[1]}..{new_w.shape[1]-1} zero-init)")
        elif (old_w.dim() == new_w.dim() == 1
              and old_w.shape[0] < new_w.shape[0]):
            # bias grew (rare); copy old, zero-init new
            new_w.zero_()
            new_w[:old_w.shape[0]].copy_(old_w)
            n_partial += 1
        else:
            if verbose:
                print(f"  [warm-start] SKIP shape mismatch: {key} "
                      f"old={tuple(old_w.shape)} new={tuple(new_w.shape)}")
    model.load_state_dict(new_sd, strict=False)
    return n_full, n_partial


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
def reward_step_selfshape(env: MeanShiftEnv, actions: torch.Tensor,
                          prev_match: torch.Tensor, prev_cov: torch.Tensor,
                          prev_profile: torch.Tensor,
                          info, profile_error: torch.Tensor,
                          isolated_count: torch.Tensor,
                          bond_penalty: torch.Tensor,
                          shape_success_now: torch.Tensor,
                          was_shape_success_prev: torch.Tensor,
                          schedule: dict, cfg: dict,
                          per_agent_bond: torch.Tensor | None = None,
                          per_agent_coll: torch.Tensor | None = None,
                          per_agent_match_d: torch.Tensor | None = None,
                          prev_per_agent_match: torch.Tensor | None = None,
                          stress_env_score: torch.Tensor | None = None,
                          stress_per_agent: torch.Tensor | None = None,
                          stress_env_w: float = 0.0,
                          stress_pa_w: float = 0.0,
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Self-shape reward (R2 + R3 + R6b mixed with legacy target reward).

    v8 per-agent extensions (when corresponding per-agent inputs provided):
    - bond_penalty: env-scalar [B] still used as fallback; per_agent_bond [B, N]
      replaces it when given (asymmetric reward per agent → breaks R4 broadcast).
    - collisions: per-agent count of close neighbors used when per_agent_coll given.
    - per-agent match: r_match_self = match_w * (prev_match - cur_match) per agent
      (Procrustes-Hungarian distance to assigned template slot).

    Returns (rewards [B, N], match_now [B], cov_now [B], profile_now [B]).
    """
    rw = cfg.get("reward_weights_selfshape", cfg.get("reward_weights", {}))
    spacing = float(cfg["target_spacing"])

    # v16: handle both discrete [B, N] long and continuous [B, N, 2] float actions.
    # All per-agent rewards must be [B, N] (NOT [B, N, ...]) regardless of action shape.
    n_agents = actions.shape[1]
    is_continuous = actions.dtype.is_floating_point and actions.dim() == 3
    # Make a [B, N] "expand-target" indexable shape for `expand(-1, n_agents)` substitutes
    def _expand_pa(t):
        """t: [B] or [B, 1] → [B, N] expanded."""
        return t.unsqueeze(-1).expand(-1, n_agents).float() if t.dim() == 1 else t.expand(-1, n_agents).float()

    # --- shape delta (collective signal — keeps team coordination) ---
    delta_profile = prev_profile - profile_error            # higher = better
    r_shape = float(schedule["shape_w"]) * _expand_pa(delta_profile)

    # --- legacy target matching reward (mixed in via target_w(s) anneal) ---
    delta_match = (prev_match - info.matching_distance) / spacing
    r_target = float(schedule["target_w"]) * _expand_pa(delta_match)
    r_cov = 0.0  # coverage not used in self-shape

    # --- shape bond penalty: v8 per-agent if provided, else env-scalar (legacy) ---
    bond_w = float(rw.get("bond", 0.3))
    if per_agent_bond is not None:
        r_bond = -bond_w * per_agent_bond.float()                      # [B, N]
    else:
        r_bond = -bond_w * _expand_pa(bond_penalty)

    # --- collisions: v8 per-agent if provided, else env-scalar ---
    coll_w = float(rw.get("collision", 0.2))
    if per_agent_coll is not None:
        r_coll = -coll_w * per_agent_coll.float()                      # [B, N]
    else:
        r_coll = -coll_w * _expand_pa(info.collision_count)

    # --- connectivity penalty (R6b: per-agent local, isolated count) ---
    # NOTE: isolated_count is env-scalar; per-agent isolation flag would be cleaner.
    r_conn = -float(rw.get("connectivity", 0.5)) * _expand_pa(isolated_count)

    # --- v8: per-agent Procrustes-Hungarian match reward ---
    match_self_w = float(rw.get("match_self", 0.0))
    if per_agent_match_d is not None and prev_per_agent_match is not None and match_self_w > 0:
        delta_match_self = (prev_per_agent_match - per_agent_match_d) / spacing
        r_match_self = match_self_w * delta_match_self.float()
    else:
        r_match_self = torch.zeros_like(r_bond)

    # --- v9 FIX #7: absolute per-agent match penalty (not delta) ---
    # Delta saturates near plateau; absolute penalty keeps pressure aligned with eval metric.
    match_self_abs_w = float(rw.get("match_self_abs", 0.0))
    if per_agent_match_d is not None and match_self_abs_w > 0:
        r_match_abs = -match_self_abs_w * (per_agent_match_d / spacing).float()  # in d-units
    else:
        r_match_abs = torch.zeros_like(r_bond)

    # --- v18: Strictly-local formation stress reward ---
    # env_score and per_agent are ALREADY -V (controller's negated potential),
    # so we just multiply by positive weight; no sign flip.
    if stress_env_score is not None and stress_env_w > 0:
        r_stress_env = float(stress_env_w) * _expand_pa(stress_env_score)          # [B, N]
    else:
        r_stress_env = torch.zeros_like(r_bond)
    if stress_per_agent is not None and stress_pa_w > 0:
        r_stress_pa = float(stress_pa_w) * stress_per_agent.float()                # [B, N]
    else:
        r_stress_pa = torch.zeros_like(r_bond)

    # --- premature STOP / action cost ---
    # v16b: handle continuous action — true STOP only when both linear and angular
    # are near-zero. v16 used `linear<0.3` which mislabeled rotate-in-place,
    # slow micro-adjust, and reverse-correction as STOP — pushing policy toward
    # persistent-forward (spr=2.0, action magnitudes uncontrolled).
    if is_continuous:
        lin_eps = float(cfg.get("continuous_stop_lin_eps", 0.05))
        ang_eps = float(cfg.get("continuous_stop_ang_eps", 0.05))
        is_stop = (actions[..., 0].abs() < lin_eps) & (actions[..., 1].abs() < ang_eps)  # [B, N] bool
    else:
        is_stop = (actions == ACT_STOP)
    not_yet_success = ~_expand_pa(shape_success_now).bool()
    premature = is_stop & not_yet_success
    r_stop = -float(rw.get("premature_stop", 0.3)) * premature.float()
    # v16b: continuous action_cost = squared-magnitude penalty (not boolean ~is_stop).
    # Boolean penalty equally penalizes |a|=0.1 and |a|=1.0 → no incentive to be gentle.
    # Squared form: r_act = -w_lin * lin² - w_ang * ang² → favors smooth small motions
    # over saturating actions, while still penalizing outright STOP via r_stop.
    if is_continuous:
        w_lin = float(rw.get("linear_action_cost", 0.02))
        w_ang = float(rw.get("angular_action_cost", 0.01))
        r_act = -(w_lin * actions[..., 0].square()
                  + w_ang * actions[..., 1].square())                  # [B, N]
    else:
        r_act = -float(rw.get("action_cost", 0.005)) * (~is_stop).float()

    # --- completion bonus on first-time shape success ---
    became_success = shape_success_now & ~was_shape_success_prev
    r_complete = float(rw.get("complete", 5.0)) * _expand_pa(became_success)

    rewards = (r_shape + r_target + r_bond + r_coll + r_conn
               + r_match_self + r_match_abs
               + r_stress_env + r_stress_pa
               + r_stop + r_act + r_complete)
    return (rewards.float(), info.matching_distance, info.coverage, profile_error)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, T: int, B: int, N: int, gru_hidden: int,
                 device: torch.device, enable_bc: bool,
                 node_dim: int = NODE_FEATURE_DIM,
                 edge_dim: int = EDGE_FEATURE_DIM,
                 critic_feature_dim: int = CRITIC_FEATURE_DIM,
                 enable_selfshape: bool = False,
                 latent_z_dim: int = LATENT_Z_DIM):
        self.T, self.B, self.N = T, B, N
        self.device = device
        self.enable_bc = enable_bc
        self.enable_selfshape = enable_selfshape
        actions_buf = torch.empty(T, B, N, 2, dtype=torch.float32, device=device)
        self.data = {
            "obs_nodes": torch.empty(T, B, N, node_dim, device=device),
            "edge_attr_full": torch.empty(T, B, N, N, edge_dim, device=device),
            "edge_mask": torch.empty(T, B, N, N, dtype=torch.bool, device=device),
            "actions": actions_buf,
            "logp": torch.empty(T, B, N, device=device),
            "rewards": torch.empty(T, B, N, device=device),
            "dones": torch.empty(T, B, N, dtype=torch.bool, device=device),
            "values": torch.empty(T, B, N, device=device),
            "h0": torch.empty(T, B, N, gru_hidden, device=device),
            "features": torch.empty(T, B, N, critic_feature_dim, device=device),
            "log_extent": torch.empty(T, B, device=device),
        }
        if enable_bc:
            self.data["bc_logits"] = torch.empty(T, B, N, N_ACTIONS, device=device)
        if enable_selfshape:
            # R13: per-env cue_present needed at PPO update.
            self.data["cue_present"] = torch.empty(T, B, dtype=torch.bool, device=device)
            # latent z is needed at PPO update for MI bonus loss
            self.data["latent_z"] = torch.empty(T, B, N, latent_z_dim, device=device)

    def store(self, t: int, **items):
        for k, v in items.items():
            if k not in self.data:
                raise KeyError(f"unknown buffer key {k!r}")
            self.data[k][t].copy_(v)

    def chunk_view(self, chunk_len: int):
        T = self.T
        n_chunks_t = T // chunk_len
        out = {}
        for k, v in self.data.items():
            v = v[:n_chunks_t * chunk_len]
            new_shape = (n_chunks_t, chunk_len) + v.shape[1:]
            v = v.reshape(new_shape).transpose(1, 2)
            v = v.reshape((n_chunks_t * v.shape[1], chunk_len) + v.shape[3:])
            out[k] = v
        return out


# ---------------------------------------------------------------------------
# Chunked actor unroll for PPO update
# ---------------------------------------------------------------------------
# v18d: continuous action helpers (Normal sampler over linear/angular ∈ [-1, 1]²).
def actor_dist(actor: GATGRUActor, logits: torch.Tensor, vel_mu: torch.Tensor):
    """Returns Normal(tanh(vel_mu), std). logits arg kept for signature compat."""
    from torch.distributions import Normal
    mu = torch.tanh(vel_mu)                                            # [..., 2] in (-1, 1)
    std = actor.action_log_std.exp()                                   # [2]
    std = std.expand_as(mu).clamp(0.05, 1.0)                           # avoid degenerate
    return Normal(mu, std)


def actor_logp(dist, action: torch.Tensor) -> torch.Tensor:
    """Sum log_prob over action dims (Normal)."""
    return dist.log_prob(action).sum(-1)


def actor_entropy(dist) -> torch.Tensor:
    """Entropy summed over action dims, mean over agents/batch."""
    return dist.entropy().sum(-1).mean()


def chunked_actor_unroll(actor: GATGRUActor, mb: dict, chunk_len: int,
                         return_h: bool = False):
    """Replay a chunk through the actor with PPO-update gradients.

    When return_h=True, also return the in-graph h trajectory
    [B, chunk_len, N, hidden].
    """
    h = mb["h0"][:, 0].detach().contiguous()
    logits_list = []
    vel_list = []                                                     # v16: also collect velocity_mu
    h_list = []
    for tau in range(chunk_len):
        edge_index, edge_attr = reconstruct_pyg(
            mb["edge_mask"][:, tau], mb["edge_attr_full"][:, tau]
        )
        logits, vel_mu, h_next = actor(mb["obs_nodes"][:, tau], edge_index, edge_attr, h)
        logits_list.append(logits)
        vel_list.append(vel_mu)
        if return_h:
            h_list.append(h_next)
        done_t = mb["dones"][:, tau].any(dim=-1)
        h = torch.where(done_t.view(-1, 1, 1), torch.zeros_like(h_next), h_next)
    logits_t = torch.stack(logits_list, dim=1)
    vel_t = torch.stack(vel_list, dim=1)
    if return_h:
        return logits_t, vel_t, torch.stack(h_list, dim=1)
    return logits_t, vel_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=Path(__file__).resolve().parent.parent / "configs" / "n10_triangle.yaml")
    p.add_argument("--out_dir", type=Path, default=Path("runs/mappo"))
    p.add_argument("--total_updates", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--actor_type", choices=["gat", "meanagg", "tolstaya_edge", "tolstaya_pure"],
                   default=None, help="Actor architecture. Defaults to cfg.actor_type or gat.")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    # Self-shape mode (R5+R7+R14)
    p.add_argument("--actor_init", type=Path, default=None,
                   help="Old 43-dim actor ckpt to warm-start the new 47-dim trainable "
                        "actor (R7+R17 partial state_dict load). Self-shape mode only.")
    p.add_argument("--bc_anchor", type=Path, default=None,
                   help="v17d: same-schema (continuous) BC actor used as MSE anchor in PPO loss. "
                        "Adds bc_anchor_w * mean((tanh(actor.vel_mu) - tanh(bc.vel_mu))**2) per-step.")
    p.add_argument("--bc_anchor_w", type=float, default=None,
                   help="v17d: weight on MSE-to-BC anchor. Default from cfg.bc_anchor_w (= 2.0).")
    p.add_argument("--T_warmup", type=int, default=None,
                   help="Override curriculum.T_warmup (self-shape only).")
    return p.parse_args()


def _build_actor(cfg, device, selfshape: bool = False):
    if selfshape:
        node_dim = int(cfg.get("node_dim_selfshape",
                               selfshape_node_dim(int(cfg.get("n_agents", 10)))))
        edge_dim = int(cfg.get("edge_dim_selfshape", EDGE_FEATURE_DIM_SELFSHAPE))
    else:
        node_dim = NODE_FEATURE_DIM
        edge_dim = EDGE_FEATURE_DIM
    common = dict(
        node_dim=node_dim, edge_dim=edge_dim,
        gat_hidden=int(cfg["gat_hidden"]), gat_heads=int(cfg.get("gat_heads", 4)),
        gru_hidden=int(cfg["gru_hidden"]),
        action_log_std_init=float(cfg.get("action_log_std_init", -1.9)),
    )
    actor_type = str(cfg.get("actor_type", "gat"))
    if actor_type == "gat":
        actor = GATGRUActor(**common)
    elif actor_type == "meanagg":
        actor = MeanAggActor(**common)
    elif actor_type in {"tolstaya_edge", "tolstaya_pure"}:
        from model_tolstaya import TolstayaActor
        actor = TolstayaActor(**common, use_edge_attr=(actor_type == "tolstaya_edge"),
                              k_taps=int(cfg.get("tolstaya_k_taps", 3)))
    else:
        raise ValueError(f"Unknown actor_type={actor_type!r}")
    return actor.to(device)


def _set_freeze_node_encoder_gat1(actor: torch.nn.Module, frozen: bool) -> None:
    """R5: freeze/unfreeze early encoder/message layer during warmup."""
    for p in actor.node_encoder.parameters():
        p.requires_grad_(not frozen)
    for name in ("gat1", "msg1", "self1"):
        module = getattr(actor, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(not frozen)


def _save_ckpt(path: Path, *, cfg_dump: dict, actor, critic, actor_opt, critic_opt, update: int):
    state = {"arch_version": ARCH_VERSION, "config": cfg_dump,
             "actor_type": str(cfg_dump.get("actor_type", "gat")),
             "n_agents": int(cfg_dump.get("n_agents", 10)),
             "actor_state": actor.state_dict(), "critic_state": critic.state_dict(),
             "actor_opt_state": actor_opt.state_dict(), "critic_opt_state": critic_opt.state_dict(),
             "update": update}
    torch.save(state, path)


def main():
    """v18d entrypoint: self-shape MAPPO with stress reward + BC anchor."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.actor_type is not None:
        cfg["actor_type"] = args.actor_type
    return main_selfshape(args, cfg)



# ============================================================================
# Self-shape MAPPO loop (R1-R17)
# ============================================================================
def _quick_eval_selfshape(actor: torch.nn.Module, cfg: dict, templates: dict,
                          n_envs: int = 64, episode_length: int = 300,
                          device: str = "cuda", seed: int = 0,
                          sampled: bool = True,
                          theta_bins: int = 36) -> dict[str, float]:
    """Self-shape eval over multiple shapes: for each shape in `templates`, force
    the env to that shape for all n_envs envs, run a full episode, compute
    metrics, return per-shape dict + mean aggregate (under keys with no prefix,
    for backward compat with single-shape ckpt scorer).
    """
    eval_cfg = dict(cfg)
    eval_cfg["episode_length"] = episode_length
    spacing = float(eval_cfg["target_spacing"])
    threshold_loose = float(eval_cfg.get("success_shape_distance", 0.4)) * spacing
    threshold_strict = float(eval_cfg.get("success_shape_distance_strict", 0.25)) * spacing

    shape_names = list(templates.keys())
    per_shape: dict[str, dict[str, float]] = {}
    actor.eval()

    for k, name in enumerate(shape_names):
        eval_env = MeanShiftEnv(eval_cfg, n_envs=n_envs, device=device, seed=seed + k)
        noise_gen = torch.Generator(device=eval_env.device)
        noise_gen.manual_seed(seed + k + 1)
        eval_env.reset_all()
        eval_env.set_cue_present_prob(0.0)

        # Force this shape for all envs in this eval pass.
        tmpl = templates[name].to(device=eval_env.device)
        from shape_metrics import pairwise_dist as _pdist
        pair_dist = _pdist(tmpl).to(device=eval_env.device)        # [N, N]
        if eval_env.slot_pair_dist.ndim == 3:
            eval_env.slot_pair_dist[:] = pair_dist.unsqueeze(0).expand(n_envs, -1, -1)
        else:
            eval_env.slot_pair_dist = pair_dist
        if name in eval_env._shape_names:
            shape_k = eval_env._shape_names.index(name)
        else:
            shape_k = 0  # will be ignored — only used for compute_shape_metrics_multi grouping
        eval_env.shape_id[:] = shape_k
        # Override target_map to this shape.
        eval_env.target_map[:] = make_target_map(
            name, eval_env.positions, spacing=spacing,
            theta=float(eval_cfg.get("target_theta", 0.0)),
            center_mode=str(eval_cfg.get("target_center_mode", "initial_swarm_centroid")),
        )

        h = torch.zeros(n_envs, eval_env.N, actor.hidden, device=eval_env.device)
        with torch.no_grad():
            for _t in range(episode_length):
                exp_out = eval_env.expert_action(noise_gen=noise_gen, build_obs=False)
                obs, ei, ea = build_selfshape_obs(eval_env, exp_out,
                                                   cue_present=eval_env.cue_present,
                                                   latent_z=eval_env.latent_z,
                                                   heading_locked=True)
                logits, vel_mu, h = actor(obs, ei, ea, h)
                if sampled:
                    dist = actor_dist(actor, logits, vel_mu)
                    action = dist.sample().clamp(-1.0, 1.0)
                else:
                    action = torch.tanh(vel_mu).clamp(-1.0, 1.0)
                eval_env.step(action)

        # Compute metrics with this shape's template.
        prof, match_d, iso, bp = eval_env.compute_shape_metrics(
            tmpl, compute_match_distance=True, theta_bins=theta_bins,
        )
        succ_loose = (match_d < threshold_loose).float().mean().item()
        succ_strict = (match_d < threshold_strict).float().mean().item()
        with torch.no_grad():
            stress_env, _ = eval_env.compute_stress_reward()
        per_shape[name] = {
            "shape_succ":         float(succ_loose),
            "shape_succ_strict":  float(succ_strict),
            "shape_match_d":      float(match_d.mean().item()) / spacing,
            "shape_match_d_max":  float(match_d.max().item()) / spacing,
            "shape_profile":      float(prof.mean().item()),
            "isolated":           float(iso.mean().item()),
            "bond_penalty":       float(bp.mean().item()),
            "coll":               float(eval_env._cum_collision.float().mean().item()),
            "stress_score":       float(stress_env.mean().item()),
        }
    actor.train()

    # Aggregate (mean across shapes) for ckpt scorer (back-compat keys).
    metric_keys = list(next(iter(per_shape.values())).keys())
    out: dict[str, float] = {
        k: float(sum(per_shape[s][k] for s in shape_names) / len(shape_names))
        for k in metric_keys
    }
    out["per_shape"] = per_shape          # type: ignore[assignment]
    out["shape_names"] = shape_names      # type: ignore[assignment]
    return out


def main_selfshape(args, cfg):
    """MAPPO loop for self-organization formations (R1-R17)."""
    if args.smoke:
        cfg["envs"] = 2
        cfg["rollout_steps"] = 32
        cfg["total_updates"] = 4
        cfg["minibatches"] = 2
        cfg["ppo_epochs"] = 1
        cfg["chunk_len"] = 8
        cfg["episode_length"] = 64
        # Tighten curriculum so smoke covers warmup → mid → late phases.
        cfg.setdefault("curriculum", {}).update({"T_warmup": 2, "T_total": 4})
    if args.total_updates is not None:
        cfg["total_updates"] = args.total_updates
    if args.device is not None:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("actor_type", "gat")
    cfg.setdefault("n_agents", 10)
    cfg.setdefault("node_dim_selfshape", selfshape_node_dim(int(cfg["n_agents"])))
    cfg.setdefault("edge_dim_selfshape", EDGE_FEATURE_DIM_SELFSHAPE)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.get("seed", 1)))

    n_envs = int(cfg["envs"])
    rollout = int(cfg["rollout_steps"])
    chunk_len = int(cfg.get("chunk_len", 20))
    if rollout % chunk_len != 0:
        raise ValueError(f"rollout_steps ({rollout}) must be divisible by chunk_len ({chunk_len})")

    cur = cfg.setdefault("curriculum", {})
    T_warmup = int(args.T_warmup if args.T_warmup is not None else cur.get("T_warmup", 200))
    T_total  = int(cur.get("T_total", cfg.get("total_updates", 1500)))
    cur["T_warmup"] = T_warmup
    cur["T_total"]  = T_total

    # --- Trainable actor (47-dim node, 13-dim edge) ---
    actor = _build_actor(cfg, device, selfshape=True)
    if args.actor_init is not None:
        ckpt = torch.load(args.actor_init, map_location=device, weights_only=False)
        if ckpt.get("arch_version") != ARCH_VERSION:
            raise SystemExit(f"--actor_init arch_version {ckpt.get('arch_version')!r} != {ARCH_VERSION!r}")
        ckpt_actor_type = ckpt.get("actor_type", ckpt.get("config", {}).get("actor_type", "gat"))
        if str(ckpt_actor_type) != str(cfg.get("actor_type", "gat")):
            raise SystemExit(
                f"--actor_init actor_type {ckpt_actor_type!r} != cfg actor_type "
                f"{cfg.get('actor_type', 'gat')!r}"
            )
        n_full, n_partial = load_partial_state_dict(actor, ckpt["actor_state"], verbose=True)
        print(f"[selfshape] warm-start from {args.actor_init}: "
              f"{n_full} layers full-copy, {n_partial} layers partial-copy (cols zero-init)")

    # FIX 2b: bias STOP logit at warm-start so actor doesn't carry collapsed
    # a0=0.00 prior from the previous run. The warm-start ckpt's policy_head
    # bias is loaded above; we just nudge bias[0] (STOP) up to give STOP a
    # fighting chance once shape_success_now starts firing (FIX 2 widens the
    # threshold so it actually does fire near T4 — without bias the policy
    # has no prior to discover STOP under PPO exploration).
    stop_bias = float(cfg.get("stop_action_bias_warmstart", 0.0))
    if stop_bias != 0.0 and args.actor_init is not None:
        with torch.no_grad():
            actor.policy_head[-1].bias[0] += stop_bias
        print(f"[selfshape] STOP-bias nudged: policy_head.bias[0] += {stop_bias:.2f}")

    # --- v17d: Frozen bc_anchor_actor (same schema as trainable actor) ---
    bc_anchor_actor = None
    bc_anchor_w = float(args.bc_anchor_w if args.bc_anchor_w is not None
                          else cfg.get("bc_anchor_w", 0.0))
    if args.bc_anchor is not None:
        bc_anchor_actor = _build_actor(cfg, device, selfshape=True)
        bc_ckpt = torch.load(args.bc_anchor, map_location=device, weights_only=False)
        ckpt_actor_type = bc_ckpt.get("actor_type", bc_ckpt.get("config", {}).get("actor_type", "gat"))
        if str(ckpt_actor_type) != str(cfg.get("actor_type", "gat")):
            raise SystemExit(
                f"--bc_anchor actor_type {ckpt_actor_type!r} != cfg actor_type "
                f"{cfg.get('actor_type', 'gat')!r}"
            )
        bc_anchor_actor.load_state_dict(bc_ckpt["actor_state"], strict=False)
        bc_anchor_actor.eval()
        for _p in bc_anchor_actor.parameters():
            _p.requires_grad_(False)
        if bc_anchor_w == 0.0:
            bc_anchor_w = 2.0                                                    # default if cfg missing
        print(f"[selfshape] BC anchor loaded from {args.bc_anchor} (frozen, same schema, w={bc_anchor_w})")

    # --- Critic ---
    critic = CentralCritic(feature_dim=CRITIC_FEATURE_DIM_SELFSHAPE,
                            hidden=int(cfg["critic_hidden"])).to(device)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=float(cfg["lr_actor"]))
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=float(cfg["lr_critic"]))

    # --- Env + buffer ---
    env = MeanShiftEnv(cfg, n_envs=n_envs, device=device, seed=int(cfg.get("seed", 1)))
    expert_noise_gen = torch.Generator(device=device); expert_noise_gen.manual_seed(int(cfg.get("seed", 1)) + 7)
    # Multi-shape: pre-build templates dict for compute_shape_metrics_multi /
    # _quick_eval_selfshape. Single-shape configs (shape_pool=["t4"]) still work.
    shape_names = list(cfg.get("shape_pool", ["t4"]))
    templates = {name: make_template(name, env.spacing, device=device)
                 for name in shape_names}
    # Canonical single-shape template for compute_per_agent_metrics call sites,
    # whose outputs are zeroed by w=0 (bond/match_self/match_self_abs) in
    # multi-shape configs, so this T4-fallback never enters the reward signal.
    template = templates[shape_names[0]]

    buf = RolloutBuffer(
        T=rollout, B=n_envs, N=env.N, gru_hidden=actor.hidden,
        device=device, enable_bc=False,
        node_dim=int(cfg["node_dim_selfshape"]), edge_dim=int(cfg["edge_dim_selfshape"]),
        critic_feature_dim=CRITIC_FEATURE_DIM_SELFSHAPE,
        enable_selfshape=True, latent_z_dim=int(cfg.get("latent_z_dim", LATENT_Z_DIM)),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = dict(cfg); cfg_dump["arch_version"] = ARCH_VERSION
    # v9 FIX #16: explicit dim metadata so render/eval can autodetect schema
    cfg_dump["actor_type"] = str(cfg.get("actor_type", "gat"))
    cfg_dump["n_agents"] = int(cfg.get("n_agents", env.N))
    cfg_dump["selfshape_node_dim"] = int(cfg["node_dim_selfshape"])
    cfg_dump["selfshape_edge_dim"] = int(cfg["edge_dim_selfshape"])
    cfg_dump["selfshape_critic_feature_dim"] = CRITIC_FEATURE_DIM_SELFSHAPE
    cfg_dump["selfshape_latent_z_dim"] = int(cfg.get("latent_z_dim", LATENT_Z_DIM))
    (args.out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    h = torch.zeros(n_envs, env.N, actor.hidden, device=device)

    # Initial cue dropout decision (curriculum at update 0)
    sched0 = compute_curriculum(0, T_warmup, T_total, cfg)
    env.set_cue_present_prob(sched0["cue_prob"])

    # Per-env reward state.
    with torch.no_grad():
        prev_prof, _, prev_iso, prev_bond = env.compute_shape_metrics_multi(templates)
        _, prev_match = compute_matching(env.positions, env.target_map)
        prev_cov = compute_coverage(env.positions, env.target_map,
                                     radius=float(cfg["arrival_radius"]))
        # v8: per-agent state for asymmetric reward
        _, _, prev_per_agent_match = env.compute_per_agent_metrics(template)
    spacing = float(cfg["target_spacing"])
    threshold = float(cfg.get("success_shape_distance", 0.4)) * spacing
    was_shape_success_prev = torch.zeros(n_envs, dtype=torch.bool, device=device)
    rw_self = cfg.get("reward_weights_selfshape", {})
    v8_per_agent_enable = float(rw_self.get("match_self", 0.0)) > 0 or \
                          float(rw_self.get("match_self_abs", 0.0)) > 0

    # v18: strictly-local formation stress reward (controller's V potential)
    v18_stress_enable = (float(rw_self.get("stress_env", 0.0)) > 0
                          or float(rw_self.get("stress_per_agent", 0.0)) > 0)
    stress_env_w = float(rw_self.get("stress_env", 0.0))
    stress_pa_w = float(rw_self.get("stress_per_agent", 0.0))

    # Self-paced safeguard state. (FIX 1 v2: relative-to-peak, not absolute threshold)
    online_succ_history: list[float] = []          # diagnostic only — old metric
    eval_shape_succ_history: list[float] = []      # what the safeguard tracks
    peak_post_warmup: float = 0.0                  # best post-warmup eval shape_succ seen
    rolling_window = int(cur.get("self_paced_window", 3))
    frozen_s: float | None = None
    # Initialize to -inf so the first eval always sets the best checkpoint bar.
    best_eval_succ = -float("inf")
    eval_every = 50

    total = int(cfg["total_updates"])
    print(f"[selfshape-mappo] device={device} envs={n_envs} rollout={rollout} "
          f"chunk={chunk_len} updates={total} T_warmup={T_warmup} T_total={T_total}")

    t_global = 0
    for update in range(1, total + 1):
        t0 = time.time()

        # --- Curriculum + freeze (R3 + R5) ---
        sched = compute_curriculum(update, T_warmup, T_total, cfg, frozen_s=frozen_s)
        in_warmup = sched["in_warmup"]
        _set_freeze_node_encoder_gat1(actor, frozen=in_warmup)
        cue_prob = sched["cue_prob"]
        kl_old_w = sched["kl_old_w"]

        # rollout-level aggregates
        ep_match: list[float] = []
        ep_cov: list[float] = []
        ep_success: list[bool] = []
        ep_swap: list[int] = []
        ep_coll: list[int] = []
        ep_profile: list[float] = []
        last_info = None
        for step in range(rollout):
            # ===== PRE-step: actor obs, value estimate =====
            # State here is s_t (BEFORE action). prev_profile / prev_per_agent_match
            # were set at end of last iter to s_t metrics.
            with torch.no_grad():
                exp_out = env.expert_action(noise_gen=expert_noise_gen, build_obs=False)
                obs, ei, ea = build_selfshape_obs(
                    env, exp_out, cue_present=env.cue_present,
                    latent_z=env.latent_z, heading_locked=True,
                )
                edge_mask, edge_attr_full = _scatter_dense_edges(ei, ea, B=env.B, N=env.N)

                logits, vel_mu, h_next = actor(obs, ei, ea, h)
                dist = actor_dist(actor, logits, vel_mu)
                action = dist.sample().clamp(-1.0, 1.0)                    # [B, N, 2] float
                logp = actor_logp(dist, action)                            # [B, N]

                # Critic features at s_t (PRE-step) — value V(s_t) for advantage
                feats, log_ext = build_critic_features_selfshape(
                    env, t_global, int(cfg["episode_length"]),
                    latent_z=env.latent_z, profile_error=prev_prof,
                    isolated_count=prev_iso, bond_penalty=prev_bond,
                )
                values = critic(feats, log_ext)

            store_kw = dict(
                obs_nodes=obs, edge_attr_full=edge_attr_full, edge_mask=edge_mask,
                actions=action, logp=logp, values=values, h0=h,
                features=feats, log_extent=log_ext,
                cue_present=env.cue_present, latent_z=env.latent_z,
            )
            buf.store(step, **store_kw)

            # ===== env step: s_t → s_{t+1} =====
            _vel_world, info, done = env.step(action)
            last_info = info

            # ===== POST-step: compute s_{t+1} metrics for reward =====
            # v9 FIX #1: reward = f(prev_state) - f(post_state) = action's actual effect.
            # Old code used pre-step metrics → reward was lagged by one step.
            with torch.no_grad():
                profile_now, _md, iso_now, bond_now = env.compute_shape_metrics_multi(templates)
                shape_success_now = profile_now < float(cfg.get(
                    "success_shape_profile_threshold", 0.15))   # FIX #6: relaxed from 0.05
                if v8_per_agent_enable:
                    pa_bond, pa_coll, pa_match = env.compute_per_agent_metrics(template)
                else:
                    pa_bond = pa_coll = pa_match = None
                # v18: strictly-local formation stress reward
                if v18_stress_enable:
                    stress_env_now, stress_pa_now = env.compute_stress_reward()
                else:
                    stress_env_now = stress_pa_now = None

            rewards, prev_match, prev_cov, prev_prof = reward_step_selfshape(
                env, action, prev_match, prev_cov, prev_prof, info,
                profile_error=profile_now, isolated_count=iso_now,
                bond_penalty=bond_now, shape_success_now=shape_success_now,
                was_shape_success_prev=was_shape_success_prev,
                schedule=sched, cfg=cfg,
                per_agent_bond=pa_bond, per_agent_coll=pa_coll,
                per_agent_match_d=pa_match,
                prev_per_agent_match=prev_per_agent_match if v8_per_agent_enable else None,
                stress_env_score=stress_env_now, stress_per_agent=stress_pa_now,
                stress_env_w=stress_env_w, stress_pa_w=stress_pa_w,
            )
            # Track post-step state as next iter's "prev"
            prev_iso = iso_now
            prev_bond = bond_now
            if v8_per_agent_enable:
                prev_per_agent_match = pa_match
            was_shape_success_prev = shape_success_now
            done_per_agent = done.unsqueeze(-1).expand(-1, env.N)
            buf.store(step, rewards=rewards, dones=done_per_agent)

            # reset finished envs (sample episode-end metrics first)
            if done.any():
                ep_match.extend(info.matching_distance[done].cpu().tolist())
                ep_cov.extend(info.coverage[done].cpu().tolist())
                ep_success.extend(info.success[done].cpu().tolist())
                ep_swap.extend(env._cum_swap[done].cpu().tolist())
                ep_coll.extend(env._cum_collision[done].cpu().tolist())
                ep_profile.extend(profile_now[done].cpu().tolist())
                env.reset_envs(done)
                # Re-roll cue dropout for freshly reset envs
                env.set_cue_present_prob(cue_prob, mask=done)
                reset = done.to(device)
                h = h_next.clone(); h[reset] = 0.0
                # Recompute prev signals for reset envs
                with torch.no_grad():
                    pp, _, pi, pb = env.compute_shape_metrics_multi(templates)
                    _, pm = compute_matching(env.positions, env.target_map)
                    pc = compute_coverage(env.positions, env.target_map,
                                           radius=float(cfg["arrival_radius"]))
                    if v8_per_agent_enable:
                        _, _, ppm = env.compute_per_agent_metrics(template)
                        prev_per_agent_match = torch.where(reset.unsqueeze(-1),
                                                            ppm, prev_per_agent_match)
                prev_iso = torch.where(reset, pi, prev_iso)
                prev_bond = torch.where(reset, pb, prev_bond)
                prev_prof = torch.where(reset, pp, prev_prof)
                prev_match = torch.where(reset, pm, prev_match)
                prev_cov = torch.where(reset, pc, prev_cov)
                was_shape_success_prev = torch.where(reset,
                                                      torch.zeros_like(was_shape_success_prev),
                                                      was_shape_success_prev)
            else:
                h = h_next
            t_global += 1

        # bootstrap value
        with torch.no_grad():
            prof_last, _, iso_last, bond_last = env.compute_shape_metrics_multi(templates)
            feats_last, le_last = build_critic_features_selfshape(
                env, t_global, int(cfg["episode_length"]),
                latent_z=env.latent_z, profile_error=prof_last,
                isolated_count=iso_last, bond_penalty=bond_last,
            )
            last_value = critic(feats_last, le_last)

        adv, returns = compute_gae(
            buf.data["rewards"], buf.data["dones"], buf.data["values"],
            last_value, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]),
        )

        # chunked PPO
        chunks = buf.chunk_view(chunk_len)
        T = rollout
        n_chunks_t = T // chunk_len

        def _to_chunks(x):
            x = x[: n_chunks_t * chunk_len]
            x = x.reshape((n_chunks_t, chunk_len) + x.shape[1:])
            x = x.transpose(1, 2)
            return x.reshape((n_chunks_t * x.shape[1], chunk_len) + x.shape[3:])
        chunks_adv = _to_chunks(adv); chunks_ret = _to_chunks(returns)
        adv_flat = chunks_adv.reshape(-1)
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        chunks_adv = adv_norm.reshape(chunks_adv.shape)

        n_chunks = chunks["obs_nodes"].shape[0]
        mb_n = max(1, n_chunks // int(cfg["minibatches"]))
        clip = float(cfg["clip"])
        ent_coef = cosine_lr(float(cfg["ent_coef_start"]),
                             float(cfg.get("ent_coef_end", cfg["ent_coef_start"])),
                             update, total)

        agg = {"policy_loss": 0.0, "policy_entropy": 0.0, "value_loss": 0.0,
               "approx_kl": 0.0, "clip_frac": 0.0,
               "kl_div_z": 0.0, "z_div_loss": 0.0,
               "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
               "bc_anchor_mse": 0.0, "n": 0}

        for _ in range(int(cfg["ppo_epochs"])):
            perm = torch.randperm(n_chunks, device=device)
            for start in range(0, n_chunks, mb_n):
                idx = perm[start: start + mb_n]
                mb = {k: chunks[k][idx] for k in chunks}
                mb_adv = chunks_adv[idx]; mb_ret = chunks_ret[idx]

                logits, vel_mu = chunked_actor_unroll(actor, mb, chunk_len)
                dist = actor_dist(actor, logits, vel_mu)
                logp = actor_logp(dist, mb["actions"])
                ratio = (logp - mb["logp"]).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                policy_loss = torch.max(pg1, pg2).mean()
                entropy = actor_entropy(dist)

                # v18d: z-diversification regularizer disabled (was discrete-only).
                z_div_loss = torch.zeros((), device=device)
                kl_div_z_value = 0.0

                # v17d: MSE-to-BC anchor (continuous mode only). Frozen BC actor sees
                # the same chunk; we penalize divergence between actor's mu and BC's mu.
                # Without this, even BC-warm-start PPO drifts away because density reward
                # landscape is misaligned with BC's optimum (v17c regression).
                bc_anchor_loss = torch.zeros((), device=device)
                if bc_anchor_actor is not None:
                    with torch.no_grad():
                        _logits_bc, vel_mu_bc = chunked_actor_unroll(bc_anchor_actor, mb, chunk_len)
                        mu_bc = torch.tanh(vel_mu_bc)                                  # [n_mb, T, N, 2]
                    mu_actor = torch.tanh(vel_mu)                                       # in-graph
                    bc_anchor_loss = ((mu_actor - mu_bc) ** 2).mean()
                bc_anchor_value = float(bc_anchor_loss.item()) if isinstance(bc_anchor_loss, torch.Tensor) else 0.0

                actor_loss = (policy_loss
                              - ent_coef * entropy
                              + z_div_loss
                              + bc_anchor_w * bc_anchor_loss)
                actor_gn = clip_grad_step(actor_loss, actor, actor_opt,
                                           float(cfg["max_grad_norm"]))

                # v16b: post-step soft clamp on continuous action_log_std.
                # Range [-4.0, -1.5] → std ∈ [0.018, 0.22]. Prevents (a) entropy
                # term blowing std up to maximize Normal entropy (unbounded above),
                # (b) std collapse to ~0 which would freeze gradient.
                with torch.no_grad():
                    ls_lo = float(cfg.get("action_log_std_min", -4.0))
                    ls_hi = float(cfg.get("action_log_std_max", -1.5))
                    actor.action_log_std.data.clamp_(ls_lo, ls_hi)

                mb_feats = mb["features"].reshape(-1, env.N, CRITIC_FEATURE_DIM_SELFSHAPE)
                mb_le = mb["log_extent"].reshape(-1)
                mb_vals = mb["values"].reshape(-1, env.N)
                mb_ret_flat = mb_ret.reshape(-1, env.N)
                closs, _ = ppo_critic_loss(critic, mb_feats, mb_le, mb_vals, mb_ret_flat,
                                            value_clip=float(cfg["value_clip"]))
                critic_gn = clip_grad_step(closs, critic, critic_opt,
                                            float(cfg["max_grad_norm"]))

                with torch.no_grad():
                    approx_kl = (mb["logp"] - logp).mean().abs().item()
                    clip_frac = ((ratio - 1.0).abs() > clip).float().mean().item()
                agg["policy_loss"] += policy_loss.item()
                agg["policy_entropy"] += entropy.item()
                agg["value_loss"] += closs.item()
                agg["actor_grad_norm"] += float(actor_gn)
                agg["critic_grad_norm"] += float(critic_gn)
                agg["approx_kl"] += approx_kl
                agg["clip_frac"] += clip_frac
                agg["kl_div_z"] += kl_div_z_value
                agg["z_div_loss"] += float(z_div_loss.item()) if isinstance(z_div_loss, torch.Tensor) else 0.0
                agg["bc_anchor_mse"] += bc_anchor_value
                agg["n"] += 1

        n_mb = max(1, agg.pop("n"))
        agg = {k: v / n_mb for k, v in agg.items()}

        # cosine LR schedule.
        actor_lr_now = cosine_lr(float(cfg["lr_actor"]),
                                  float(cfg["lr_actor"]) * 0.1, update, total)
        actor_opt.param_groups[0]["lr"] = actor_lr_now
        critic_lr_now = cosine_lr(float(cfg["lr_critic"]),
                                   float(cfg["lr_critic"]) * 0.1, update, total)
        for g in critic_opt.param_groups:
            g["lr"] = critic_lr_now

        with torch.no_grad():
            if buf.data["actions"].dtype.is_floating_point:
                # v16 continuous: report (linear, angular) mean and abs.mean
                actions_flat = buf.data["actions"].reshape(-1, 2)
                lin_mean = actions_flat[:, 0].mean().item()
                ang_mean = actions_flat[:, 1].mean().item()
                lin_abs = actions_flat[:, 0].abs().mean().item()
                ang_abs = actions_flat[:, 1].abs().mean().item()
                act_dist = [lin_mean, ang_mean, lin_abs, ang_abs]
            else:
                act_counts = torch.bincount(buf.data["actions"].reshape(-1).long(), minlength=4).float()
                act_dist = (act_counts / act_counts.sum()).cpu().tolist()
        elapsed = time.time() - t0

        # Episode aggregates
        if ep_match:
            md = sum(ep_match) / len(ep_match) / spacing
            sr = sum(1.0 if s else 0.0 for s in ep_success) / len(ep_success)
            cl = sum(ep_coll) / len(ep_coll)
            pf = sum(ep_profile) / len(ep_profile)
            n_ep = len(ep_match)
            src = "ep"
        elif last_info is not None:
            md = float(last_info.matching_distance.mean().item()) / spacing
            sr = float(last_info.success.float().mean().item())
            cl = float(env._cum_collision.float().mean().item())
            pf = float(prev_prof.mean().item())
            n_ep = 0
            src = "tail"
        else:
            md = sr = cl = pf = 0.0; n_ep = 0; src = "na"

        in_warmup_str = "WARMUP" if in_warmup else "      "
        sk_str = rp_str = dn_str = sector_str = ""
        # v18: log stress (negated potential V; higher = closer to target shape)
        stress_str = ""
        if v18_stress_enable and stress_env_now is not None:
            stress_str = (f"str={float(stress_env_now.mean().item()):+.4f} ")
        role_str = ""
        print(
            f"[upd {update:04d}/{total} {in_warmup_str}] {elapsed:.2f}s "
            f"s={sched['s']:.2f} cue={cue_prob:.2f} sw={sched['shape_w']:.1f} "
            f"tw={sched['target_w']:.1f} klw={kl_old_w:.2f} "
            f"r_mean={buf.data['rewards'].mean().item():+.4f} "
            f"pl={agg['policy_loss']:+.3f} ent={agg['policy_entropy']:+.3f} "
            f"vl={agg['value_loss']:.3f} kl={agg['approx_kl']:.3f} "
            f"klz={agg['kl_div_z']:.4f} zdl={agg['z_div_loss']:.4f} "
            f"{'lin=' if buf.data['actions'].dtype.is_floating_point else 'a0='}{act_dist[0]:+.2f} "
            f"{'ang=' if buf.data['actions'].dtype.is_floating_point else 'a1='}{act_dist[1]:+.2f} "
            f"{'|l|=' if buf.data['actions'].dtype.is_floating_point else 'a2='}{act_dist[2]:.2f} "
            f"{'|a|=' if buf.data['actions'].dtype.is_floating_point else 'a3='}{act_dist[3]:.2f} "
            f"std_l={float(actor.action_log_std[0].exp().item()):.3f} std_a={float(actor.action_log_std[1].exp().item()):.3f} "
            f"{('bc='+format(agg['bc_anchor_mse'], '.4f')+' ') if bc_anchor_actor is not None else ''}"
            f"{sk_str}{rp_str}{dn_str}{sector_str}{stress_str}{role_str}"
            f"[{src} n={n_ep}] match={md:.2f}d profile={pf:.3f} succ={sr:.2f} coll={cl:.1f}",
            flush=True,
        )

        # FIX 1: Self-paced safeguard now feeds from EVAL shape_succ (computed
        # below). The old version appended online `sr` (info.success = TARGET match)
        # which mechanically drops as cue → 0 by curriculum design — false freeze.
        # We just track online_succ for diagnostics; safeguard logic below uses
        # eval_shape_succ_history instead, updated in the eval block.
        if src == "ep" and n_ep > 0:
            online_succ_history.append(sr)

        # Periodic eval (uses shape_match_distance; selects best_eval.pt)
        if (update % eval_every == 0) or update == total or args.smoke:
            ev = _quick_eval_selfshape(actor, cfg, templates,
                                        n_envs=64 if not args.smoke else 4,
                                        episode_length=300 if not args.smoke else 32,
                                        device=str(device), seed=update, sampled=True,
                                        theta_bins=36)
            print(f"  [eval@{update}] succ@0.4d={ev['shape_succ']:.3f} "
                  f"succ@0.25d={ev['shape_succ_strict']:.3f} "
                  f"match={ev['shape_match_d']:.2f}d max={ev['shape_match_d_max']:.2f}d "
                  f"profile={ev['shape_profile']:.3f} bond={ev['bond_penalty']:.3f} "
                  f"coll={ev['coll']:.0f}", flush=True)
            # Per-shape breakdown when multi-shape training.
            per_shape = ev.get("per_shape", {})
            if len(per_shape) > 1:
                for name, m in per_shape.items():
                    print(f"     {name:<8} succ@0.4d={m['shape_succ']:.3f} "
                          f"succ@0.25d={m['shape_succ_strict']:.3f} "
                          f"match={m['shape_match_d']:.2f}d", flush=True)
            # v11: strict-first composite score with floor.
            # - Heavily weights succ@0.25d (visual shape) over succ@0.4d (loose).
            # - Floor: if succ_loose < 0.30, score = -inf to avoid selecting
            #   collapsed ckpts.
            ckpt_score_mode = str(cfg.get("ckpt_score_mode", "v11"))
            if ckpt_score_mode != "v11":
                raise ValueError(f"Unsupported ckpt_score_mode={ckpt_score_mode!r}; v18d+ uses 'v11'.")
            if ev["shape_succ"] < float(cfg.get("ckpt_score_loose_floor", 0.30)):
                score = -1e9
            else:
                score = (ev["shape_succ_strict"] * 2.0
                         + ev["shape_succ"] * 0.3
                         - ev["shape_match_d_max"] * 0.3
                         - (ev["coll"] / 50.0) * 0.2
                         - ev["bond_penalty"] * 0.2)
            if score > best_eval_succ:
                best_eval_succ = score
                _save_ckpt(args.out_dir / "best_eval.pt", cfg_dump=cfg_dump,
                           actor=actor, critic=critic, actor_opt=actor_opt,
                           critic_opt=critic_opt, update=update)
                print(f"  ↑ best_eval (composite): succ_strict={ev['shape_succ_strict']:.3f}, "
                      f"succ_loose={ev['shape_succ']:.3f}, max_match={ev['shape_match_d_max']:.2f}d, "
                      f"bond={ev['bond_penalty']:.3f}, score={score:.3f}", flush=True)

            # FIX 1 v2: RELATIVE-TO-PEAK self-paced safeguard.
            #
            # Why not absolute threshold (v1 of this fix): warmup-trained
            # actor evaluated at cue=0 produces a "baseline OOD level" of
            # ~0.15-0.30 shape_succ; an absolute threshold confuses this
            # baseline with degradation and freezes immediately at warmup
            # end. Relative-to-peak only triggers on TRUE degradation:
            # rolling-N eval shape_succ drops to < freeze_drop_frac × peak.
            #
            # peak is tracked only post-warmup (warmup-era values are OOD).
            eval_shape_succ_history.append(ev["shape_succ"])
            if not in_warmup:
                peak_post_warmup = max(peak_post_warmup, ev["shape_succ"])
            if (len(eval_shape_succ_history) >= rolling_window
                    and not in_warmup
                    and peak_post_warmup > 0.10):
                roll = sum(eval_shape_succ_history[-rolling_window:]) / rolling_window
                drop_frac = float(cfg.get("curriculum", {}).get("self_paced_drop_frac", 0.5))
                recover_frac = float(cfg.get("curriculum", {}).get("self_paced_recover_frac", 0.85))
                if roll < drop_frac * peak_post_warmup:
                    if frozen_s is None or frozen_s > sched["s_raw"]:
                        frozen_s = sched["s_raw"]
                        print(f"  ↓ self-paced freeze: rolling-{rolling_window} eval shape_succ "
                              f"={roll:.3f} < {drop_frac:.2f}×peak({peak_post_warmup:.3f})"
                              f"={drop_frac*peak_post_warmup:.3f}, pin s={frozen_s:.3f}",
                              flush=True)
                elif frozen_s is not None and roll >= recover_frac * peak_post_warmup:
                    print(f"  ↑ self-paced unfreeze: rolling shape_succ={roll:.3f} "
                          f">= {recover_frac:.2f}×peak({peak_post_warmup:.3f}), resume curriculum",
                          flush=True)
                    frozen_s = None

        # Periodic ckpt
        if (update % args.save_every == 0) or update == total:
            _save_ckpt(args.out_dir / f"ckpt_{update:06d}.pt", cfg_dump=cfg_dump,
                       actor=actor, critic=critic, actor_opt=actor_opt,
                       critic_opt=critic_opt, update=update)

    print(f"[selfshape-mappo] done. best_eval_shape_succ={best_eval_succ:.3f}")




if __name__ == "__main__":
    main()
