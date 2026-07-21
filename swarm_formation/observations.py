"""Communicated mean-shift expert + observation builder for the actor.

Outputs per step
    velocity        [B, N, 2]            world frame, capped at v_max
    action          [B, N]   long        STOP=0 LEFT=1 RIGHT=2 FORWARD=3
    sample_weight   [B, N]   float       1.0 normally, lower in label-margin band
    state_next      ExpertState          claim_count + blocked_steps
    diagnostics                          density / hop / blocked / collision_risk
    obs_nodes       [B, N, node_dim]     actor node features (no goal / slot / GW)
    edge_index      [2, E]               PyG batched edge index
    edge_attr       [E, edge_dim]        edge features
    edge_batch_n    int                  N (for split-back convenience)

The expert is *deterministic given state and a fixed noise generator*; the noise
generator is passed in as a torch.Generator so demo collection is reproducible
and tests can disable it by passing None.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch

# Action ids — this is the canonical mapping used everywhere in this package.
ACT_STOP, ACT_LEFT, ACT_RIGHT, ACT_FORWARD = 0, 1, 2, 3
N_ACTIONS = 4

NODE_FEATURE_DIM = 8                       # v18e: 8 informative strict-local base dims (heading/target/sel-flat pruned)
EDGE_FEATURE_DIM = 9

# Self-shape mode (R7+R8+R15+R17): append 4 latent z to node, 4 raw z_src to edge.
LATENT_Z_DIM = 4
# v15: 3-dim sector one-hot (apex/bot-left/bot-right) — DEPRECATED, replaced by v16 slot.
# v16: slot one-hot (default N=10), frozen per-episode from rank(z[..., 0]).
# v18: target_vec_body (oracle nav cue computed from global centroid + best_θ) REMOVED
# to enforce strict locality. Actor obs is now: 19 base + 24 sel_flat (43)
#       + 4 latent_z (47) + N slot one-hot = 47+N self-shape obs.
SECTOR_DIM = 10                                                          # v16: 3 → 10 slots

# v19: slot can also be encoded as the 2-D template coordinate of the assigned
# slot (normalized by spacing). This decouples slot-block width from N, so the
# same actor architecture transfers across team sizes (10 / 21 / 28 / ...).
SLOT_EMBED_DIM_TEMPLATE = 2


def selfshape_slot_dim(n_agents: int, slot_repr: str = "one_hot") -> int:
    """Slot block width for strict-local self-shape observations.

    slot_repr:
        "one_hot"          — legacy: N-dim indicator (default, preserves v18d).
        "template_coords"  — 2-D normalized template position of the assigned slot.
    """
    if slot_repr == "one_hot":
        return int(n_agents)
    if slot_repr == "template_coords":
        return SLOT_EMBED_DIM_TEMPLATE
    raise ValueError(f"unknown slot_repr={slot_repr!r}; expected 'one_hot' or 'template_coords'")


def selfshape_node_dim(n_agents: int, slot_repr: str = "one_hot") -> int:
    """Node feature dim for self-shape mode: base + latent z + slot block."""
    return NODE_FEATURE_DIM + LATENT_Z_DIM + selfshape_slot_dim(n_agents, slot_repr)


NODE_FEATURE_DIM_SELFSHAPE = selfshape_node_dim(SECTOR_DIM)              # 57 (one_hot, N=10)
# v17b': edge schema augmented with desired_pair_dist[slot_src, slot_dst]/spacing
# (1 dim) — closes the info bottleneck that prevented BC from learning the
# controller's role-distance spring rule. Edge dim 13 → 14. d_desired is task
# spec (T4 template lookup), not state, so still strictly local.
EDGE_FEATURE_DIM_SELFSHAPE = EDGE_FEATURE_DIM + LATENT_Z_DIM + 1     # 14 (geom 9 + raw z_src 4 + d_desired 1)


# ---------------------------------------------------------------------------
# Config / invariants
# ---------------------------------------------------------------------------
def assert_time_scales(cfg: dict) -> None:
    """Cheap asserts on the user-facing time/length scales. Run at startup."""
    fs = float(cfg["forward_step"]); vmx = float(cfg["v_max"]); dt = float(cfg["dt"])
    assert math.isclose(fs, vmx * dt, rel_tol=1e-6), \
        f"forward_step ({fs}) != v_max*dt ({vmx*dt})"
    rs = float(cfg["rotate_step"]); thf = float(cfg["angle_threshold_far"])
    assert math.isclose(rs, thf, rel_tol=1e-6), \
        f"rotate_step ({rs}) != angle_threshold_far ({thf})"
    rc = float(cfg["r_claim"]); ar = float(cfg["arrival_radius"]); rr = float(cfg["r_release"])
    assert rc < ar < rr, f"need r_claim ({rc}) < arrival_radius ({ar}) < r_release ({rr})"
    dr = float(cfg["density_radius"]); ks = float(cfg["kernel_sigma"])
    assert dr >= 2.0 * ks, f"density_radius ({dr}) < 2*kernel_sigma ({2*ks})"
    cr = float(cfg["comm_radius"])
    assert cr >= 2.0 * dr, f"comm_radius ({cr}) < 2*density_radius ({2*dr})"
    tsr = float(cfg["target_sense_radius"])
    assert tsr >= dr, f"target_sense_radius ({tsr}) < density_radius ({dr}); Q_i should " \
                      f"be at least as wide as the density-contribution cutoff."
    if fs > 0.5 * ar:
        # Crossing detection would need to be enabled; this prototype runs slow steps so we
        # only warn instead of forcing it on. The user's config tunes fs < 0.5*ar.
        raise AssertionError(
            f"forward_step ({fs}) > 0.5 * arrival_radius ({0.5*ar}); enable crossing detection."
        )


# ---------------------------------------------------------------------------
# Body-frame helpers (the only convention borrowed from the v2 codebase).
# ---------------------------------------------------------------------------
def world_to_body(v_world: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into per-agent body frame.
    v_world: [..., 2]  heading: [...] same broadcast shape.
    Returns [..., 2] = [v_ahead, v_left].
    """
    c = heading.cos()
    s = heading.sin()
    vx, vy = v_world[..., 0], v_world[..., 1]
    return torch.stack([vx * c + vy * s, -vx * s + vy * c], dim=-1)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class ExpertState:
    claim_count: torch.Tensor       # [B, N, M] float
    blocked_steps: torch.Tensor     # [B, N]    int

    @classmethod
    def zeros(cls, B: int, N: int, M: int, device: torch.device | str = "cpu") -> "ExpertState":
        return cls(
            claim_count=torch.zeros(B, N, M, device=device),
            blocked_steps=torch.zeros(B, N, dtype=torch.long, device=device),
        )

    def reset_envs(self, mask: torch.Tensor) -> None:
        """Reset entries where mask[b] is True."""
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0]
            self.claim_count[idx] = 0.0
            self.blocked_steps[idx] = 0


# ---------------------------------------------------------------------------
# Pairwise / mask helpers (borrowed convention).
# ---------------------------------------------------------------------------
def _pairwise_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[B, A, 2], [B, B_, 2] -> [B, A, B_, 2]."""
    return a.unsqueeze(2) - b.unsqueeze(1)


def _pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _pairwise_diff(a, b).norm(dim=-1)


def _self_mask(N: int, device) -> torch.Tensor:
    return torch.eye(N, device=device, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Hop-count BFS on the communication graph.
# ---------------------------------------------------------------------------
def _hop_count(adj: torch.Tensor, sources: torch.Tensor, hop_max: int) -> torch.Tensor:
    """adj: [B, N, N] bool (no self-loops). sources: [B, N] bool.
    Returns hop count [B, N] int — `hop_max+1` for unreachable nodes.
    """
    B, N, _ = adj.shape
    INF = hop_max + 1
    hop = torch.full((B, N), INF, device=adj.device, dtype=torch.long)
    hop = torch.where(sources, torch.zeros_like(hop), hop)
    if not sources.any():
        return hop
    for _ in range(hop_max):
        # neighbor_min[b, i] = min over j (adj[b, i, j] ? hop[b, j] : INF)
        h = hop.unsqueeze(1).expand(B, N, N)            # [B, N, N] (j varies on -1)
        masked = torch.where(adj, h, torch.full_like(h, INF))
        nbr_min = masked.min(dim=-1).values             # [B, N]
        new_hop = torch.minimum(hop, nbr_min + 1)
        if torch.equal(new_hop, hop):
            break
        hop = new_hop
    return hop


# ---------------------------------------------------------------------------
# Action discretization
# ---------------------------------------------------------------------------
def _angle_threshold(dist_to_nearest_target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Lerp between near and far thresholds based on distance to nearest target."""
    near = float(cfg["angle_threshold_near"])
    far = float(cfg["angle_threshold_far"])
    blend_dist = float(cfg["angle_blend_dist"])
    t = (dist_to_nearest_target / blend_dist).clamp(0.0, 1.0)
    return near + t * (far - near)


def velocity_to_action(velocity: torch.Tensor, heading: torch.Tensor,
                       dist_nearest: torch.Tensor, cfg: dict
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Discretize world-frame velocity to {STOP, LEFT, RIGHT, FORWARD}.

    Returns (action, heading_error, threshold). heading_error and threshold are
    returned for label-margin sample weighting.
    """
    speed = velocity.norm(dim=-1)                              # [B, N]
    body = world_to_body(velocity, heading)                     # [B, N, 2]
    v_ahead, v_left = body[..., 0], body[..., 1]
    heading_error = torch.atan2(v_left, v_ahead)
    threshold = _angle_threshold(dist_nearest, cfg)             # [B, N]

    low_speed = speed < float(cfg["min_move_speed"])
    left_mask = (heading_error > threshold) & ~low_speed
    right_mask = (heading_error < -threshold) & ~low_speed

    action = torch.full_like(speed, float(ACT_FORWARD), dtype=torch.long)
    action[low_speed] = ACT_STOP
    action[left_mask] = ACT_LEFT
    action[right_mask] = ACT_RIGHT
    return action, heading_error, threshold


def label_sample_weight(heading_error: torch.Tensor, threshold: torch.Tensor,
                        action: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Down-weight samples close to a STOP/turn threshold to avoid teaching noise."""
    margin = float(cfg["label_margin"])
    amb_w = float(cfg["ambiguous_label_weight"])
    abs_he = heading_error.abs()
    near_threshold = (abs_he - threshold).abs() < margin
    return torch.where(near_threshold & (action != ACT_STOP),
                       torch.full_like(heading_error, amb_w),
                       torch.ones_like(heading_error))


def local_stop_proxy(positions: torch.Tensor, targets: torch.Tensor,
                     claim_count: torch.Tensor, velocity: torch.Tensor,
                     min_neighbor_dist: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Local STOP signal — derived only from per-robot sensing, no global success.
    Returns [B, N] bool.
    """
    d = _pairwise_dist(positions, targets)                      # [B, N, M]
    nearest_dist, nearest_k = d.min(dim=-1)
    near_any = nearest_dist < float(cfg["arrival_radius"])
    # claim count for the nearest target
    cc_nearest = claim_count.gather(2, nearest_k.unsqueeze(-1)).squeeze(-1)
    stable_claim = cc_nearest >= 0.95 * float(cfg["claim_steps"])
    low_speed = velocity.norm(dim=-1) < float(cfg["min_move_speed"])
    low_collision = min_neighbor_dist > float(cfg["collision_risk_radius"])
    return near_any & stable_claim & low_speed & low_collision


# ---------------------------------------------------------------------------
# Main expert step
# ---------------------------------------------------------------------------
@dataclass
class ExpertOutput:
    velocity: torch.Tensor                 # [B, N, 2]
    action: torch.Tensor                   # [B, N] long
    sample_weight: torch.Tensor            # [B, N] float
    state_next: ExpertState
    # diagnostics
    density: torch.Tensor                  # [B, N, M]
    claim_count: torch.Tensor              # [B, N, M]
    blocked_flag: torch.Tensor             # [B, N] bool
    hop_count: torch.Tensor                # [B, N] long
    hop_gradient: torch.Tensor             # [B, N, 2] body-frame
    collision_risk: torch.Tensor           # [B, N] int (count of neighbors within risk radius)
    q_visible: torch.Tensor | None = None  # [B, N, M] bool: real Q_i mask (d < target_sense_radius)
    matching_distance: torch.Tensor | None = None   # [B] (filled by env)
    # observation graph (built in same pass to avoid re-doing pairwise math)
    obs_nodes: torch.Tensor | None = None
    edge_index: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None
    edge_batch_n: int | None = None


def _update_claim(prev: torch.Tensor, dist_to_targets: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Hysteresis claim counter on (robot, target) pairs."""
    rc = float(cfg["r_claim"])
    rr = float(cfg["r_release"])
    decay = float(cfg["claim_decay"])
    claim_steps = float(cfg["claim_steps"])
    inside = dist_to_targets < rc
    outside = dist_to_targets > rr
    new = torch.where(inside, (prev + 1.0).clamp_max(claim_steps), prev)
    new = torch.where(outside, torch.zeros_like(prev), new)
    in_band = ~inside & ~outside
    new = torch.where(in_band, prev * decay, new)
    return new


def _density(positions: torch.Tensor, targets: torch.Tensor,
             claim_count: torch.Tensor, adj_with_self: torch.Tensor, cfg: dict
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Communicated target-centric density + Q_i visibility.

    m_{j,k}      = density_cutoff_{j,k} * gauss(d_{j,k}) * (1 + boost * sat(claim_{j,k}))
        density_cutoff_{j,k} = 1[||x_j - q_k|| < density_radius]    -- contribution scope
    rho_{i,k}    = sum_{j: i==j or j∈N_i} m_{j,k}                   communicated total
    m_self_{i,k} = m_{i,k}                                          robot i's own message
    q_visible_{i,k} = 1[||x_i - q_k|| < target_sense_radius]        -- robot i's Q_i set
    """
    d_jk = _pairwise_dist(positions, targets)                    # [B, N, M]
    sigma = float(cfg["kernel_sigma"])
    density_cutoff = d_jk < float(cfg["density_radius"])
    q_visible = d_jk < float(cfg["target_sense_radius"])
    gauss = torch.exp(-(d_jk ** 2) / (2.0 * sigma ** 2))
    boost = 1.0 + float(cfg["claim_boost"]) * (claim_count / float(cfg["claim_steps"])).clamp(0, 1)
    m = density_cutoff.float() * gauss * boost                   # [B, N, M] per-robot message
    rho = torch.einsum("bij,bjm->bim", adj_with_self.float(), m)  # [B, N, M]
    return rho, m, density_cutoff, q_visible


def expert_step(positions: torch.Tensor, rotations: torch.Tensor,
                last_velocity: torch.Tensor, last_action: torch.Tensor,
                target_map: torch.Tensor, state: ExpertState, cfg: dict,
                stuck_counter: torch.Tensor,
                noise_gen: torch.Generator | None = None,
                build_obs: bool = True) -> ExpertOutput:
    """One expert tick.

    Args:
        positions:     [B, N, 2]
        rotations:     [B, N]   heading in rad
        last_velocity: [B, N, 2] world frame from previous step (for blocked detection)
        last_action:   [B, N]    long
        target_map:    [B, M, 2]
        state:         ExpertState (mutated *not* in-place; new tensors returned)
        stuck_counter: [B, N] int — used as a node feature only

    Build_obs=False skips the observation graph (used by tests / sanity).
    """
    B, N, _ = positions.shape
    M = target_map.shape[1]
    device = positions.device

    # --- communication graph -------------------------------------------------
    d_ij = _pairwise_dist(positions, positions)                 # [B, N, N]
    not_self = ~_self_mask(N, device).unsqueeze(0)
    adj = (d_ij < float(cfg["comm_radius"])) & not_self          # [B, N, N]
    # adj_with_self includes the diagonal so a robot aggregates its own message.
    adj_with_self = adj | _self_mask(N, device).unsqueeze(0)

    # --- claim hysteresis ----------------------------------------------------
    d_im = _pairwise_dist(positions, target_map)                 # [B, N, M]
    claim_count = _update_claim(state.claim_count, d_im, cfg)

    # --- communicated density + Q_i ------------------------------------------
    rho, m_self, density_cutoff, q_visible = _density(
        positions, target_map, claim_count, adj_with_self, cfg,
    )
    rho_minus_self = (rho - m_self).clamp_min(0.0)

    # --- blocked detection + hop count (precede v_rho so hop can penalize density) -
    diff_ij = _pairwise_diff(positions, positions)               # [B, N, N, 2] = pos_i - pos_j
    last_speed = last_velocity.norm(dim=-1)                      # [B, N]
    is_slow = last_speed < float(cfg["blocked_speed_threshold"])
    blocked_steps = torch.where(is_slow, state.blocked_steps + 1,
                                torch.zeros_like(state.blocked_steps))
    blocked_flag = blocked_steps >= int(cfg["blocked_steps_threshold"])

    hop = _hop_count(adj, blocked_flag, int(cfg["hop_max"]))     # [B, N] long
    # hop_gradient_world points AWAY from blocked sources (escape direction).
    # For each (i, j∈N_i) with hop_j < hop_i, j is closer to a blocked source. The
    # vector from j to i is `pos_i - pos_j = diff_ij[i, j]`, which points away from
    # the blocked region; summing over those j yields the escape direction. Do NOT
    # negate this — it is locked by test_hop_gradient_sign.
    hop_diff = hop.unsqueeze(2) - hop.unsqueeze(1)               # [i, j]: hop_i - hop_j > 0 => j closer
    lower_neighbor = (hop_diff > 0) & adj
    grad_world = (diff_ij * lower_neighbor.unsqueeze(-1).float()).sum(dim=2)
    grad_norm = grad_world.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    hop_gradient_world = torch.where(
        grad_norm > 1e-6, grad_world / grad_norm, torch.zeros_like(grad_world)
    )
    hop_gradient_body = world_to_body(hop_gradient_world, rotations)

    # --- mean-shift density-gradient velocity --------------------------------
    # Weight = (share)^p * m_self, masked by Q_i and downweighted in directions
    # toward blocked sources (hop penalty). "Share" turns the degenerate mean-shift
    # (everyone pulled to the highest-density mode) into an assignment-free
    # formation: whichever robot has the largest m_self at slot k wins by share^p.
    # Hop penalty multiplies w by 1 / (1 + α * align), where align = max(0, dot(
    # (q_k - x_i)/||·||, -hop_gradient)) is high when slot k is in the direction of
    # blocked sources (the inward / occupied region).
    sharpness = float(cfg.get("density_sharpness", 3.0))
    share = m_self / rho.clamp_min(1e-9)                          # [B, N, M] in [0, 1]
    rel = target_map.unsqueeze(1) - positions.unsqueeze(2)        # [B, N, M, 2]
    rel_norm = rel.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    target_dir = rel / rel_norm                                   # [B, N, M, 2]
    blocked_dir = (-hop_gradient_world).unsqueeze(2)              # [B, N, 1, 2] toward blocked
    align = (target_dir * blocked_dir).sum(dim=-1).clamp_min(0.0)  # [B, N, M] in [0, 1]
    hop_pen_alpha = float(cfg.get("density_hop_penalty", 1.0))
    hop_penalty = 1.0 / (1.0 + hop_pen_alpha * align)
    w = (share ** sharpness) * m_self * q_visible.float() * hop_penalty
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    weighted_target = (w.unsqueeze(-1) * target_map.unsqueeze(1)).sum(dim=-2)  # [B, N, 2]
    centroid = weighted_target / w_sum
    v_rho = centroid - positions
    speed_rho = v_rho.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v_rho = v_rho * (float(cfg["v_ms"]) / speed_rho).clamp_max(1.0)

    has_visible = q_visible.any(dim=-1)                          # [B, N]

    # --- shape-centroid guidance when Q_i empty ------------------------------
    shape_centroid = target_map.mean(dim=1, keepdim=True)        # [B, 1, 2]
    v_a_dir = shape_centroid - positions
    v_a_norm = v_a_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v_a = v_a_dir * (float(cfg["v_guidance"]) / v_a_norm)
    v_a = torch.where(has_visible.unsqueeze(-1), torch.zeros_like(v_a), v_a)

    # --- collision repulsion -------------------------------------------------
    coll_radius = float(cfg["collision_radius"])
    coll_hard = float(cfg["collision_hard"])
    coll_gain = float(cfg["collision_gain"])
    d_safe = d_ij.clamp_min(coll_hard)
    in_range = (d_ij < coll_radius) & not_self                   # [B, N, N]
    repel_dir = diff_ij / d_safe.unsqueeze(-1).clamp_min(1e-9)   # i - j direction
    strength = (1.0 - d_ij / coll_radius).clamp_min(0.0) * coll_gain
    strength = strength * in_range.float()
    v_c = (repel_dir * strength.unsqueeze(-1)).sum(dim=2)        # [B, N, 2]
    v_c_speed = v_c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v_c = v_c * (float(cfg["v_collision"]) / v_c_speed).clamp_max(1.0)

    # --- escape velocity for blocked robots ----------------------------------
    # Pick the visible target with the lowest competitive weight (least claimed by
    # others) as escape target; fall back to hop_gradient direction if Q_i empty.
    competitor_w = (rho_minus_self * q_visible.float()).masked_fill(~q_visible, float("inf"))
    low_k = competitor_w.argmin(dim=-1)                          # [B, N] long
    target_low = target_map.gather(1, low_k.unsqueeze(-1).expand(-1, -1, 2))
    v_esc = target_low - positions
    v_esc_speed = v_esc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v_esc = v_esc * (float(cfg["v_escape"]) / v_esc_speed)
    v_esc = torch.where(has_visible.unsqueeze(-1), v_esc,
                        hop_gradient_world * float(cfg["v_escape"]))
    v_esc = v_esc * blocked_flag.unsqueeze(-1).float()

    # --- noise (symmetry break) ---------------------------------------------
    noise_std = float(cfg["expert_velocity_noise"])
    if noise_std > 0.0:
        if noise_gen is not None:
            noise = torch.randn(B, N, 2, device=device, generator=noise_gen) * noise_std
        else:
            noise = torch.randn(B, N, 2, device=device) * noise_std
    else:
        noise = torch.zeros(B, N, 2, device=device)

    # --- combine + cap to v_max ---------------------------------------------
    v = v_rho + v_a + v_c + v_esc + noise
    speed = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v = v * (float(cfg["v_max"]) / speed).clamp_max(1.0)

    # --- discretize ----------------------------------------------------------
    nearest_target_dist = d_im.min(dim=-1).values
    action, heading_err, threshold = velocity_to_action(v, rotations, nearest_target_dist, cfg)
    sample_weight = label_sample_weight(heading_err, threshold, action, cfg)

    # local STOP override (only for samples that the threshold rule didn't already STOP)
    # so that the dataset has plenty of stable-target STOP labels.
    min_neighbor_dist = d_ij.masked_fill(~not_self, float("inf")).min(dim=-1).values
    stop_mask = local_stop_proxy(positions, target_map, claim_count, v, min_neighbor_dist, cfg)
    action = torch.where(stop_mask, torch.full_like(action, ACT_STOP), action)

    # --- collision risk count ------------------------------------------------
    risk_radius = float(cfg["collision_risk_radius"])
    collision_risk = ((d_ij < risk_radius) & not_self).sum(dim=-1)

    state_next = ExpertState(claim_count=claim_count, blocked_steps=blocked_steps)

    out = ExpertOutput(
        velocity=v,
        action=action,
        sample_weight=sample_weight,
        state_next=state_next,
        density=rho,
        claim_count=claim_count,
        blocked_flag=blocked_flag,
        hop_count=hop,
        hop_gradient=hop_gradient_body,
        collision_risk=collision_risk,
        q_visible=q_visible,                                          # FIX #2: expose real Q_i
    )

    # --- observation -- nodes + PyG edges -----------------------------------
    if build_obs:
        out.obs_nodes = build_node_features(
            rotations=rotations, last_velocity=last_velocity,
            last_action=last_action, stuck_counter=stuck_counter,
            blocked_flag=blocked_flag, hop=hop,
            min_neighbor_dist=min_neighbor_dist, collision_risk=collision_risk, cfg=cfg,
        )
        ei, ea = build_edges(positions=positions, rotations=rotations,
                              last_velocity=last_velocity, adj=adj, cfg=cfg)
        out.edge_index = ei
        out.edge_attr = ea
        out.edge_batch_n = N
    return out


# ---------------------------------------------------------------------------
# Observation builders (actor-input only — no goal/slot/GW)
# ---------------------------------------------------------------------------
def build_node_features(rotations: torch.Tensor,
                        last_velocity: torch.Tensor, last_action: torch.Tensor,
                        stuck_counter: torch.Tensor,
                        blocked_flag: torch.Tensor, hop: torch.Tensor,
                        min_neighbor_dist: torch.Tensor, collision_risk: torch.Tensor,
                        cfg: dict | None = None,
                        latent_z: torch.Tensor | None = None,
                        sector_one_hot: torch.Tensor | None = None) -> torch.Tensor:
    """[B, N, NODE_FEATURE_DIM (+ latent_z + slot one-hot)] actor inputs.

    Strict-local, target-blind per-robot features (v18e). Only the signals that
    are informative in the final cue=0 setup are kept (8 base dims): forward
    speed, distance to the nearest neighbour, blocked flag, hop count to the
    nearest congestion, collision risk, stuck counter, and the last-action
    one-hot restricted to its two live categories (STOP / FORWARD). The latent z
    (symmetry-breaking role seed) and the slot one-hot are appended at the end.
    Pruned vs. earlier versions: heading channels, all target/density features,
    the hop gradient, the dead LEFT/RIGHT action slots, and the sel-flat block.
    """
    assert cfg is not None, "cfg is required"
    B, N = rotations.shape
    v_max = float(cfg["v_max"])
    spacing = float(cfg["target_spacing"])
    body_v = world_to_body(last_velocity, rotations)             # [B, N, 2]
    onehot = torch.nn.functional.one_hot(
        last_action.clamp(min=0, max=N_ACTIONS - 1), num_classes=N_ACTIONS).float()

    base = torch.stack(
        [
            body_v[..., 0] / v_max,                                          # forward speed
            min_neighbor_dist / spacing,                                     # nearest-neighbour dist
            blocked_flag.float(),                                            # blocked flag (0/1)
            (hop.float() / float(cfg["hop_max"])).clamp_max(1.0),            # hops to nearest congestion
            collision_risk.float() / float(N),                              # neighbours within risk radius
            (stuck_counter.float()
             / float(cfg["blocked_steps_threshold"])).clamp_max(2.0),       # stuck duration
            onehot[..., ACT_STOP],                                           # last action = STOP
            onehot[..., ACT_FORWARD],                                        # last action = FORWARD
        ],
        dim=-1,
    )                                                                        # [B, N, 8]

    nodes = base

    # latent z (role seed) appended after the base block; slot one-hot after that.
    if latent_z is not None:
        if latent_z.shape != (B, N, LATENT_Z_DIM):
            raise ValueError(
                f"latent_z shape {tuple(latent_z.shape)} != (B={B}, N={N}, "
                f"LATENT_Z_DIM={LATENT_Z_DIM})"
            )
        nodes = torch.cat([nodes, latent_z], dim=-1)                         # base + latent

        # Append the N-dim slot one-hot after latent_z (caller controls slot_dim).
        if sector_one_hot is not None:
            slot_dim = int(sector_one_hot.shape[-1])
            if sector_one_hot.shape != (B, N, slot_dim):
                raise ValueError(
                    f"sector_one_hot shape {tuple(sector_one_hot.shape)} != "
                    f"(B={B}, N={N}, slot_dim={slot_dim})"
                )
            nodes = torch.cat([nodes, sector_one_hot], dim=-1)               # base + latent + slot
            expected = NODE_FEATURE_DIM + LATENT_Z_DIM + slot_dim
        else:
            expected = NODE_FEATURE_DIM + LATENT_Z_DIM                       # base + latent only
    else:
        expected = NODE_FEATURE_DIM

    if nodes.shape[-1] != expected:
        raise RuntimeError(
            f"node feature dim {nodes.shape[-1]} != expected={expected}"
        )
    return nodes


def build_edges(positions: torch.Tensor, rotations: torch.Tensor,
                last_velocity: torch.Tensor, adj: torch.Tensor,
                cfg: dict,
                latent_z: torch.Tensor | None = None,
                slot_assignment: torch.Tensor | None = None,
                slot_pair_dist: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build PyG-compatible (edge_index, edge_attr) for a batch of N-node graphs.

    Edge convention: for edge `src -> dst`, edge_attr describes src as seen
    from dst's body frame (so a GATv2Conv aggregating into `dst` reads geometry
    relative to itself). Rotation is therefore keyed on `dst`.

    Node ordering: deterministic, `env_idx * N + agent_idx` (matches `obs_nodes`
    flattening in `model.py::Actor.forward`).

    Self-shape extension (R7+R15): if `latent_z` is provided ([B, N, 4]), append
    the source agent's RAW z_src (not diff) as 4 extra dims, so EDGE_FEATURE_DIM
    becomes 13 (= 9 geom + 4 raw z_src). Raw rather than diff because:
    1) symmetry breaking is the goal — diff preserves agent permutation
       invariance which we want to break;
    2) raw lets neighbors directly read each other's z without the network
       having to learn an addition op.
    """
    B, N, _ = positions.shape
    spacing = float(cfg["target_spacing"])
    diff_ij = _pairwise_diff(positions, positions)               # [B, N, N, 2] = pos_i - pos_j
    # rotations broadcast on dst axis: [b, i, j] -> rotations[b, j]
    rot_dst = rotations.unsqueeze(1).expand(B, N, N).reshape(-1)
    diff_body = world_to_body(diff_ij.reshape(-1, 2), rot_dst).reshape(B, N, N, 2)
    d_ij = diff_ij.norm(dim=-1)                                  # [B, N, N]
    bearing = torch.atan2(diff_body[..., 1], diff_body[..., 0])  # [B, N, N] from dst's view
    rel_heading = rotations.unsqueeze(1) - rotations.unsqueeze(2)  # heading_src - heading_dst
    rel_vel_world = last_velocity.unsqueeze(1) - last_velocity.unsqueeze(2)  # vel_src - vel_dst
    rel_vel_body = world_to_body(rel_vel_world.reshape(-1, 2), rot_dst).reshape(B, N, N, 2)
    vmax = float(cfg["v_max"])
    edge_feats_full = torch.stack(
        [
            diff_body[..., 0] / spacing,
            diff_body[..., 1] / spacing,
            d_ij / spacing,
            bearing.sin(),
            bearing.cos(),
            rel_heading.sin(),
            rel_heading.cos(),
            rel_vel_body[..., 0] / vmax,
            rel_vel_body[..., 1] / vmax,
        ],
        dim=-1,
    )                                                            # [B, N, N, EDGE_FEATURE_DIM]

    # --- flatten to PyG batched format --------------------------------------
    nz = adj.nonzero(as_tuple=False)                              # [E, 3]: (b, src, dst)
    batch_idx, src_local, dst_local = nz[:, 0], nz[:, 1], nz[:, 2]
    # PyG batched node index: batch * N + node
    src = batch_idx * N + src_local
    dst = batch_idx * N + dst_local
    edge_index = torch.stack([src, dst], dim=0)                  # [2, E]
    edge_attr = edge_feats_full[batch_idx, src_local, dst_local] # [E, 9]

    # R7+R15: append raw z_src per edge if latent provided.
    if latent_z is not None:
        if latent_z.shape != (B, N, LATENT_Z_DIM):
            raise ValueError(
                f"latent_z shape {tuple(latent_z.shape)} != (B={B}, N={N}, "
                f"LATENT_Z_DIM={LATENT_Z_DIM})"
            )
        z_src = latent_z[batch_idx, src_local]                    # [E, 4]
        edge_attr = torch.cat([edge_attr, z_src], dim=-1)         # [E, 13]
    # v17b': append desired_pair_dist[slot_src, slot_dst] / spacing per edge if provided.
    # This closes the info bottleneck — actor can now read the same role-distance
    # constraint the hand-coded controller uses, instead of having to recover it
    # from raw z (an unsolvable global ranking problem locally).
    if slot_assignment is not None and slot_pair_dist is not None:
        slot_src = slot_assignment[batch_idx, src_local]          # [E] long in [0, N-1]
        slot_dst = slot_assignment[batch_idx, dst_local]          # [E] long in [0, N-1]
        if slot_pair_dist.ndim == 3:                              # [B, N, N] multi-shape
            d_desired = slot_pair_dist[batch_idx, slot_src, slot_dst] / spacing
        else:                                                      # [N, N] single-shape
            d_desired = slot_pair_dist[slot_src, slot_dst] / spacing
        edge_attr = torch.cat([edge_attr, d_desired.unsqueeze(-1)], dim=-1)  # [E, 14]
    return edge_index, edge_attr
