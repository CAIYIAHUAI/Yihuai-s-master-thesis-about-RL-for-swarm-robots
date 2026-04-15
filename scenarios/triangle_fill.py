import math
from typing import Optional
import torch
from vmas import render_interactively
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from utils import patch_world_for_fast_sphere_collisions
from utils.triangle_reward import (
    build_triangle_template,
    build_template_knn_signatures,
    make_formation_soft_permutation_fn,
    per_agent_lattice_loss,
    scale_invariant_distance_loss,
    squared_distance_matrix_batched,
)

class Kilobot4ActionDynamics(Dynamics):
    """
    Action space (strict 4):
    - 0: STOP
    - 1: LEFT
    - 2: RIGHT
    - 3: STRAIGHT

    VMAS discrete actions are internally mapped to a scalar u in [-u_range, u_range].
    With nvec=[4] and u_range=1.0 the mapping is:
      0 -> -1.0, 1 -> -1/3, 2 -> +1/3, 3 -> +1.0
    We decode u back to {0,1,2,3} via binning.
    """

    def __init__(self, world: World, v0: float, w0: float, v_turn: float):
        super().__init__()
        self.dt = world.dt
        self.world = world
        self.v0 = float(v0)
        self.w0 = float(w0)
        self.v_turn = float(v_turn)  # forward speed used during LEFT/RIGHT (curved turning)

    @property
    def needed_action_size(self) -> int:
        return 1

    def _decode_action_id(self) -> torch.Tensor:
        u = self.agent.action.u[:, 0]
        a0 = u <= (-2.0 / 3.0)
        a1 = (u > (-2.0 / 3.0)) & (u <= 0.0)
        a2 = (u > 0.0) & (u <= (2.0 / 3.0))
        action_id = torch.zeros_like(u, dtype=torch.long)
        action_id[a1] = 1
        action_id[a2] = 2
        action_id[~(a0 | a1 | a2)] = 3
        return action_id

    def process_action(self):
        action_id = self._decode_action_id()

        v_cmd = torch.zeros(self.agent.batch_dim, device=self.agent.device)
        w_cmd = torch.zeros(self.agent.batch_dim, device=self.agent.device)

        v_cmd[action_id == 3] = self.v0

        # Turning is a curved motion: add a small forward component during LEFT/RIGHT.
        v_cmd[action_id == 1] = self.v_turn
        v_cmd[action_id == 2] = self.v_turn

        w_cmd[action_id == 1] = self.w0
        w_cmd[action_id == 2] = -self.w0

        theta = self.agent.state.rot[:, 0]
        dx = v_cmd * torch.cos(theta)
        dy = v_cmd * torch.sin(theta)
        dtheta = w_cmd
        delta_state = self.dt * torch.stack((dx, dy, dtheta), dim=-1)  # [B,3]

        v_cur_x = self.agent.state.vel[:, 0]
        v_cur_y = self.agent.state.vel[:, 1]
        v_cur_angular = self.agent.state.ang_vel[:, 0]

        acc_x = (delta_state[:, 0] - v_cur_x * self.dt) / (self.dt**2)
        acc_y = (delta_state[:, 1] - v_cur_y * self.dt) / (self.dt**2)
        acc_ang = (delta_state[:, 2] - v_cur_angular * self.dt) / (self.dt**2)

        self.agent.state.force[:, 0] = self.agent.mass * acc_x
        self.agent.state.force[:, 1] = self.agent.mass * acc_y
        self.agent.state.torque = (self.agent.moment_of_inertia * acc_ang).unsqueeze(-1)

class Scenario(BaseScenario):
    def make_world(self, batch_dim:int, device:torch.device, **kwargs) -> World:
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]
        self.n_agents = int(kwargs.pop("n_agents", 30))
        self.share_reward = bool(kwargs.pop("share_reward", True)) # I dont want agents to be selfish

        self.mm_to_unit = float(kwargs.pop("mm_to_unit", 1e-3))
        self.world_semidim_mm = float(kwargs.pop("world_semidim_mm", 350.0))
        self.comm_r_mm = float(kwargs.pop("comm_r_mm", 85.0))
        self.agent_radius_mm = float(kwargs.pop("agent_radius_mm", 16.5))

        self.v0_mm_s = float(kwargs.pop("v0_mm_s", 20.0)) #Forward speed (20mm/s), popped from kwargs for validation.
        self.w0_rad_s = float(kwargs.pop("w0_rad_s", 1.5)) # Turning speed (1.5 rad/s), controls spin rate for turns.
        self.dt = float(kwargs.pop("dt", 0.0416666)) # Sim timestep (1/24s), matches standard Kilobot controller frequency.

        # Motion model: in real kilobots, turning is typically a curved motion (not pure in-place rotation).
        # We add a small forward component during LEFT/RIGHT to improve exploration and realism.
        self.turn_v_frac = float(kwargs.pop("turn_v_frac", 0.3))

        # Structure reward settings (used only in formation mode).
        self.formation_w = float(kwargs.pop("formation_w", 1.0))
        self.target_spacing_mm = float(kwargs.pop("target_spacing_mm", 45.0))
        self.spacing_w = float(kwargs.pop("spacing_w", 1.0))
        # Flat-bottom spacing band:
        # - default lo=hi=1.0 preserves the legacy squared error exactly
        # - wider bands can be enabled in configs to treat spacing as a soft constraint
        self.spacing_ratio_lo = float(kwargs.pop("spacing_ratio_lo", 1.0))
        self.spacing_ratio_hi = float(kwargs.pop("spacing_ratio_hi", 1.0))
        self.boundary_spacing_scale = float(kwargs.pop("boundary_spacing_scale", 1.0))
        # Optional direct shape reward hook. Default 0.0 keeps legacy reward behavior.
        self.triangle_w = float(kwargs.pop("triangle_w", 0.0))
        self.boundary_frac = float(kwargs.pop("boundary_frac", 0.35))
        self.corner_peak_ratio = float(kwargs.pop("corner_peak_ratio", 1.15))
        self.triangle_shape_w_tri = float(kwargs.pop("triangle_shape_w_tri", 0.45))
        self.triangle_shape_w_corner = float(kwargs.pop("triangle_shape_w_corner", 0.30))
        self.triangle_shape_w_straight = float(kwargs.pop("triangle_shape_w_straight", 0.25))
        self.lattice_w = float(kwargs.pop("lattice_w", 0.0))
        self.lattice_k = int(kwargs.pop("lattice_k", 6))
        self.progress_reward = bool(kwargs.pop("progress_reward", True))
        self.mixed_reward = bool(kwargs.pop("mixed_reward", False))
        self.success_bonus = float(kwargs.pop("success_bonus", 0.05))
        self.success_threshold = float(kwargs.pop("success_threshold", 0.05))
        self.formation_sinkhorn_tau = float(kwargs.pop("formation_sinkhorn_tau", 0.001))
        self.formation_sinkhorn_iters = int(kwargs.pop("formation_sinkhorn_iters", 30))
        self.formation_eps = float(kwargs.pop("formation_eps", 1e-8))
        self.formation_template_seed = int(kwargs.pop("formation_template_seed", 0))
        self.safe_collision_w = float(kwargs.pop("safe_collision_w", 0.5))
        self.safe_action_w = float(kwargs.pop("safe_action_w", 0.02))
        self.torch_compile = bool(kwargs.pop("torch_compile", False))
        self.fast_collisions = bool(kwargs.pop("fast_collisions", False))

        # Per-agent target assignment (potential-based shaping toward assigned template positions).
        self.target_progress_w = float(kwargs.pop("target_progress_w", 0.0))
        self.target_success_radius_mm = float(kwargs.pop("target_success_radius_mm", 33.0))
        self.target_success_bonus = float(kwargs.pop("target_success_bonus", 0.0))

        # Spawn (pile) config for formation training.
        # We keep the default spawn below origin to avoid immediate dense collisions.
        self.pile_center_mm = kwargs.pop("pile_center_mm", (0.0, -225.0))
        # Optional curriculum / domain randomization: randomize the pile center Y each reset.
        # Format: (y_min_mm, y_max_mm). If provided, this overrides the Y component of pile_center_mm.
        self.pile_center_y_mm_range = kwargs.pop("pile_center_y_mm_range", None)
        if self.pile_center_y_mm_range is not None:
            self.pile_center_y_mm_range = (float(self.pile_center_y_mm_range[0]), float(self.pile_center_y_mm_range[1]))
        # Half-width (mm) of the spawn box. Keep it large enough so 30 robots can be placed without failures.
        self.pile_halfwidth_mm = float(kwargs.pop("pile_halfwidth_mm", 120.0))

        # Observation normalization (keeps network inputs roughly O(1) to help optimization).
        self.normalize_obs = bool(kwargs.pop("normalize_obs", True))
        # Backward-compat: optionally fit observation vector to a fixed dimension (useful when loading checkpoints).
        # If None (default), no fitting is applied and obs_dim is determined by the current observation() definition.
        # 兼容旧模型：可选将观测向量适配到固定维度（加载 checkpoint 时很有用）。
        # 若为 None（默认），不做适配，obs_dim 由当前 observation() 定义决定。
        self.obs_pad_to_dim = kwargs.pop("obs_pad_to_dim", None)
        if self.obs_pad_to_dim is not None:
            self.obs_pad_to_dim = int(self.obs_pad_to_dim)

        # Top-K neighbor observation: how many nearest neighbors to include in observation
        self.obs_top_k_neighbors = int(kwargs.pop("obs_top_k_neighbors", 8))

        # Goal-relative observation: include position/heading relative to formation center
        self.obs_include_goal_rel = bool(kwargs.pop("obs_include_goal_rel", False))
        # In GNN ablations, keep observations self-only so all neighbor information must flow through graph messages.
        self.gnn_obs_self_only = bool(kwargs.pop("gnn_obs_self_only", False))
        # Formation center (where robots should converge to form triangle)
        self.formation_center_mm = kwargs.pop("formation_center_mm", (0.0, 0.0))

        if self.mixed_reward and self.progress_reward:
            raise ValueError("mixed_reward currently requires progress_reward=False.")

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Convert map boundary size from mm to simulation units.
        self.world_semidim = self.world_semidim_mm * self.mm_to_unit
        # Convert neighbor communication range from mm to simulation units.
        self.comm_r = self.comm_r_mm * self.mm_to_unit
        # Convert robot collision radius from mm to simulation units.
        self.agent_radius = self.agent_radius_mm * self.mm_to_unit
        # Convert forward speed from mm/s to simulation units/s.
        self.v0 = self.v0_mm_s * self.mm_to_unit
        # Angular speed (rad/s) is unit-agnostic, so keep as is.
        self.w0 = self.w0_rad_s
        # Formation center in simulation units
        self.formation_center = torch.tensor(
            [self.formation_center_mm[0] * self.mm_to_unit, self.formation_center_mm[1] * self.mm_to_unit],
            device=device,
            dtype=torch.float32,
        )

        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=self.dt,
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
            dim_c=0,
        )

        self.viewer_zoom = 1.2 ## Sets the default camera zoom level for the visualizer window.
        self.visualize_semidims = True

        for i in range(self.n_agents):
                    agent = Agent(
                        name=f"agent_{i}",
                        shape=Sphere(radius=self.agent_radius),
                        collide=True,
                        color=Color.BLUE,
                        action_size=1,
                        discrete_action_nvec=[4],
                        u_range=1.0,
                        u_multiplier=1.0,
                        dynamics=Kilobot4ActionDynamics(world=world, v0=self.v0, w0=self.w0, v_turn=self.turn_v_frac * self.v0),
                    )
                    world.add_agent(agent)

        self.fast_collisions_enabled = False
        if self.fast_collisions:
            enabled, reason = patch_world_for_fast_sphere_collisions(world)
            self.fast_collisions_enabled = enabled
            status = "enabled" if enabled else "skipped"
            print(f"triangle_fill fast collisions {status}: {reason}")

        # Build deterministic formation template once (shape only; not tied to world coordinates).
        tmpl = build_triangle_template(
            n_agents=self.n_agents,
            seed=self.formation_template_seed,
            side_length=1.0,
        ).to(device=device, dtype=torch.float32)
        self._formation_template_points = tmpl.clone()  # [N, 2] raw template for target assignment
        self._formation_template_dist2 = squared_distance_matrix_batched(tmpl.unsqueeze(0))[0]  # [N,N]
        self._formation_soft_perm = make_formation_soft_permutation_fn(
            tau=self.formation_sinkhorn_tau,
            iters=self.formation_sinkhorn_iters,
            eps=self.formation_eps,
            compile_enabled=self.torch_compile and device_type == "cuda",
        )
        if self.lattice_w > 0.0:
            self._template_knn_sigs = build_template_knn_signatures(
                tmpl,
                k=self.lattice_k,
                eps=self.formation_eps,
            )
        else:
            self._template_knn_sigs = None
        self.target_spacing = self.target_spacing_mm * self.mm_to_unit

        self._rew_per_agent = None
        self._shared_rew = None
        self._info_cache = None
        self._prev_struct_loss = None
        self._prev_spacing_loss = None
        self._prev_lattice_loss = None
        self._prev_per_agent_spacing_loss = None
        self._prev_per_agent_lattice_loss = None

        # Per-agent target assignment state (lazily initialized on first reset)
        self._target_per_agent = None   # [B, N, 2]
        self._prev_target_dist = None   # [B, N]
        self.target_success_radius = self.target_success_radius_mm * self.mm_to_unit

        return world

    def _assign_targets_for_env(self, env_index: int) -> None:
        """Compute per-agent targets for one env via Hungarian on best-oriented template."""
        if self._target_per_agent is None:
            B = self.world.batch_dim
            self._target_per_agent = torch.zeros(
                (B, self.n_agents, 2), device=self.world.device, dtype=torch.float32
            )
            self._prev_target_dist = torch.full(
                (B, self.n_agents), -1.0, device=self.world.device, dtype=torch.float32
            )

        pos = torch.stack(
            [a.state.pos[env_index] for a in self.world.agents], dim=0
        )  # [N, 2]

        # Scale template so NN distance ≈ target_spacing
        tmpl = self._formation_template_points.clone()  # [N, 2], centroid ≈ origin
        tmpl_dist = torch.cdist(tmpl.unsqueeze(0), tmpl.unsqueeze(0))[0]
        tmpl_dist.fill_diagonal_(float("inf"))
        tmpl_nn = tmpl_dist.min(dim=-1).values.mean()
        scale = self.target_spacing / max(float(tmpl_nn), 1e-8)
        tmpl_scaled = tmpl * scale  # [N, 2], still centroid ≈ origin

        # Anchor at formation_center (NOT swarm centroid)
        anchor = self.formation_center  # [2]

        # Try 12 candidate orientations (6 rotations × 2 mirrors) and pick best
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        best_cost = float("inf")
        best_targets = None
        best_col_idx = None

        for do_mirror in [False, True]:
            for k in range(6):
                angle = k * math.pi / 3.0  # 0, 60, 120, 180, 240, 300 degrees
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rot_mat = torch.tensor(
                    [[cos_a, -sin_a], [sin_a, cos_a]],
                    device=tmpl_scaled.device, dtype=tmpl_scaled.dtype,
                )
                candidate = tmpl_scaled @ rot_mat.T  # [N, 2]
                if do_mirror:
                    candidate[:, 0] = -candidate[:, 0]
                candidate = candidate + anchor  # translate to formation_center

                # Hungarian assignment: minimize total squared distance
                cost = ((pos.unsqueeze(1) - candidate.unsqueeze(0)) ** 2).sum(dim=-1)  # [N, N]
                cost_np = cost.detach().cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)
                total_cost = cost_np[row_idx, col_idx].sum()
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_targets = candidate
                    best_col_idx = col_idx

        col_idx_t = torch.from_numpy(np.array(best_col_idx)).to(
            device=best_targets.device, dtype=torch.long
        )
        self._target_per_agent[env_index] = best_targets[col_idx_t]
        # Set prev_target_dist as final state for this env (审稿修正 #5)
        self._prev_target_dist[env_index] = (
            pos - self._target_per_agent[env_index]
        ).norm(dim=-1)

    def reset_world_at(self, env_index: Optional[int] = None):
        # VMAS may call reset_world_at(None) to reset all envs at once. We implement that
        # by looping over env indices so we can sample a different pile_center_y per env.
        if env_index is None:
            for i in range(self.world.batch_dim):
                self.reset_world_at(env_index=i)
            # Clear caches once after the whole batch reset.
            self._rew_per_agent = None
            self._shared_rew = None
            self._info_cache = None
            self._prev_struct_loss = None
            self._prev_spacing_loss = None
            self._prev_lattice_loss = None
            self._prev_per_agent_spacing_loss = None
            self._prev_per_agent_lattice_loss = None
            return

        # Resolve pile center for this env.
        center_x_mm = float(self.pile_center_mm[0])
        center_y_mm = float(self.pile_center_mm[1])
        if self.pile_center_y_mm_range is not None:
            y0, y1 = self.pile_center_y_mm_range
            # Sample on the simulator device for reproducible RNG behavior under torch seeds.
            center_y_mm = float(torch.empty((), device=self.world.device).uniform_(y0, y1).item())

        center = torch.tensor(
            [center_x_mm * self.mm_to_unit, center_y_mm * self.mm_to_unit],
            device=self.world.device,
            dtype=torch.float32,
        )
        half = self.pile_halfwidth_mm * self.mm_to_unit
        x_bounds = (center[0].item() - half, center[0].item() + half)
        y_bounds = (center[1].item() - half, center[1].item() + half)

        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=2.01 * self.agent_radius,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            disable_warn=True,
        )

        # Randomize headings for this env.
        for agent in self.world.agents:
            agent.set_rot(
                torch.empty((1,), device=self.world.device, dtype=torch.float32).uniform_(-math.pi, math.pi),
                batch_index=env_index,
            )

        # Per-agent target assignment (after spawn + heading are finalized)
        if self.target_progress_w > 0.0:
            self._assign_targets_for_env(env_index)

        self._rew_per_agent = None
        self._shared_rew = None
        self._info_cache = None
        if self._prev_struct_loss is not None:
            self._prev_struct_loss[env_index] = -1.0
        if self._prev_spacing_loss is not None:
            self._prev_spacing_loss[env_index] = -1.0
        if self._prev_lattice_loss is not None:
            self._prev_lattice_loss[env_index] = -1.0
        if self._prev_per_agent_spacing_loss is not None:
            self._prev_per_agent_spacing_loss[env_index] = -1.0
        if self._prev_per_agent_lattice_loss is not None:
            self._prev_per_agent_lattice_loss[env_index] = -1.0

    def _compute_metrics(self):
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1) # [B,N,2] collect all robots' (x,y)
        vel = torch.stack([a.state.vel for a in self.world.agents], dim=1) # [B,N,2] collect all robots' (vx,vy)
        dist_mat = torch.cdist(pos, pos) # [B, N, N],算出每两个机器人之间的距离。这是一个30 by 30的矩阵。
        overlap = (2 * self.agent_radius - dist_mat).clamp_min(0.0)
        overlap = overlap - torch.diag_embed(torch.diagonal(overlap, dim1=1, dim2=2))
        collision_pen = overlap.sum(dim=-1)  # [B,N]
        # Action cost (for "braking"/settling):
        # IMPORTANT: VMAS discrete actions are mapped to a scalar u in [-1,1], e.g. STOP=0 -> u=-1.
        # If we used u^2, STOP would be penalized as much as STRAIGHT, which is the opposite of "brake".
        # So we decode action id and assign a simple cost:
        #   STOP: 0, LEFT/RIGHT/STRAIGHT: 1 (you can refine later).
        action_u = torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1)  # [B,N]
        # Decode u back to {0,1,2,3} by binning in [-1,1].
        bins = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=self.world.device, dtype=action_u.dtype)
        # Compute nearest bin index (stable even with tiny numeric noise).
        action_id = (action_u.unsqueeze(-1) - bins.view(1, 1, -1)).abs().argmin(dim=-1)  # [B,N] in {0,1,2,3}
        action_cost = (action_id != 0).float()  # [B,N]
        speed_mean = vel.norm(dim=-1).mean(dim=-1)  # [B] mean speed across agents
        return pos, collision_pen, action_cost, speed_mean

    def _formation_terms(
        self, pos: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return global logging terms plus per-agent formation terms."""
        dist2 = squared_distance_matrix_batched(pos)  # [B,N,N]
        soft_perm = self._formation_soft_perm(dist2, self._formation_template_dist2)

        tmpl = self._formation_template_dist2.unsqueeze(0).expand(pos.shape[0], -1, -1)
        dist2_soft = torch.matmul(torch.matmul(soft_perm, tmpl), soft_perm.transpose(1, 2))
        shape_loss = scale_invariant_distance_loss(dist2, dist2_soft, eps=self.formation_eps)  # [B]
        row_a = dist2 - dist2.mean(dim=-1, keepdim=True)
        row_b = dist2_soft - dist2_soft.mean(dim=-1, keepdim=True)
        row_dot = (row_a * row_b).sum(dim=-1)
        row_norm = row_a.norm(dim=-1) * row_b.norm(dim=-1) + self.formation_eps
        per_agent_shape_loss = 1.0 - row_dot / row_norm  # [B,N]

        dist_mat = dist2.sqrt()
        inf_diag = torch.full((pos.shape[0], pos.shape[1]), float("inf"), device=pos.device, dtype=dist_mat.dtype)
        dist_mat = dist_mat + torch.diag_embed(inf_diag)
        k = min(3, pos.shape[1] - 1)
        knn = dist_mat.topk(k, dim=-1, largest=False).values
        per_agent_3nn = knn.mean(dim=-1)
        ratio = per_agent_3nn / max(self.target_spacing, 1e-8)
        spacing_lo = self.spacing_ratio_lo
        spacing_hi = self.spacing_ratio_hi
        if spacing_lo < spacing_hi:
            per_agent_spacing_loss = torch.where(
                ratio < spacing_lo,
                (ratio - spacing_lo).pow(2),
                torch.where(
                    ratio > spacing_hi,
                    (ratio - spacing_hi).pow(2),
                    torch.zeros_like(ratio),
                ),
            )
        else:
            per_agent_spacing_loss = (ratio - 1.0).pow(2)
        if self._template_knn_sigs is not None:
            per_agent_lattice = per_agent_lattice_loss(
                dist_mat,
                self._template_knn_sigs,
                k=self.lattice_k,
                eps=self.formation_eps,
            )
            lattice_loss = per_agent_lattice.mean(dim=-1)
        else:
            per_agent_lattice = torch.zeros_like(per_agent_spacing_loss)
            lattice_loss = torch.zeros(pos.shape[0], device=pos.device, dtype=dist_mat.dtype)

        boundary_mask, _, boundary_pos = self._boundary_topk_mask(pos)
        if self.boundary_spacing_scale != 1.0:
            boundary_scale = torch.where(
                boundary_mask,
                torch.full_like(per_agent_spacing_loss, self.boundary_spacing_scale),
                torch.ones_like(per_agent_spacing_loss),
            )
            per_agent_spacing_loss = per_agent_spacing_loss * boundary_scale
        spacing_loss = per_agent_spacing_loss.mean(dim=-1)
        triangularity = self._boundary_triangularity(pos, boundary_pos)
        boundary_peak_count, boundary_corner_score = self._boundary_corner_score(pos, boundary_pos)
        boundary_straightness = self._boundary_straightness_score(pos, boundary_pos)
        triangle_shape_score = (
            self.triangle_shape_w_tri * triangularity
            + self.triangle_shape_w_corner * boundary_corner_score
            + self.triangle_shape_w_straight * boundary_straightness
        )
        entropy = -(soft_perm * (soft_perm + self.formation_eps).log()).sum(dim=-1).mean(dim=-1)  # [B]
        return (
            shape_loss,
            spacing_loss,
            triangularity,
            entropy,
            per_agent_shape_loss,
            per_agent_spacing_loss,
            per_agent_lattice,
            lattice_loss,
            boundary_peak_count,
            boundary_corner_score,
            boundary_straightness,
            triangle_shape_score,
        )

    def _boundary_topk_mask(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the farthest agents from the centroid as boundary points."""
        centroid = pos.mean(dim=1, keepdim=True)  # [B, 1, 2]
        dist_to_centroid = (pos - centroid).norm(dim=-1)  # [B, N]
        k_boundary = max(3, int(math.ceil(self.boundary_frac * self.n_agents)))
        k_boundary = min(k_boundary, self.n_agents)
        _, boundary_idx = dist_to_centroid.topk(k_boundary, dim=-1, largest=True, sorted=False)  # [B, K]
        boundary_pos = torch.gather(
            pos,
            1,
            boundary_idx.unsqueeze(-1).expand(-1, -1, pos.shape[-1]),
        )  # [B, K, 2]
        boundary_mask = torch.zeros(
            (pos.shape[0], pos.shape[1]),
            device=pos.device,
            dtype=torch.bool,
        )
        boundary_mask.scatter_(1, boundary_idx, True)
        return boundary_mask, boundary_idx, boundary_pos

    def _boundary_triangularity(self, pos: torch.Tensor, boundary_pos: torch.Tensor) -> torch.Tensor:
        """3-fold Fourier coefficient of the boundary radial profile around the global centroid."""
        centroid = pos.mean(dim=1, keepdim=True)  # [B, 1, 2]
        boundary_centered = boundary_pos - centroid  # [B, K, 2]
        r = boundary_centered.norm(dim=-1)  # [B, K]
        theta = torch.atan2(boundary_centered[:, :, 1], boundary_centered[:, :, 0])  # [B, K]
        sort_idx = theta.argsort(dim=-1)
        r_sorted = torch.gather(r, -1, sort_idx)
        theta_sorted = torch.gather(theta, -1, sort_idx)
        r_cos3 = (r_sorted * torch.cos(3.0 * theta_sorted)).mean(dim=-1)  # [B]
        r_sin3 = (r_sorted * torch.sin(3.0 * theta_sorted)).mean(dim=-1)  # [B]
        r_mean = r.mean(dim=-1).clamp_min(1e-8)  # [B]
        return torch.sqrt(r_cos3.square() + r_sin3.square()) / r_mean

    def _boundary_corner_score(self, pos: torch.Tensor, boundary_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Count prominent peaks in the sorted boundary radial profile."""
        centroid = pos.mean(dim=1, keepdim=True)  # [B, 1, 2]
        boundary_centered = boundary_pos - centroid  # [B, K, 2]
        r = boundary_centered.norm(dim=-1)  # [B, K]
        theta = torch.atan2(boundary_centered[:, :, 1], boundary_centered[:, :, 0])  # [B, K]
        sort_idx = theta.argsort(dim=-1)
        r_sorted = torch.gather(r, -1, sort_idx)
        r_mean = r_sorted.mean(dim=-1, keepdim=True).clamp_min(1e-8)
        r_prev = torch.roll(r_sorted, shifts=1, dims=-1)
        r_next = torch.roll(r_sorted, shifts=-1, dims=-1)
        peak_thresh = r_mean * self.corner_peak_ratio
        is_peak = (r_sorted > r_prev) & (r_sorted >= r_next) & (r_sorted > peak_thresh)
        peak_count = is_peak.float().sum(dim=-1)
        corner_score = torch.exp(-0.5 * (peak_count - 3.0).square())
        return peak_count, corner_score

    def _boundary_straightness_score(self, pos: torch.Tensor, boundary_pos: torch.Tensor) -> torch.Tensor:
        """Measure whether boundary tangents align with three dominant edge directions."""
        centroid = pos.mean(dim=1, keepdim=True)  # [B, 1, 2]
        boundary_centered = boundary_pos - centroid  # [B, K, 2]
        theta = torch.atan2(boundary_centered[:, :, 1], boundary_centered[:, :, 0])  # [B, K]
        sort_idx = theta.argsort(dim=-1)
        boundary_sorted = torch.gather(
            boundary_pos,
            1,
            sort_idx.unsqueeze(-1).expand(-1, -1, boundary_pos.shape[-1]),
        )  # [B, K, 2]
        prev_pts = torch.roll(boundary_sorted, shifts=1, dims=1)
        next_pts = torch.roll(boundary_sorted, shifts=-1, dims=1)
        tangent = next_pts - prev_pts  # [B, K, 2]
        phi = torch.atan2(tangent[:, :, 1], tangent[:, :, 0])  # [B, K]
        orient_cos6 = torch.cos(6.0 * phi).mean(dim=-1)
        orient_sin6 = torch.sin(6.0 * phi).mean(dim=-1)
        return torch.sqrt(orient_cos6.square() + orient_sin6.square()).clamp(0.0, 1.0)

    def _build_formation_info(
        self,
        formation_loss: torch.Tensor,
        collision_pen: torch.Tensor,
        action_cost: torch.Tensor,
        speed_mean: torch.Tensor,
        sinkhorn_entropy: torch.Tensor,
        lattice_loss: torch.Tensor,
        triangularity: torch.Tensor,
        boundary_peak_count: torch.Tensor,
        boundary_corner_score: torch.Tensor,
        boundary_straightness: torch.Tensor,
        triangle_shape_score: torch.Tensor,
        local_spacing_progress_mean: torch.Tensor,
        local_lattice_progress_mean: torch.Tensor,
        global_shape_progress_mean: torch.Tensor,
    ) -> dict:
        collision_mean = collision_pen.mean(dim=-1)
        action_mean = action_cost.mean(dim=-1)
        return {
            "formation_loss": formation_loss,
            "lattice_loss": lattice_loss,
            "local_spacing_progress_mean": local_spacing_progress_mean,
            "local_lattice_progress_mean": local_lattice_progress_mean,
            "global_shape_progress_mean": global_shape_progress_mean,
            "formation_score": torch.exp(-formation_loss),
            "triangularity": triangularity,
            "boundary_triangularity": triangularity,
            "boundary_peak_count": boundary_peak_count,
            "boundary_corner_score": boundary_corner_score,
            "boundary_straightness": boundary_straightness,
            "triangle_shape_score": triangle_shape_score,
            "collision_mean": collision_mean,
            # Backward-compatible key for existing tooling.
            "collisions_mean": collision_mean,
            "action_mean": action_mean,
            "sinkhorn_entropy": sinkhorn_entropy,
            "speed_mean": speed_mean,
        }

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first or (self._rew_per_agent is None and self._shared_rew is None):
            pos, collision_pen, action_cost, speed_mean = self._compute_metrics()
            (
                shape_loss,
                spacing_loss,
                triangularity,
                sinkhorn_entropy,
                per_agent_shape_loss,
                per_agent_spacing_loss,
                per_agent_lattice,
                lattice_loss,
                boundary_peak_count,
                boundary_corner_score,
                boundary_straightness,
                triangle_shape_score,
            ) = self._formation_terms(pos)
            shape_progress = torch.zeros_like(shape_loss)
            spacing_progress_local = torch.zeros_like(per_agent_spacing_loss)
            lattice_progress_local = torch.zeros_like(per_agent_lattice)

            if self.mixed_reward:
                success_mask = (shape_loss < self.success_threshold).float().unsqueeze(-1)
                per_agent = (
                    -self.formation_w * shape_loss.unsqueeze(-1)
                    - self.spacing_w * per_agent_spacing_loss
                    - self.lattice_w * per_agent_lattice
                    - self.safe_collision_w * collision_pen
                    - self.safe_action_w * action_cost
                    + self.success_bonus * success_mask
                )
                # Per-agent target progress reward (potential-based shaping)
                if self.target_progress_w > 0.0 and self._target_per_agent is not None:
                    cur_target_dist = (pos - self._target_per_agent).norm(dim=-1)  # [B, N]
                    valid_prev = self._prev_target_dist >= 0.0  # [B, N]
                    target_progress = torch.where(
                        valid_prev,
                        self._prev_target_dist - cur_target_dist,
                        torch.zeros_like(cur_target_dist),
                    )  # [B, N]
                    self._prev_target_dist = cur_target_dist.clone()
                    target_arrived = (cur_target_dist < self.target_success_radius).float()
                    per_agent = per_agent + (
                        self.target_progress_w * target_progress
                        + self.target_success_bonus * target_arrived
                    )
            elif self.share_reward:
                if self.progress_reward:
                    if self._prev_struct_loss is None:
                        self._prev_struct_loss = shape_loss.clone()
                        self._prev_spacing_loss = spacing_loss.clone()
                        self._prev_lattice_loss = lattice_loss.clone()

                    valid_prev = self._prev_struct_loss >= 0.0
                    shape_progress = torch.where(
                        valid_prev, self._prev_struct_loss - shape_loss, torch.zeros_like(shape_loss)
                    )
                    spacing_progress = torch.where(
                        valid_prev, self._prev_spacing_loss - spacing_loss, torch.zeros_like(spacing_loss)
                    )
                    lattice_progress = torch.where(
                        valid_prev, self._prev_lattice_loss - lattice_loss, torch.zeros_like(lattice_loss)
                    )
                    self._prev_struct_loss = shape_loss.clone()
                    self._prev_spacing_loss = spacing_loss.clone()
                    self._prev_lattice_loss = lattice_loss.clone()
                    formation_reward = (
                        self.formation_w * shape_progress
                        + self.spacing_w * spacing_progress
                        + self.lattice_w * lattice_progress
                    )
                else:
                    formation_reward = (
                        -self.formation_w * shape_loss
                        - self.spacing_w * spacing_loss
                        - self.lattice_w * lattice_loss
                    )

                success_mask = (shape_loss < self.success_threshold).float()
                formation_reward = formation_reward + self.success_bonus * success_mask
                per_agent = (
                    formation_reward.unsqueeze(-1)
                    - self.safe_collision_w * collision_pen
                    - self.safe_action_w * action_cost
                )
            else:
                success_mask = (shape_loss < self.success_threshold).float().unsqueeze(-1)
                if self.progress_reward:
                    if self._prev_struct_loss is None:
                        self._prev_struct_loss = shape_loss.clone()
                    if self._prev_per_agent_spacing_loss is None:
                        self._prev_per_agent_spacing_loss = per_agent_spacing_loss.clone()
                    if self._prev_per_agent_lattice_loss is None:
                        self._prev_per_agent_lattice_loss = per_agent_lattice.clone()

                    valid_prev_shape = self._prev_struct_loss >= 0.0
                    valid_prev_spacing = self._prev_per_agent_spacing_loss >= 0.0
                    valid_prev_lattice = self._prev_per_agent_lattice_loss >= 0.0
                    shape_progress = torch.where(
                        valid_prev_shape, self._prev_struct_loss - shape_loss, torch.zeros_like(shape_loss)
                    )
                    spacing_progress_local = torch.where(
                        valid_prev_spacing,
                        self._prev_per_agent_spacing_loss - per_agent_spacing_loss,
                        torch.zeros_like(per_agent_spacing_loss),
                    )
                    lattice_progress_local = torch.where(
                        valid_prev_lattice,
                        self._prev_per_agent_lattice_loss - per_agent_lattice,
                        torch.zeros_like(per_agent_lattice),
                    )
                    self._prev_struct_loss = shape_loss.clone()
                    self._prev_per_agent_spacing_loss = per_agent_spacing_loss.clone()
                    self._prev_per_agent_lattice_loss = per_agent_lattice.clone()
                    per_agent = (
                        self.formation_w * shape_progress.unsqueeze(-1)
                        + self.spacing_w * spacing_progress_local
                        + self.lattice_w * lattice_progress_local
                        - self.safe_collision_w * collision_pen
                        - self.safe_action_w * action_cost
                        + self.success_bonus * success_mask
                    )
                else:
                    per_agent = (
                        -self.formation_w * per_agent_shape_loss
                        - self.spacing_w * per_agent_spacing_loss
                        - self.lattice_w * per_agent_lattice
                        - self.safe_collision_w * collision_pen
                        - self.safe_action_w * action_cost
                        + self.success_bonus * success_mask
                    )
            if self.triangle_w > 0.0:
                per_agent = per_agent + self.triangle_w * triangle_shape_score.unsqueeze(-1)
            self._info_cache = self._build_formation_info(
                formation_loss=shape_loss,
                collision_pen=collision_pen,
                action_cost=action_cost,
                speed_mean=speed_mean,
                sinkhorn_entropy=sinkhorn_entropy,
                lattice_loss=lattice_loss,
                triangularity=triangularity,
                boundary_peak_count=boundary_peak_count,
                boundary_corner_score=boundary_corner_score,
                boundary_straightness=boundary_straightness,
                triangle_shape_score=triangle_shape_score,
                local_spacing_progress_mean=spacing_progress_local.mean(dim=-1),
                local_lattice_progress_mean=lattice_progress_local.mean(dim=-1),
                global_shape_progress_mean=shape_progress,
            )
            self._info_cache["spacing_loss"] = spacing_loss
            if self.target_progress_w > 0.0 and self._target_per_agent is not None:
                self._info_cache["target_dist_mean"] = cur_target_dist.mean(dim=-1)
                self._info_cache["target_arrived_frac"] = target_arrived.mean(dim=-1)

            if self.share_reward and not self.mixed_reward:
                self._shared_rew = per_agent.mean(dim=-1)  # [B]
            else:
                self._rew_per_agent = per_agent

        if self.share_reward and not self.mixed_reward:
            return self._shared_rew

        idx = self.world.agents.index(agent)
        return self._rew_per_agent[:, idx]

    def _compute_neighbor_features_cache(self):
        """Compute Top-K local neighbor features for all agents.

        Cached tensor shapes:
        - self._neighbor_count: [B, N, 1]
        - self._top_k_neighbors: [B, N, K, 4] with
          [dx_body, dy_body, dist, valid]
        - self._local_geom_features: [B, N, 6] with
          [max_gap_size, gap_dir_cos, gap_dir_sin, linearity, tangent_cos2, tangent_sin2]
        """
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, N, 2]
        rot = torch.stack([a.state.rot for a in self.world.agents], dim=1)  # [B, N, 1]

        # 2. 计算所有agent之间的距离矩阵
        dist_matrix = torch.cdist(pos, pos)  # [B, N, N]

        # ⭐ 使用有限大数而非inf（避免GPU数值问题）
        BIG = 1e9  # 远大于任何实际距离即可

        # 3. 排除自己（对角线设为BIG）
        dist_matrix_masked = dist_matrix + torch.diag_embed(
            torch.full((pos.shape[0], pos.shape[1]), BIG, device=pos.device, dtype=dist_matrix.dtype)
        )

        # ⭐ 4. 限制在通信范围内：超出comm_r的距离设为BIG（关键！）
        dist_matrix_masked = torch.where(
            dist_matrix_masked <= self.comm_r,
            dist_matrix_masked,
            torch.full_like(dist_matrix_masked, BIG)
        )

        # ⭐ 5. 计算邻居数量（用于观测）- 注意用BIG/2作为阈值
        self._neighbor_count = (dist_matrix_masked < BIG / 2).float().sum(dim=-1, keepdim=True)  # [B, N, 1]

        # 6. 找到每个agent的Top-K最近邻居（仅在comm_r范围内）
        K = self.obs_top_k_neighbors
        # ⭐ topk会自动处理，即使K > 实际邻居数，只需在最后padding
        topk_dists, topk_indices = torch.topk(
            dist_matrix_masked, k=min(K, self.n_agents - 1), dim=-1, largest=False, sorted=True
        )  # [B, N, min(K, N-1)], [B, N, min(K, N-1)]

        # ⭐ 7. 判断哪些槽位是有效邻居（dist < BIG/2）
        topk_valid = (topk_dists < BIG / 2).float()  # [B, N, min(K, N-1)]

        # 8. 提取Top-K邻居的位置
        actual_k = topk_dists.shape[2]  # 实际返回的邻居数：min(K, N-1)
        batch_idx = torch.arange(pos.shape[0], device=pos.device).view(-1, 1, 1).expand(-1, pos.shape[1], actual_k)
        topk_pos = pos[batch_idx, topk_indices]  # [B, N, actual_k, 2]

        # 9. 计算相对位置（world frame）
        pos_expanded = pos.unsqueeze(2)  # [B, N, 1, 2]
        topk_rel_pos_world = topk_pos - pos_expanded  # [B, N, actual_k, 2]

        # ⭐ 10. 转换到body frame（关键！避免泄露世界坐标轴方向）
        # 旋转矩阵：[cos(θ), sin(θ); -sin(θ), cos(θ)]
        rot_expanded = rot.squeeze(-1).unsqueeze(2)  # [B, N, 1]
        cos_theta = torch.cos(rot_expanded)  # [B, N, 1]
        sin_theta = torch.sin(rot_expanded)  # [B, N, 1]

        # dx_body = cos(θ) * dx_world + sin(θ) * dy_world
        # dy_body = -sin(θ) * dx_world + cos(θ) * dy_world
        dx_world = topk_rel_pos_world[:, :, :, 0]  # [B, N, actual_k]
        dy_world = topk_rel_pos_world[:, :, :, 1]  # [B, N, actual_k]
        dx_body = cos_theta * dx_world + sin_theta * dy_world  # [B, N, actual_k]
        dy_body = -sin_theta * dx_world + cos_theta * dy_world  # [B, N, actual_k]
        topk_rel_pos_body = torch.stack([dx_body, dy_body], dim=-1)  # [B, N, actual_k, 2]

        topk_dists = torch.where(topk_valid > 0.5, topk_dists, torch.zeros_like(topk_dists))
        # Also zero out relative position for invalid slots.
        topk_rel_pos_body = topk_rel_pos_body * topk_valid.unsqueeze(-1)

        # ⭐ 12. 组合成完整的Top-K邻居特征：[dx_body, dy_body, dist, valid]
        topk_features = torch.cat([
            topk_rel_pos_body,               # [B, N, actual_k, 2] - dx_body, dy_body
            topk_dists.unsqueeze(-1),        # [B, N, actual_k, 1] - dist
            topk_valid.unsqueeze(-1)         # [B, N, actual_k, 1] - valid
        ], dim=-1)  # [B, N, actual_k, 4]

        # ⭐ 13. 简化padding逻辑：只在K > N-1时需要padding到K个槽位
        if actual_k < K:
            num_padding = K - actual_k
            padding = torch.zeros(
                (pos.shape[0], pos.shape[1], num_padding, 4),
                device=pos.device,
                dtype=topk_features.dtype
            )
            self._top_k_neighbors = torch.cat([topk_features, padding], dim=2)  # [B, N, K, 4]
        else:
            self._top_k_neighbors = topk_features  # [B, N, K, 4]

        # 14. Local role + boundary-shape features derived from Top-K body-frame neighbors.
        valid_mask = topk_valid > 0.5  # [B, N, actual_k]
        n_valid = valid_mask.sum(dim=-1)  # [B, N]
        n_valid_f = n_valid.to(topk_rel_pos_body.dtype)
        n_valid_safe = n_valid_f.clamp_min(1.0)

        dx_body = topk_rel_pos_body[:, :, :, 0]  # [B, N, actual_k]
        dy_body = topk_rel_pos_body[:, :, :, 1]  # [B, N, actual_k]
        angles = torch.atan2(dy_body, dx_body)  # [B, N, actual_k], [-pi, pi]

        inf_fill = torch.full_like(angles, BIG)
        angles_sorted, _ = torch.sort(torch.where(valid_mask, angles, inf_fill), dim=-1)

        if actual_k > 0:
            arange_km1 = torch.arange(max(actual_k - 1, 1), device=pos.device).view(1, 1, -1)
            if actual_k > 1:
                consecutive_gaps = angles_sorted[:, :, 1:] - angles_sorted[:, :, :-1]  # [B, N, actual_k-1]
                consecutive_valid = arange_km1[:, :, : actual_k - 1] < (n_valid - 1).unsqueeze(-1)
                consecutive_gaps = torch.where(
                    consecutive_valid,
                    consecutive_gaps,
                    torch.zeros_like(consecutive_gaps),
                )
            else:
                consecutive_gaps = torch.zeros_like(angles_sorted)

            last_valid_idx = (n_valid - 1).clamp_min(0).unsqueeze(-1)  # [B, N, 1]
            first_angle = angles_sorted[:, :, 0]
            last_valid_angle = angles_sorted.gather(-1, last_valid_idx).squeeze(-1)
            wrap_gap = torch.where(
                n_valid > 0,
                first_angle + (2.0 * math.pi) - last_valid_angle,
                torch.zeros_like(first_angle),
            )  # [B, N]

            gaps = torch.cat([consecutive_gaps, wrap_gap.unsqueeze(-1)], dim=-1)  # [B, N, actual_k]
            gap_start_angles = torch.cat(
                [angles_sorted[:, :, :-1], last_valid_angle.unsqueeze(-1)],
                dim=-1,
            )  # [B, N, actual_k]

            max_gap, max_gap_idx = gaps.max(dim=-1)  # [B, N], [B, N]
            gap_start = gap_start_angles.gather(-1, max_gap_idx.unsqueeze(-1)).squeeze(-1)
            gap_center = gap_start + 0.5 * max_gap
            max_gap_size = torch.where(
                n_valid > 0,
                (max_gap / (2.0 * math.pi)).clamp(0.0, 1.0),
                torch.zeros_like(max_gap),
            )
            gap_dir_cos = max_gap_size * torch.cos(gap_center)
            gap_dir_sin = max_gap_size * torch.sin(gap_center)
        else:
            max_gap_size = torch.zeros_like(n_valid_f)
            gap_dir_cos = torch.zeros_like(n_valid_f)
            gap_dir_sin = torch.zeros_like(n_valid_f)

        # Local line structure from the covariance of valid body-frame neighbor coordinates.
        mean_x = dx_body.sum(dim=-1) / n_valid_safe  # [B, N]
        mean_y = dy_body.sum(dim=-1) / n_valid_safe  # [B, N]
        centered_x = (dx_body - mean_x.unsqueeze(-1)) * topk_valid
        centered_y = (dy_body - mean_y.unsqueeze(-1)) * topk_valid

        xx = (centered_x * centered_x).sum(dim=-1) / n_valid_safe  # [B, N]
        yy = (centered_y * centered_y).sum(dim=-1) / n_valid_safe  # [B, N]
        xy = (centered_x * centered_y).sum(dim=-1) / n_valid_safe  # [B, N]

        delta = torch.sqrt((xx - yy).square() + 4.0 * xy.square() + self.formation_eps)
        trace = xx + yy
        linearity = torch.where(
            n_valid >= 2,
            (delta / (trace + self.formation_eps)).clamp(0.0, 1.0),
            torch.zeros_like(trace),
        )

        tangent_cos2 = torch.where(
            n_valid >= 2,
            linearity * ((xx - yy) / delta),
            torch.zeros_like(linearity),
        )
        tangent_sin2 = torch.where(
            n_valid >= 2,
            linearity * ((2.0 * xy) / delta),
            torch.zeros_like(linearity),
        )

        self._local_geom_features = torch.stack(
            [
                max_gap_size,
                gap_dir_cos,
                gap_dir_sin,
                linearity,
                tangent_cos2,
                tangent_sin2,
            ],
            dim=-1,
        )  # [B, N, 6]


    def observation(self, agent: Agent):
        """Per-agent local observation for formation task.

        Default feature layout:
        - vel_body: [2]
        - neighbor_count: [1]
        - top_k_neighbors: [K*4], each neighbor is [dx_body, dy_body, dist, valid]
        - local_geom_features: [6], each feature is
          [max_gap_size, gap_dir_cos, gap_dir_sin, linearity, tangent_cos2, tangent_sin2]

        If gnn_obs_self_only=True, only self features are exposed here:
        - vel_body: [2]
        - optional goal-relative features: [4]
        """

        # 第一次调用agent_0的observation时重新计算缓存
        is_first = agent == self.world.agents[0]
        if (not self.gnn_obs_self_only) and (is_first or not hasattr(self, '_neighbor_count')):
            self._compute_neighbor_features_cache()

        idx = self.world.agents.index(agent)

        # ⭐ 1. 提取自身速度并转换到body frame
        vel_world = agent.state.vel  # [B, 2] - world frame
        rot = agent.state.rot  # [B, 1] - agent朝向
        cos_theta = torch.cos(rot)  # [B, 1]
        sin_theta = torch.sin(rot)  # [B, 1]

        # 转换到body frame：前进速度 = vel沿朝向的分量，横向速度 = vel垂直朝向的分量
        # vx_body (forward) = cos(θ) * vx_world + sin(θ) * vy_world
        # vy_body (lateral) = -sin(θ) * vx_world + cos(θ) * vy_world
        vx_body = cos_theta.squeeze(-1) * vel_world[:, 0] + sin_theta.squeeze(-1) * vel_world[:, 1]  # [B]
        vy_body = -sin_theta.squeeze(-1) * vel_world[:, 0] + cos_theta.squeeze(-1) * vel_world[:, 1]  # [B]
        vel_body = torch.stack([vx_body, vy_body], dim=-1)  # [B, 2]

        # 2. 归一化（保持网络输入在 ~O(1) 数量级）
        if self.normalize_obs:
            # vel_body：按最大速度v0归一化
            vel_body_n = vel_body / max(self.v0, 1e-8)  # ~[-1, 1]
        else:
            vel_body_n = vel_body

        if self.gnn_obs_self_only:
            obs = vel_body_n
        else:
            # 3. 提取邻居数量
            neighbor_count = self._neighbor_count[:, idx, :]  # [B, 1]

            # 4. 提取Top-K邻居信息（已经是body frame）
            top_k_neighbors = self._top_k_neighbors[:, idx, :, :]  # [B, K, 4]
            local_geom_features = self._local_geom_features[:, idx, :]  # [B, 6]

            # 展平邻居信息：[B, K, 4] -> [B, K*4]
            K = self.obs_top_k_neighbors
            top_k_neighbors_flat = top_k_neighbors.reshape(top_k_neighbors.shape[0], K * 4)  # [B, K*4]
            local_geom_features_n = local_geom_features

            if self.normalize_obs:
                # neighbor_count：按最大可能邻居数归一化
                neighbor_count_n = neighbor_count / max(float(self.n_agents - 1), 1.0)  # [0, 1]

                # Top-K邻居：展开成 [B, K, 4] 方便处理
                top_k_neighbors_expanded = top_k_neighbors_flat.reshape(-1, K, 4)

                # dx_body, dy_body: 按通信范围comm_r归一化
                top_k_neighbors_expanded[:, :, 0] = top_k_neighbors_expanded[:, :, 0] / max(self.comm_r, 1e-8)  # dx_body
                top_k_neighbors_expanded[:, :, 1] = top_k_neighbors_expanded[:, :, 1] / max(self.comm_r, 1e-8)  # dy_body

                # dist: 按通信范围comm_r归一化
                top_k_neighbors_expanded[:, :, 2] = top_k_neighbors_expanded[:, :, 2] / max(self.comm_r, 1e-8)

                # valid: 已经是[0,1]，无需归一化（第4个维度保持不变）

                # 重新展平
                top_k_neighbors_flat_n = top_k_neighbors_expanded.reshape(-1, K * 4)  # [B, K*4]
            else:
                neighbor_count_n = neighbor_count
                top_k_neighbors_flat_n = top_k_neighbors_flat

            # 5. 构建最终观测向量
            obs = torch.cat([
                vel_body_n,                 # [2] 自身速度（body frame）
                neighbor_count_n,           # [1] 邻居数量
                top_k_neighbors_flat_n,     # [K*4] Top-K邻居信息（展平，body frame）
                local_geom_features_n,      # [6] 局部角色/边界形状特征
            ], dim=-1)  # Total dim / 总维度：9 + K*4

        # 6. Optional: include goal-relative observation.
        # When target assignment is active, points toward agent's assigned target;
        # otherwise falls back to formation_center.
        if self.obs_include_goal_rel:
            pos_world = agent.state.pos  # [B, 2]
            if self.target_progress_w > 0.0 and self._target_per_agent is not None:
                # Per-agent target from assignment
                target_world = self._target_per_agent[:, idx, :]  # [B, 2]
                goal_rel_world = target_world - pos_world  # [B, 2]
            else:
                # Legacy: relative to formation center
                goal_rel_world = self.formation_center.unsqueeze(0) - pos_world  # [B, 2]

            # Convert to body frame
            rot = agent.state.rot  # [B, 1]
            cos_theta = torch.cos(rot).squeeze(-1)  # [B]
            sin_theta = torch.sin(rot).squeeze(-1)  # [B]
            goal_rel_x_body = cos_theta * goal_rel_world[:, 0] + sin_theta * goal_rel_world[:, 1]
            goal_rel_y_body = -sin_theta * goal_rel_world[:, 0] + cos_theta * goal_rel_world[:, 1]
            goal_rel_body = torch.stack([goal_rel_x_body, goal_rel_y_body], dim=-1)  # [B, 2]

            # Distance and heading error
            goal_dist = goal_rel_world.norm(dim=-1, keepdim=True)  # [B, 1]
            goal_angle = torch.atan2(goal_rel_world[:, 1], goal_rel_world[:, 0]).unsqueeze(-1)  # [B, 1]
            heading_error = goal_angle - rot  # [B, 1]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

            # Normalize
            if self.normalize_obs:
                goal_rel_body_n = goal_rel_body / max(self.world_semidim, 1e-8)
                goal_dist_n = goal_dist / max(self.world_semidim, 1e-8)
                heading_error_n = heading_error / math.pi
            else:
                goal_rel_body_n = goal_rel_body
                goal_dist_n = goal_dist
                heading_error_n = heading_error

            goal_features = torch.cat([goal_rel_body_n, goal_dist_n, heading_error_n], dim=-1)  # [B, 4]
            obs = torch.cat([obs, goal_features], dim=-1)

        # 5. Optional checkpoint compatibility: fit observation dim to obs_pad_to_dim.
        if self.obs_pad_to_dim is not None:
            if obs.shape[-1] < self.obs_pad_to_dim:
                pad = torch.zeros(
                    (obs.shape[0], self.obs_pad_to_dim - obs.shape[-1]),
                    device=obs.device,
                    dtype=obs.dtype,
                )
                obs = torch.cat([obs, pad], dim=-1)
            elif obs.shape[-1] > self.obs_pad_to_dim:
                obs = obs[:, : self.obs_pad_to_dim]

        return obs

    def info(self, agent: Agent):
        if self._info_cache is None:
            pos, collision_pen, action_cost, speed_mean = self._compute_metrics()
            (
                formation_loss,
                spacing_loss,
                triangularity,
                sinkhorn_entropy,
                _,
                _,
                _,
                lattice_loss,
                boundary_peak_count,
                boundary_corner_score,
                boundary_straightness,
                triangle_shape_score,
            ) = self._formation_terms(pos)
            self._info_cache = self._build_formation_info(
                formation_loss=formation_loss,
                collision_pen=collision_pen,
                action_cost=action_cost,
                speed_mean=speed_mean,
                sinkhorn_entropy=sinkhorn_entropy,
                lattice_loss=lattice_loss,
                triangularity=triangularity,
                boundary_peak_count=boundary_peak_count,
                boundary_corner_score=boundary_corner_score,
                boundary_straightness=boundary_straightness,
                triangle_shape_score=triangle_shape_score,
                local_spacing_progress_mean=torch.zeros_like(formation_loss),
                local_lattice_progress_mean=torch.zeros_like(formation_loss),
                global_shape_progress_mean=torch.zeros_like(formation_loss),
            )
            self._info_cache["spacing_loss"] = spacing_loss
        return self._info_cache

    def done(self):
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)


    def extra_render(self, env_index: int = 0):
        return []


if __name__ == "__main__":
    render_interactively(
        __file__,
        continuous_actions=False,
        dict_spaces=False,
        n_agents=30,
    )
