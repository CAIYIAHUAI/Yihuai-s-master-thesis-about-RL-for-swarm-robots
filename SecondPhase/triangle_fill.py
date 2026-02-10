import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from vmas import render_interactively
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


@dataclass(frozen=True)
class TriangleSpec:
    a: Tuple[float, float]
    b: Tuple[float, float]
    c: Tuple[float, float]

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
        self.n_agents = int(kwargs.pop("n_agents", 30))
        self.share_reward = bool(kwargs.pop("share_reward", True)) # I dont want agents to be selfish
        self.cover_points_n = int(kwargs.pop("cover_points_n", 60))

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

        # reward weights
        self.w_in = float(kwargs.pop("w_in", 1.0)) # Positive reward for staying inside the target triangle.
        self.w_out = float(kwargs.pop("w_out", 1.0)) # FIXED: Reduced from 10.0 to 1.0 to avoid overly harsh penalties.
        self.w_collision = float(kwargs.pop("w_collision", 2.0)) # High penalty for overlaps to prevent clumping.
        self.w_action = float(kwargs.pop("w_action", 0.0))  # Start with 0 to avoid the common 'don't move' local optimum; add later if needed.
        self.w_cover = float(kwargs.pop("w_cover", 0.0)) # Bonus for covering area, currently disabled for curriculum.
        # Extra shaping: once inside, reward being deeper inside (continuous gradient, helps avoid "park at the edge").
        self.w_depth = float(kwargs.pop("w_depth", 1.0))
        self.depth_scale_mm = float(kwargs.pop("depth_scale_mm", 50.0))  # ~50mm is a reasonable "depth" scale

        # Spawn (pile) config
        #
        # IMPORTANT (training stability):
        # The target triangle is centered around the origin (0,0). If we also spawn the pile around (0,0) with a
        # large box (e.g. 400x400mm), then a significant fraction of robots will start *inside* the triangle "for free".
        # This creates a strong lazy local optimum: keep those inside robots still -> inside_frac stays near that
        # initial fraction and training plateaus.
        #
        # To avoid this, we spawn the pile *below* the triangle by default so inside_frac starts near 0.
        # You can override these via make_env(..., pile_center_mm=(0,0), pile_halfwidth_mm=200) if you want.
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

        # Radar observation (range-limited; local sensing).
        # 雷达观测（有限量程；局部传感）。
        self.radar_num_beams = int(kwargs.pop("radar_num_beams", 8))
        radar_max_range_mm = kwargs.pop("radar_max_range_mm", None)
        if radar_max_range_mm is None:
            radar_max_range_mm = self.comm_r_mm
        self.radar_max_range_mm = float(radar_max_range_mm)

        triangle_mm = kwargs.pop(
            "triangle_mm",
            TriangleSpec(a=(-169.0, -97.5), b=(169.0, -97.5), c=(0.0, 195.0)),
        )        

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Convert map boundary size from mm to simulation units.
        self.world_semidim = self.world_semidim_mm * self.mm_to_unit
        # Convert neighbor communication range from mm to simulation units.
        self.comm_r = self.comm_r_mm * self.mm_to_unit
        # Convert radar range from mm to simulation units.
        # 将雷达量程从毫米转换为仿真单位。
        self.radar_max_range = max(self.radar_max_range_mm * self.mm_to_unit, 1e-8)
        # Convert robot collision radius from mm to simulation units.
        self.agent_radius = self.agent_radius_mm * self.mm_to_unit
        # Convert forward speed from mm/s to simulation units/s.
        self.v0 = self.v0_mm_s * self.mm_to_unit
        # Angular speed (rad/s) is unit-agnostic, so keep as is.
        self.w0 = self.w0_rad_s
        # Convert depth scale to simulation units.
        self.depth_scale = max(self.depth_scale_mm * self.mm_to_unit, 1e-8)

        self.tri = TriangleSpec(
            a=(triangle_mm.a[0] * self.mm_to_unit, triangle_mm.a[1] * self.mm_to_unit),
            b=(triangle_mm.b[0] * self.mm_to_unit, triangle_mm.b[1] * self.mm_to_unit),
            c=(triangle_mm.c[0] * self.mm_to_unit, triangle_mm.c[1] * self.mm_to_unit),
        )
        self._tri_vertices = torch.tensor(
            [self.tri.a, self.tri.b, self.tri.c],
            device=device,
            dtype=torch.float32,
        )  # [3,2]

        # Precompute radar beam unit directions in the body frame.
        # 预计算 body frame 下雷达射线的单位方向向量。
        k = int(self.radar_num_beams)
        if k <= 0:
            self._radar_dir_body = torch.zeros((0, 2), device=device, dtype=torch.float32)
        else:
            angles = (torch.arange(k, device=device, dtype=torch.float32) / float(k)) * (2.0 * math.pi)
            self._radar_dir_body = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [K, 2]

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

        self._build_triangle_geometry(device=device)
        self._init_cover_points(device=device)

        self._rew_per_agent = None
        self._shared_rew = None
        self._info_cache = None
        self._radar_dist = None
        self._radar_hit = None

        return world

    def _build_triangle_geometry(self, device: torch.device):
        v = self._tri_vertices # Get the triangle corner coordinates.
        edges = torch.stack([v[1] - v[0], v[2] - v[1], v[0] - v[2]], dim=0) # Compute vectors representing the three walls (A->B, B->C, C->A).
        normals_in = torch.stack([-edges[:, 1], edges[:, 0]], dim=-1)  # Compute vectors representing the three walls (A->B, B->C, C->A).
        normals_in = normals_in / torch.linalg.vector_norm(normals_in, dim=-1, keepdim=True) # Normalize arrows to have a length of 1.
        self._tri_edge_start = v # Store the starting point of each wall.
        self._tri_normals_in = normals_in # Store the inward-pointing direction arrows.

    def _init_cover_points(self, device: torch.device):
        m = self.cover_points_n # Number of cover points to generate.
        u = torch.rand((m, 1), device=device) # Generate raw random values between 0 and 1.
        v = torch.rand((m, 1), device=device) # Generate raw random values between 0 and 1.
        r1 = torch.sqrt(u) # preventing corner clumping.
        w1 = 1.0 - r1 # Calculate the barycentric weight for the first vertex.
        w2 = r1 * (1.0 - v) # Calculate weights for the second and third vertices.
        w3 = r1 * v
        a = self._tri_vertices[0].unsqueeze(0)
        b = self._tri_vertices[1].unsqueeze(0)
        c = self._tri_vertices[2].unsqueeze(0)
        # Compute final point positions using weighted vertex sums.
        self._cover_points = w1 * a + w2 * b + w3 * c 

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

        self._rew_per_agent = None
        self._shared_rew = None
        self._info_cache = None

    def _signed_edge_dists(self, pos_bn2: torch.Tensor) -> torch.Tensor:
        # Reshape positions to broadcast against 3 edges later.
        p = pos_bn2.unsqueeze(2) # [Batch, N, 2] -> [B,N,1,2]
        # Get edge start points and reshape for broadcasting. | 准备边的起点：取出之前存好的 3 个顶点坐标。
        e0 = self._tri_edge_start.view(1,1,3,2)
        # Get inward normal vectors and reshape for broadcasting.| 准备法线向量：取出之前存好的 3 个法线向量。
        n = self._tri_normals_in.view(1, 1, 3, 2)
        # Compute signed distances: dot product of (pos-edge_start) and normal. | 计算有向距离：位置-边起点 与 法线 的点积。
        return ((p - e0) * n).sum(dim=-1)  # [B,N,3]

    @staticmethod
    def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """2D cross product (scalar) for the last dimension.
        二维向量叉乘（标量），作用于最后一个维度。
        """
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def _compute_radar_cache(self, pos: torch.Tensor, rot: torch.Tensor) -> None:
        """Compute radar distances/hits for all agents and cache them.
        计算所有智能体的雷达距离/命中并缓存。

        The radar is range-limited and returns only local measurements:
        distances along fixed body-frame rays to the triangle edges (line segments).
        雷达是有限量程的，只返回局部测量：沿 body frame 固定射线到三角形边界（线段）的距离。

        Cached:
          - self._radar_dist: [B, N, K] (distance in simulation units, clipped to max range)
          - self._radar_hit:  [B, N, K] (1 if hit within range else 0)
        缓存：
          - self._radar_dist: [B, N, K]（仿真单位距离，超过量程截断）
          - self._radar_hit:  [B, N, K]（量程内命中为1，否则为0）
        """
        b, n, _ = pos.shape
        k = int(self.radar_num_beams)
        if k <= 0:
            self._radar_dist = torch.zeros((b, n, 0), device=pos.device, dtype=pos.dtype)
            self._radar_hit = torch.zeros((b, n, 0), device=pos.device, dtype=pos.dtype)
            return

        # Build world-frame ray directions from body-frame unit vectors.
        # 将 body frame 单位射线方向转换到 world frame。
        dir_body = self._radar_dir_body.to(device=pos.device, dtype=pos.dtype)  # [K,2]
        dx_b = dir_body[:, 0].view(1, 1, k)  # [1,1,K]
        dy_b = dir_body[:, 1].view(1, 1, k)  # [1,1,K]
        cos_t = torch.cos(rot).squeeze(-1).unsqueeze(-1)  # [B,N,1]
        sin_t = torch.sin(rot).squeeze(-1).unsqueeze(-1)  # [B,N,1]

        # world = R(theta) * body
        # world = R(theta) * body（将局部向量旋转到世界坐标）
        dx_w = cos_t * dx_b - sin_t * dy_b  # [B,N,K]
        dy_w = sin_t * dx_b + cos_t * dy_b  # [B,N,K]
        d = torch.stack([dx_w, dy_w], dim=-1)  # [B,N,K,2]

        # Triangle segments: s -> e (3 edges).
        # 三角形线段：起点 s 到终点 e（三条边）。
        v = self._tri_vertices.to(device=pos.device, dtype=pos.dtype)  # [3,2]
        s = v
        e = torch.roll(v, shifts=-1, dims=0)
        r = (e - s).view(1, 1, 1, 3, 2)  # [1,1,1,3,2]
        s = s.view(1, 1, 1, 3, 2)  # [1,1,1,3,2]

        # Broadcast to [B,N,K,3,2].
        # 广播到 [B,N,K,3,2]。
        p = pos.view(b, n, 1, 1, 2)
        d = d.view(b, n, k, 1, 2)
        q = s - p  # [B,N,K,3,2]

        eps = 1e-8
        denom = self._cross2(d, r)  # [B,N,K,3]
        denom_ok = denom.abs() > eps
        denom_safe = torch.where(denom_ok, denom, torch.ones_like(denom))

        t = self._cross2(q, r) / denom_safe  # [B,N,K,3]
        u = self._cross2(q, d) / denom_safe  # [B,N,K,3]

        hit = denom_ok & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)

        BIG = 1e9
        t_hit = torch.where(hit, t, torch.full_like(t, BIG))
        t_min = t_hit.min(dim=-1).values  # [B,N,K]

        # Apply range limit: beyond range is treated as no hit.
        # 施加量程限制：超出量程视为未命中。
        max_r = float(self.radar_max_range)
        hit_in_range = t_min <= max_r
        dist = torch.where(hit_in_range, t_min, torch.full_like(t_min, max_r))

        self._radar_dist = dist  # [B,N,K]
        self._radar_hit = hit_in_range.to(dtype=pos.dtype)  # [B,N,K]

    def _compute_metrics(self):
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1) # [B,N,2] collect all robots' (x,y)
        vel = torch.stack([a.state.vel for a in self.world.agents], dim=1) # [B,N,2] collect all robots' (vx,vy)
        signed = self._signed_edge_dists(pos) # [B,N,3] compute signed distances to each edge.
        min_signed = signed.min(dim=-1).values # [B,N] find closest edge for each robot.
        inside = min_signed >= 0.0 # [B,N] check if each robot is inside the triangle.
        outside_dist = (-min_signed).clamp(min=0.0) # [B,N] distance to closest edge (or 0 if inside).
        inside_depth = min_signed.clamp(min=0.0) # [B,N] how deep inside (0 if outside).

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

        # ⭐ OPTIMIZATION: Skip expensive cover_error computation when w_cover=0 (saves ~5-10% env step time)
        if self.w_cover > 0.0:
            points = self._cover_points.unsqueeze(0).expand(self.world.batch_dim, -1, -1)  # [B,M,2]
            d_pm = torch.cdist(points, pos)  # [B,M,N] - expensive for M=60, N=30
            cover_error = d_pm.min(dim=-1).values.mean(dim=-1)  # [B]
        else:
            cover_error = torch.zeros(self.world.batch_dim, device=self.world.device)

        inside_frac = inside.float().mean(dim=-1)  # [B]
        collisions_mean = collision_pen.mean(dim=-1)  # [B]
        outside_mean = outside_dist.mean(dim=-1)  # [B]
        speed_mean = vel.norm(dim=-1).mean(dim=-1)  # [B] mean speed across agents

        return inside, outside_dist, inside_depth, collision_pen, action_cost, cover_error, inside_frac, collisions_mean, outside_mean, speed_mean

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first or (self._rew_per_agent is None and self._shared_rew is None):
            # Compute all physical metrics (distances, collisions) for all agents at once.
            inside, outside_dist, inside_depth, collision_pen, action_cost, cover_error, inside_frac, collisions_mean, outside_mean, speed_mean = (
                self._compute_metrics()
            )

            # Calculate raw score:
            # - base reward for being inside (discrete jump helps "entering" behavior)
            # - continuous depth shaping once inside (prevents "hug the edge" and provides gradient)
            # - penalties for outside/collision/energy
            depth_bonus = torch.tanh(inside_depth / self.depth_scale)  # [B,N] in [0,1)
            per_agent = (
                self.w_in * inside.float()
                + self.w_depth * depth_bonus
                - self.w_out * outside_dist
                - self.w_collision * collision_pen
                # Brake tax ONLY after entering the target: we want agents to move/explore when outside,
                # but "settle" (STOP) once inside to keep inside_frac high and reduce jitter.
                - self.w_action * action_cost * inside.float()
            )  # [B,N]

            # If enabled, subtract penalty for not covering the full triangle area.
            if self.w_cover > 0.0:
                per_agent = per_agent - self.w_cover * cover_error.unsqueeze(-1)

            # Logic for Team Reward vs Individual Reward.
            if self.share_reward:
                # TEAM MODE: Everyone gets the average score of the group.
                self._shared_rew = per_agent.mean(dim=-1)  # [B]
            else:
                # SOLO MODE: Keep individual scores separate.
                self._rew_per_agent = per_agent

            # Cache statistics for logging/debugging (not used for training).
            self._info_cache = {
                "inside_frac": inside_frac,
                "cover_error": cover_error,
                "collisions_mean": collisions_mean,
                "outside_mean": outside_mean,
                "speed_mean": speed_mean,
            }

        # Return the pre-calculated score.
        if self.share_reward:
            return self._shared_rew
            
        # If individual, find this specific agent's index to get its score.
        idx = self.world.agents.index(agent)
        return self._rew_per_agent[:, idx]

    def _compute_neighbor_features_cache(self):
        """Compute Top-K neighbor features for all agents (vectorized).
        计算所有智能体的 Top-K 邻居特征（向量化实现）。

        Key design points:
        关键点：
        1) Neighbors are limited to comm_r (out-of-range treated as invalid slots).
           邻居限制在通信半径 comm_r 内（超出范围视为无效槽位）。
        2) Relative positions are expressed in the agent body frame (to avoid leaking world axes).
           相对位置使用 body frame（避免泄露世界坐标轴）。
        3) A `valid` mask distinguishes real neighbors from padding.
           用 `valid` 区分真实邻居与填充项。

        Cached tensor shapes:
        缓存张量维度：
        - self._inside: [B, N] (inside triangle or not) / 是否在三角形内
        - self._neighbor_count: [B, N, 1] (count within comm_r) / 通信范围内邻居数量
        - self._top_k_neighbors: [B, N, K, 5] where the last dim is
          [dx_body, dy_body, dist, is_inside, valid] / 最后一维为 [dx_body, dy_body, dist, is_inside, valid]
        - self._radar_dist: [B, N, R] (distance clipped to range) / 雷达距离（截断到量程）
        - self._radar_hit:  [B, N, R] (0/1 hit mask) / 雷达命中（0/1）
        """
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, N, 2]
        rot = torch.stack([a.state.rot for a in self.world.agents], dim=1)  # [B, N, 1]

        # 1. 计算所有agent的inside状态
        signed = self._signed_edge_dists(pos)  # [B, N, 3]
        min_signed = signed.min(dim=-1).values  # [B, N]
        self._inside = min_signed >= 0.0  # [B, N]

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

        # 11. 提取Top-K邻居的inside状态
        topk_inside = self._inside[batch_idx, topk_indices].float()  # [B, N, actual_k]

        # ⭐ CRITICAL FIX: Replace invalid distances with comm_r (so they normalize to 1.0, not 1e10!).
        # ⭐ 关键修复：将无效距离替换为 comm_r（归一化后为 1.0，而不是 1e10！）
        # Without this fix, invalid slots use dist=1e9; after dividing by comm_r≈0.085 this becomes ~1e10,
        # which saturates the network and kills learning.
        # 如果不修复，无效槽位 dist=1e9，除以 comm_r≈0.085 后约为 1e10，会让网络饱和、无法学习。
        topk_dists = torch.where(topk_valid > 0.5, topk_dists, torch.full_like(topk_dists, self.comm_r))
        # Also zero out relative position and inside status for invalid slots (cleaner semantics)
        topk_rel_pos_body = topk_rel_pos_body * topk_valid.unsqueeze(-1)
        topk_inside = topk_inside * topk_valid

        # ⭐ 12. 组合成完整的Top-K邻居特征：[dx_body, dy_body, dist, is_inside, valid]
        topk_features = torch.cat([
            topk_rel_pos_body,               # [B, N, actual_k, 2] - dx_body, dy_body
            topk_dists.unsqueeze(-1),        # [B, N, actual_k, 1] - dist
            topk_inside.unsqueeze(-1),       # [B, N, actual_k, 1] - is_inside
            topk_valid.unsqueeze(-1)         # [B, N, actual_k, 1] - valid
        ], dim=-1)  # [B, N, actual_k, 5]

        # ⭐ 13. 简化padding逻辑：只在K > N-1时需要padding到K个槽位
        if actual_k < K:
            # ⚠️ 重要：填充dist用原始尺度（comm_r），后续observation()会归一化成1.0
            # 如果填1.0会被误认为"1.0/comm_r = 很近"！
            num_padding = K - actual_k
            padding = torch.zeros(
                (pos.shape[0], pos.shape[1], num_padding, 5),
                device=pos.device,
                dtype=topk_features.dtype
            )
            padding[:, :, :, 2] = self.comm_r  # ⭐ dist填充为comm_r（原始尺度"很远"），归一化后变1.0
            # dx_body=0, dy_body=0, is_inside=0, valid=0 默认已是0
            self._top_k_neighbors = torch.cat([topk_features, padding], dim=2)  # [B, N, K, 5]
        else:
            # 不需要padding，直接使用
            self._top_k_neighbors = topk_features  # [B, N, K, 5]

        # Compute radar once per step (cache).
        # 每步只计算一次雷达（缓存）。
        self._compute_radar_cache(pos=pos, rot=rot)


    def observation(self, agent: Agent):
        """Per-agent observation (purely local; no global position/goal features).
        每个 agent 的观测（纯局部：不包含全局位置/目标特征）。

        Key design points:
        关键点：
        1) `vel` is expressed in the body frame (forward/lateral).
           `vel` 使用 body frame（前进/横向速度）。
        2) Neighbor relative positions are also in the body frame (to avoid leaking world axes).
           邻居相对位置同样使用 body frame（避免泄露世界坐标轴）。
        3) Keep `neighbor_count` as a direction-free statistic.
           保留 `neighbor_count` 作为无方向统计量。

        Feature layout:
        特征结构：
        - vel_body: [2]
        - neighbor_count: [1]
        - is_inside: [1]
        - top_k_neighbors: [K*5] flattened; each neighbor has
          [dx_body, dy_body, dist, is_inside, valid]
          Top-K 邻居展平后为 [K*5]；每个邻居为 [dx_body, dy_body, dist, is_inside, valid]

        Radar extension:
        雷达扩展：
        - radar_dist: [R]
        - radar_hit:  [R]
        Total extra dim = 2*R.
        额外维度 = 2*R。

        Total dim = 4 + K*5 + 2*R.
        总维度 = 4 + K*5 + 2*R。
        """

        # 第一次调用agent_0的observation时重新计算缓存
        is_first = agent == self.world.agents[0]
        if is_first or not hasattr(self, '_inside'):
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

        # 2. 提取邻居数量
        neighbor_count = self._neighbor_count[:, idx, :]  # [B, 1]

        # 3. 提取自身is_inside状态
        is_inside = self._inside[:, idx].unsqueeze(-1).float()  # [B, 1]

        # 4. 提取Top-K邻居信息（已经是body frame）
        top_k_neighbors = self._top_k_neighbors[:, idx, :, :]  # [B, K, 5]

        # 展平邻居信息：[B, K, 5] -> [B, K*5]
        K = self.obs_top_k_neighbors
        top_k_neighbors_flat = top_k_neighbors.reshape(top_k_neighbors.shape[0], K * 5)  # [B, K*5]

        # 5. 归一化（保持网络输入在 ~O(1) 数量级）
        if self.normalize_obs:
            # vel_body：按最大速度v0归一化
            vel_body_n = vel_body / max(self.v0, 1e-8)  # ~[-1, 1]

            # neighbor_count：按最大可能邻居数归一化
            neighbor_count_n = neighbor_count / max(float(self.n_agents - 1), 1.0)  # [0, 1]

            # is_inside：已经是[0,1]，无需归一化
            is_inside_n = is_inside

            # Top-K邻居：需要对每个维度分别归一化
            # 展开成 [B, K, 5] 方便处理
            top_k_neighbors_expanded = top_k_neighbors_flat.reshape(-1, K, 5)

            # dx_body, dy_body: 按通信范围comm_r归一化
            top_k_neighbors_expanded[:, :, 0] = top_k_neighbors_expanded[:, :, 0] / max(self.comm_r, 1e-8)  # dx_body
            top_k_neighbors_expanded[:, :, 1] = top_k_neighbors_expanded[:, :, 1] / max(self.comm_r, 1e-8)  # dy_body

            # dist: 按通信范围comm_r归一化
            top_k_neighbors_expanded[:, :, 2] = top_k_neighbors_expanded[:, :, 2] / max(self.comm_r, 1e-8)

            # is_inside: 已经是[0,1]，无需归一化（第4个维度保持不变）

            # valid: 已经是[0,1]，无需归一化（第5个维度保持不变）

            # 重新展平
            top_k_neighbors_flat_n = top_k_neighbors_expanded.reshape(-1, K * 5)  # [B, K*5]

            # Radar: normalize distance by max range (in [0,1]); keep hit as {0,1}.
            # 雷达：距离按最大量程归一化到 [0,1]；命中保持 {0,1}。
            radar_dist = self._radar_dist[:, idx, :]  # [B, R]
            radar_hit = self._radar_hit[:, idx, :]  # [B, R]
            radar_dist_n = radar_dist / max(self.radar_max_range, 1e-8)
        else:
            vel_body_n = vel_body
            neighbor_count_n = neighbor_count
            is_inside_n = is_inside
            top_k_neighbors_flat_n = top_k_neighbors_flat
            radar_dist_n = self._radar_dist[:, idx, :]
            radar_hit = self._radar_hit[:, idx, :]

        # 6. 构建最终观测向量
        obs = torch.cat([
            vel_body_n,                 # [2] 自身速度（body frame）
            neighbor_count_n,           # [1] 邻居数量
            is_inside_n,                # [1] 自己是否在内部
            top_k_neighbors_flat_n,     # [K*5] Top-K邻居信息（展平，body frame）
            radar_dist_n,               # [R] 雷达距离（归一化或原始尺度）
            radar_hit,                  # [R] 雷达命中（0/1）
        ], dim=-1)  # Total dim / 总维度：4 + K*5 + 2*R

        # 7. Optional checkpoint compatibility: fit observation dim to obs_pad_to_dim.
        # 7. 可选：为 checkpoint 兼容，将观测维度适配到 obs_pad_to_dim。
        if self.obs_pad_to_dim is not None:
            # If current obs is smaller, pad with zeros; if larger, truncate from the end.
            # 若当前观测更短则用 0 填充；若更长则从末尾截断。
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
            (
                _inside,
                _outside_dist,
                _inside_depth,
                _collision_pen,
                _action_cost,
                cover_error,
                inside_frac,
                collisions_mean,
                outside_mean,
                speed_mean,
            ) = self._compute_metrics()
            self._info_cache = {
                "inside_frac": inside_frac,
                "cover_error": cover_error,
                "collisions_mean": collisions_mean,
                "outside_mean": outside_mean,
                "speed_mean": speed_mean,
            }
        return self._info_cache

    def done(self):
        return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)


    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        # Draw the target triangle so it's visible in render_interactively().

        color = Color.BLACK.value
        a = self.tri.a
        b = self.tri.b
        c = self.tri.c
        lines = [
            rendering.Line(start=a, end=b, width=2),
            rendering.Line(start=b, end=c, width=2),
            rendering.Line(start=c, end=a, width=2),
        ]
        for line in lines:
            line.set_color(*color)
        return lines


if __name__ == "__main__":
    render_interactively(
        __file__,
        continuous_actions=False,
        dict_spaces=False,
        n_agents=30,
    )
