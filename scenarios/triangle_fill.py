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
        self.w_out = float(kwargs.pop("w_out", 10.0)) # Stronger pull towards the triangle (helps avoid plateau).
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
        # Backward-compat: optionally pad observation vector to a fixed dimension (useful when loading older checkpoints).
        # If None (default), no padding is applied and obs_dim is determined by the current observation() definition.
        self.obs_pad_to_dim = kwargs.pop("obs_pad_to_dim", None)
        if self.obs_pad_to_dim is not None:
            self.obs_pad_to_dim = int(self.obs_pad_to_dim)

        triangle_mm = kwargs.pop(
            "triangle_mm",
            TriangleSpec(a=(-169.0, -97.5), b=(169.0, -97.5), c=(0.0, 195.0)),
        )        

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
        # Convert depth scale to simulation units.
        self.depth_scale = max(self.depth_scale_mm * self.mm_to_unit, 1e-8)

        self.tri = TriangleSpec(
            a=(triangle_mm.a[0] * self.mm_to_unit, triangle_mm.a[1] * self.mm_to_unit),
            b=(triangle_mm.b[0] * self.mm_to_unit, triangle_mm.b[1] * self.mm_to_unit),
            c=(triangle_mm.c[0] * self.mm_to_unit, triangle_mm.c[1] * self.mm_to_unit),
        )
        # Goal direction (triangle centroid). We include (centroid - pos) in observations to provide
        # a strong navigation signal when robots start outside (prevents "wander near the gate" behavior).
        self.tri_centroid = (
            (self.tri.a[0] + self.tri.b[0] + self.tri.c[0]) / 3.0,
            (self.tri.a[1] + self.tri.b[1] + self.tri.c[1]) / 3.0,
        )
        self._tri_centroid = torch.tensor(self.tri_centroid, device=device, dtype=torch.float32)  # [2]
        self._tri_vertices = torch.tensor(
            [self.tri.a, self.tri.b, self.tri.c],
            device=device,
            dtype=torch.float32,
        )  # [3,2]

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

        cover_error = torch.zeros(self.world.batch_dim, device=self.world.device)
        points = self._cover_points.unsqueeze(0).expand(self.world.batch_dim, -1, -1)  # [B,M,2]
        d_pm = torch.cdist(points, pos)  # [B,M,N]
        cover_error = d_pm.min(dim=-1).values.mean(dim=-1)  # [B]

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
        """Compute neighbor-related features for *all* agents (vectorized).

        Why: each agent needs some awareness of others to avoid the "blind agents" plateau.
        We cache the result because VMAS will call observation(agent) once per agent per step.

        Cached tensors shapes:
          - self._nearest_rel: [B, N, 2]   (nearest neighbor relative position)
          - self._nearest_dist: [B, N, 1]  (nearest neighbor distance)
          - self._neighbor_count: [B, N, 1] (how many neighbors within comm_r)
        """
        pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)  # [B, N, 2]
        dist = torch.cdist(pos, pos)  # [B, N, N]

        # Exclude self by setting diagonal to +inf.
        dist = dist + torch.diag_embed(torch.full((pos.shape[0], pos.shape[1]), float('inf'), device=pos.device))

        nearest_dist, nearest_idx = dist.min(dim=-1)  # [B, N], [B, N]
        batch_idx = torch.arange(pos.shape[0], device=pos.device).unsqueeze(-1).expand_as(nearest_idx)
        nearest_pos = pos[batch_idx, nearest_idx]  # [B, N, 2]

        self._nearest_rel = nearest_pos - pos  # [B, N, 2]
        self._nearest_dist = nearest_dist.unsqueeze(-1)  # [B, N, 1]

        # Neighbor count inside comm_r (excluding self, already inf on diagonal).
        self._neighbor_count = (dist < self.comm_r).float().sum(dim=-1, keepdim=True)  # [B, N, 1]


    def observation(self, agent: Agent):
        """Per-agent observation.

        Base features (goal-directed):
          - self pos/vel/rot
          - signed distance to 3 triangle edges

        Coordination features (social):
          - nearest neighbor relative position (dx, dy)
          - nearest neighbor distance
          - neighbor count within comm_r

        These extra features greatly reduce the plateau where agents cannot coordinate filling.
        """

        # VMAS calls observation() once per agent; recompute neighbor features when
        # observation() is first called for agent_0 (once per environment step).
        is_first = agent == self.world.agents[0]
        if is_first or not hasattr(self, '_nearest_rel'):
            self._compute_neighbor_features_cache()

        signed = self._signed_edge_dists(agent.state.pos.unsqueeze(1)).squeeze(1)  # [B, 3]

        idx = self.world.agents.index(agent)
        nearest_rel = self._nearest_rel[:, idx, :]  # [B, 2]
        nearest_dist = self._nearest_dist[:, idx, :]  # [B, 1]
        neighbor_count = self._neighbor_count[:, idx, :]  # [B, 1]
        # Vector pointing from robot to the triangle centroid (navigation cue).
        # NOTE: appended at the end of the observation vector to preserve backward-compatibility of the
        # earlier feature ordering (so old checkpoints can still be fine-tuned safely).
        goal_rel = self._tri_centroid.unsqueeze(0) - agent.state.pos  # [B, 2]

        # NOTE: We normalize to keep magnitudes roughly O(1) (helps PPO stability).
        # This is especially useful when mixing meters, meters/sec, radians, and counts in one vector.
        if self.normalize_obs:
            pos = agent.state.pos / max(self.world_semidim, 1e-8)  # ~[-1,1]
            vel = agent.state.vel / max(self.v0, 1e-8)  # ~[-1,1] when moving at v0
            rot = agent.state.rot / math.pi  # ~[-1,1]
            signed_n = signed / max(self.world_semidim, 1e-8)  # scale triangle edge distances by world size
            nearest_rel_n = nearest_rel / max(self.comm_r, 1e-8)  # scale by comm range
            nearest_dist_n = nearest_dist / max(self.comm_r, 1e-8)
            neighbor_count_n = neighbor_count / max(float(self.n_agents - 1), 1.0)  # in [0,1]
            goal_rel_n = goal_rel / max(self.world_semidim, 1e-8)
        else:
            pos, vel, rot = agent.state.pos, agent.state.vel, agent.state.rot
            signed_n, nearest_rel_n, nearest_dist_n, neighbor_count_n = signed, nearest_rel, nearest_dist, neighbor_count
            goal_rel_n = goal_rel

        # Keep the original feature order first, then append goal_rel at the end:
        #   [pos, vel, rot, signed, nearest_rel, nearest_dist, neighbor_count, goal_rel]
        obs = torch.cat([pos, vel, rot, signed_n, nearest_rel_n, nearest_dist_n, neighbor_count_n, goal_rel_n], dim=-1)

        # Optional padding for checkpoint compatibility: if a saved policy expects a larger obs_dim,
        # we append zeros so shapes match. This does NOT change the semantics of the first part of obs.
        if self.obs_pad_to_dim is not None and obs.shape[-1] < self.obs_pad_to_dim:
            pad = torch.zeros((obs.shape[0], self.obs_pad_to_dim - obs.shape[-1]), device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, pad], dim=-1)

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
