import math
from typing import Optional
import torch
from vmas import render_interactively
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.core import Agent, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils
from utils.triangle_reward import (
    build_triangle_template,
    row_signature_cost_matrix,
    scale_invariant_distance_loss,
    sinkhorn,
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
        self.formation_sinkhorn_tau = float(kwargs.pop("formation_sinkhorn_tau", 0.1))
        self.formation_sinkhorn_iters = int(kwargs.pop("formation_sinkhorn_iters", 20))
        self.formation_eps = float(kwargs.pop("formation_eps", 1e-8))
        self.formation_template_seed = int(kwargs.pop("formation_template_seed", 0))
        self.safe_collision_w = float(kwargs.pop("safe_collision_w", 0.5))
        self.safe_action_w = float(kwargs.pop("safe_action_w", 0.02))

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

        # Build deterministic formation template once (shape only; not tied to world coordinates).
        tmpl = build_triangle_template(
            n_agents=self.n_agents,
            seed=self.formation_template_seed,
            side_length=1.0,
        ).to(device=device, dtype=torch.float32)
        self._formation_template_dist2 = squared_distance_matrix_batched(tmpl.unsqueeze(0))[0]  # [N,N]

        self._rew_per_agent = None
        self._shared_rew = None
        self._info_cache = None

        return world

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

    def _formation_terms(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return structural loss and Sinkhorn entropy, both shape [B]."""
        dist2 = squared_distance_matrix_batched(pos)  # [B,N,N]
        cost = row_signature_cost_matrix(dist2, self._formation_template_dist2)  # [B,N,N]
        soft_perm = sinkhorn(
            cost,
            tau=self.formation_sinkhorn_tau,
            iters=self.formation_sinkhorn_iters,
            eps=self.formation_eps,
        )

        tmpl = self._formation_template_dist2.unsqueeze(0).expand(pos.shape[0], -1, -1)
        dist2_soft = torch.matmul(torch.matmul(soft_perm, tmpl), soft_perm.transpose(1, 2))
        loss_struct = scale_invariant_distance_loss(dist2, dist2_soft, eps=self.formation_eps)  # [B]
        entropy = -(soft_perm * (soft_perm + self.formation_eps).log()).sum(dim=-1).mean(dim=-1)  # [B]
        return loss_struct, entropy

    def _build_formation_info(
        self,
        formation_loss: torch.Tensor,
        collision_pen: torch.Tensor,
        action_cost: torch.Tensor,
        speed_mean: torch.Tensor,
        sinkhorn_entropy: torch.Tensor,
    ) -> dict:
        collision_mean = collision_pen.mean(dim=-1)
        action_mean = action_cost.mean(dim=-1)
        return {
            "formation_loss": formation_loss,
            "formation_score": torch.exp(-formation_loss),
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
            formation_loss, sinkhorn_entropy = self._formation_terms(pos)
            per_agent = (
                -self.formation_w * formation_loss.unsqueeze(-1)
                - self.safe_collision_w * collision_pen
                - self.safe_action_w * action_cost
            )
            self._info_cache = self._build_formation_info(
                formation_loss=formation_loss,
                collision_pen=collision_pen,
                action_cost=action_cost,
                speed_mean=speed_mean,
                sinkhorn_entropy=sinkhorn_entropy,
            )

            if self.share_reward:
                self._shared_rew = per_agent.mean(dim=-1)  # [B]
            else:
                self._rew_per_agent = per_agent

        if self.share_reward:
            return self._shared_rew

        idx = self.world.agents.index(agent)
        return self._rew_per_agent[:, idx]

    def _compute_neighbor_features_cache(self):
        """Compute Top-K local neighbor features for all agents.

        Cached tensor shapes:
        - self._neighbor_count: [B, N, 1]
        - self._top_k_neighbors: [B, N, K, 4] with
          [dx_body, dy_body, dist, valid]
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

        # ⭐ CRITICAL FIX: Replace invalid distances with comm_r (so they normalize to 1.0, not 1e10!).
        topk_dists = torch.where(topk_valid > 0.5, topk_dists, torch.full_like(topk_dists, self.comm_r))
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
            padding[:, :, :, 2] = self.comm_r
            self._top_k_neighbors = torch.cat([topk_features, padding], dim=2)  # [B, N, K, 4]
        else:
            self._top_k_neighbors = topk_features  # [B, N, K, 4]


    def observation(self, agent: Agent):
        """Per-agent local observation for formation task.

        Feature layout:
        - vel_body: [2]
        - neighbor_count: [1]
        - top_k_neighbors: [K*4], each neighbor is [dx_body, dy_body, dist, valid]

        Total dim = 3 + K*4.
        """

        # 第一次调用agent_0的observation时重新计算缓存
        is_first = agent == self.world.agents[0]
        if is_first or not hasattr(self, '_neighbor_count'):
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

        # 3. 提取Top-K邻居信息（已经是body frame）
        top_k_neighbors = self._top_k_neighbors[:, idx, :, :]  # [B, K, 4]

        # 展平邻居信息：[B, K, 4] -> [B, K*4]
        K = self.obs_top_k_neighbors
        top_k_neighbors_flat = top_k_neighbors.reshape(top_k_neighbors.shape[0], K * 4)  # [B, K*4]

        # 5. 归一化（保持网络输入在 ~O(1) 数量级）
        if self.normalize_obs:
            # vel_body：按最大速度v0归一化
            vel_body_n = vel_body / max(self.v0, 1e-8)  # ~[-1, 1]

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
            vel_body_n = vel_body
            neighbor_count_n = neighbor_count
            top_k_neighbors_flat_n = top_k_neighbors_flat

        # 4. 构建最终观测向量
        obs = torch.cat([
            vel_body_n,                 # [2] 自身速度（body frame）
            neighbor_count_n,           # [1] 邻居数量
            top_k_neighbors_flat_n,     # [K*4] Top-K邻居信息（展平，body frame）
        ], dim=-1)  # Total dim / 总维度：3 + K*4

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
            formation_loss, sinkhorn_entropy = self._formation_terms(pos)
            self._info_cache = self._build_formation_info(
                formation_loss=formation_loss,
                collision_pen=collision_pen,
                action_cost=action_cost,
                speed_mean=speed_mean,
                sinkhorn_entropy=sinkhorn_entropy,
            )
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
