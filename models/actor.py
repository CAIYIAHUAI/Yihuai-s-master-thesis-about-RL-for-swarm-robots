from __future__ import annotations

import math
import torch
import torch.nn as nn


class ActorBase(nn.Module):
    recurrent: bool
    hidden_size: int
    is_graph_actor: bool

    def forward(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        rot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


class MLPActor(ActorBase):
    def __init__(self, obs_dim: int, hidden: int, n_actions: int = 4, recurrent: bool = False):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_size = hidden
        self.is_graph_actor = False
        if recurrent:
            self.fc_in = nn.Linear(obs_dim, hidden)
            self.gru_cell = nn.GRUCell(hidden, hidden)
            self.fc_out = nn.Linear(hidden, n_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, n_actions),
            )

    def forward(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        rot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        obs_was_batched = obs.ndim == 3
        if obs_was_batched:
            batch_size, n_agents, obs_dim = obs.shape
            obs_flat = obs.reshape(batch_size * n_agents, obs_dim)
        else:
            obs_flat = obs

        if not self.recurrent:
            logits = self.net(obs_flat)
            if obs_was_batched:
                logits = logits.view(batch_size, n_agents, -1)
            return logits, None

        x = torch.tanh(self.fc_in(obs_flat)).to(dtype=self.fc_in.weight.dtype)
        if hx is None:
            hx = torch.zeros((x.shape[0], self.hidden_size), device=x.device, dtype=x.dtype)
        else:
            hx = hx.to(dtype=x.dtype)
        hx_new = self.gru_cell(x, hx)
        logits = self.fc_out(hx_new)
        if obs_was_batched:
            logits = logits.view(batch_size, n_agents, -1)
        return logits, hx_new


class _MessagePassingLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_agents = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n_agents, -1)
        h_j = h.unsqueeze(1).expand(-1, n_agents, -1, -1)
        msg_in = torch.cat([h_i, h_j, edge_feat], dim=-1)
        msg = self.message_mlp(msg_in)
        msg = msg * edge_mask.unsqueeze(-1)
        deg = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = msg.sum(dim=2) / deg
        updated = self.update_mlp(torch.cat([h, agg], dim=-1))
        return torch.tanh(h + updated)


class _GatedMessagePassingLayer(nn.Module):
    def __init__(self, state_dim: int, edge_dim: int, inner_dim: int):
        super().__init__()
        in_dim = state_dim * 2 + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, state_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, 1),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(state_dim * 2 + 1, inner_dim),
            nn.Tanh(),
            nn.Linear(inner_dim, state_dim),
        )
        # Post-residual LayerNorm: keeps message-passing output statistics stable across
        # rapid weight updates, so downstream fc_out sees a non-drifting input distribution.
        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_agents = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n_agents, -1)
        h_j = h.unsqueeze(1).expand(-1, n_agents, -1, -1)
        pair = torch.cat([h_i, h_j, edge_feat], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(pair)) * edge_mask.unsqueeze(-1)
        msg = self.message_mlp(pair)
        denom = gate.sum(dim=2).clamp_min(1.0)
        agg = (gate * msg).sum(dim=2) / denom
        degree = edge_mask.sum(dim=-1, keepdim=True)
        degree_norm = degree / max(n_agents - 1, 1)
        delta = torch.tanh(self.update_mlp(torch.cat([h, agg, degree_norm], dim=-1)))
        return self.norm(h + delta)


class GNNActor(ActorBase):
    def __init__(
        self,
        obs_dim: int,
        hidden: int,
        obs_top_k_neighbors: int,
        graph_hidden: int = 64,
        gnn_layers: int = 2,
        graph_radius: float = 0.0,
        graph_top_k: int = 0,
        gnn_residual_init: float = 0.1,
        n_actions: int = 4,
        recurrent: bool = False,
    ):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_size = hidden
        self.is_graph_actor = True
        self.obs_top_k_neighbors = int(obs_top_k_neighbors)
        self.obs_dim = int(obs_dim)
        self.graph_radius = float(graph_radius)
        self.graph_top_k = int(graph_top_k)
        self.fc_in = nn.Linear(self.obs_dim, hidden)
        self.layers = nn.ModuleList(
            [
                _GatedMessagePassingLayer(
                    state_dim=hidden,
                    edge_dim=3,
                    inner_dim=graph_hidden,
                )
                for _ in range(int(gnn_layers))
            ]
        )
        # Per-feature residual gate: shape [hidden] instead of scalar.
        # Lets PPO selectively keep useful GNN feature dimensions while damping noisy ones,
        # rather than collapsing the GNN contribution as a single global scalar.
        init = float(gnn_residual_init)
        init = min(max(init, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init / (1.0 - init))
        self.residual_alpha_logit = nn.Parameter(
            torch.full((hidden,), init_logit, dtype=torch.float32)
        )
        if recurrent:
            self.gru_cell = nn.GRUCell(hidden, hidden)
            self.fc_out = nn.Linear(hidden, n_actions)
        else:
            self.fc_out = nn.Linear(hidden, n_actions)

    def _build_graph(
        self,
        pos: torch.Tensor,
        rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos is None or rot is None:
            raise ValueError("GNNActor requires pos and rot tensors to build the interaction graph.")

        rel_world = pos.unsqueeze(1) - pos.unsqueeze(2)  # [B, N, N, 2], j - i
        dx_world = rel_world[..., 0]
        dy_world = rel_world[..., 1]
        dist = torch.sqrt(dx_world.square() + dy_world.square() + 1e-8)

        rot_i = rot.squeeze(-1).unsqueeze(-1)
        cos_i = torch.cos(rot_i)
        sin_i = torch.sin(rot_i)
        dx_body = cos_i * dx_world + sin_i * dy_world
        dy_body = -sin_i * dx_world + cos_i * dy_world

        radius = max(self.graph_radius, 1e-8)
        edge_feat = torch.stack([dx_body / radius, dy_body / radius, dist / radius], dim=-1)

        batch_size, n_agents, _ = pos.shape
        self_mask = torch.eye(n_agents, device=pos.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = ~self_mask
        if self.graph_radius > 0.0:
            edge_mask = edge_mask & (dist < self.graph_radius)

        if self.graph_top_k > 0:
            # Keep only each agent's nearest valid neighbors. This preserves a true graph
            # while preventing the dense early-training graph from homogenizing all node states.
            inf = torch.full_like(dist, float("inf"))
            dist_valid = torch.where(edge_mask, dist, inf)
            k = min(self.graph_top_k, max(n_agents - 1, 1))
            topk_idx = dist_valid.topk(k=k, dim=-1, largest=False).indices  # [B, N, K]
            topk_mask = torch.zeros((batch_size, n_agents, n_agents), device=pos.device, dtype=torch.bool)
            topk_mask.scatter_(dim=-1, index=topk_idx, value=True)
            edge_mask = edge_mask & topk_mask
        return edge_feat, edge_mask.to(dtype=pos.dtype)

    def forward(
        self,
        obs: torch.Tensor,
        hx: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        rot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if obs.ndim != 3:
            raise ValueError(f"GNNActor expects obs with shape [B, N, D], got {tuple(obs.shape)}")

        batch_size, n_agents, _ = obs.shape
        z = torch.tanh(self.fc_in(obs)).to(dtype=self.fc_in.weight.dtype)
        h = z
        edge_feat, edge_mask = self._build_graph(pos=pos, rot=rot)

        for layer in self.layers:
            h = layer(h, edge_feat=edge_feat, edge_mask=edge_mask)

        alpha = torch.sigmoid(self.residual_alpha_logit).to(dtype=z.dtype)
        x = z + alpha * (h - z)

        if not self.recurrent:
            return self.fc_out(x), None

        h_flat = x.reshape(batch_size * n_agents, -1)
        if hx is None:
            hx = torch.zeros((h_flat.shape[0], self.hidden_size), device=h_flat.device, dtype=h_flat.dtype)
        else:
            hx = hx.to(dtype=h_flat.dtype)
        hx_new = self.gru_cell(h_flat, hx)
        logits = self.fc_out(hx_new).view(batch_size, n_agents, -1)
        return logits, hx_new


def build_actor(
    obs_dim: int,
    hidden: int,
    recurrent: bool,
    n_actions: int = 4,
    gnn: bool = False,
    obs_top_k_neighbors: int = 8,
    gnn_hidden: int = 64,
    gnn_layers: int = 2,
    gnn_radius: float = 0.0,
    gnn_top_k: int = 0,
    gnn_residual_init: float = 0.1,
) -> ActorBase:
    if not gnn:
        return MLPActor(obs_dim=obs_dim, hidden=hidden, n_actions=n_actions, recurrent=recurrent)
    return GNNActor(
        obs_dim=obs_dim,
        hidden=hidden,
        obs_top_k_neighbors=obs_top_k_neighbors,
        graph_hidden=gnn_hidden,
        gnn_layers=gnn_layers,
        graph_radius=gnn_radius,
        graph_top_k=gnn_top_k,
        gnn_residual_init=gnn_residual_init,
        n_actions=n_actions,
        recurrent=recurrent,
    )
