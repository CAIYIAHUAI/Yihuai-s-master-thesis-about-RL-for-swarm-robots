"""PyG GATv2Conv + GRU actor and a centralized critic for swarm_formation.

Actor input contract (per step)
    obs_nodes:  [B, N, node_dim]              local node features
    edge_index: [2, E]                        PyG batched (env_idx*N + agent_idx)
    edge_attr:  [E, edge_dim]                 edge features (dst-frame geometry)
    h_in:       [B, N, gru_hidden]            recurrent state

Actor output
    logits:        [B, N, 4]
    velocity_aux:  [B, N, 2]                  auxiliary BC head (body-frame v)
    h_out:         [B, N, gru_hidden]

The critic is centralized: it sees the global per-agent feature stack plus
swarm-level diagnostics (mean / std / extent) and outputs a per-agent value.

We deliberately do NOT pass any goal / slot / GW signals into the actor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from observations import NODE_FEATURE_DIM, EDGE_FEATURE_DIM, N_ACTIONS


class GATGRUActor(nn.Module):
    """PyG GATv2 ×2 → per-agent reshape → GRUCell → policy + velocity heads.

    v16: supports two action modes:
      - "discrete" (legacy): policy_head outputs N_ACTIONS=4 logits, Categorical sampling.
      - "continuous": velocity_head outputs (mu_linear, mu_angular) in [-1, 1] (via tanh),
        plus a learned log_std scalar parameter. action sampled from Normal(mu, exp(log_std)),
        clamped to [-1, 1] for env.step. PPO uses Normal.log_prob.
    """

    def __init__(self, node_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                 gat_hidden: int = 64, gat_heads: int = 4, gru_hidden: int = 128,
                 stop_action_bias: float = 0.0,
                 action_log_std_init: float = -1.9):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = gru_hidden                                  # alias kept for compat with old buffers
        # v18d: continuous action mode only (Normal sampler over linear/angular ∈ [-1,1]²).
        self.action_mode = "continuous"

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden), nn.GELU(),
        )
        self.gat1 = GATv2Conv(
            in_channels=gat_hidden, out_channels=gat_hidden, heads=gat_heads,
            edge_dim=edge_dim, concat=False, add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=gat_hidden, out_channels=gat_hidden, heads=gat_heads,
            edge_dim=edge_dim, concat=False, add_self_loops=True,
        )
        self.gru = nn.GRUCell(gat_hidden, gru_hidden)
        self.policy_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.GELU(),
            nn.Linear(gru_hidden, N_ACTIONS),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2), nn.GELU(),
            nn.Linear(gru_hidden // 2, 2),
        )
        # Small-init velocity_head last layer so initial mu ≈ 0 (avoids BC-derived
        # bias trapping continuous PPO at initialization).
        with torch.no_grad():
            nn.init.orthogonal_(self.velocity_head[-1].weight, gain=0.01)
            nn.init.zeros_(self.velocity_head[-1].bias)
        # Continuous-action log_std (learned, shared across batch/agents).
        # Default init = -1.9 → std ≈ 0.15 (tight enough for PPO to converge).
        self.action_log_std = nn.Parameter(torch.full((2,), float(action_log_std_init)))
        # Bias the STOP logit at init to encourage stopping when uncertain
        # (legacy hook from discrete mode; harmless under continuous since policy_head
        # logits are unused but still produced).
        with torch.no_grad():
            self.policy_head[-1].bias[0] += float(stop_action_bias)

    def forward(self, obs_nodes: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, h_in: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_nodes:  [B, N, node_dim]
        edge_index: [2, E]   indices in [0, B*N)
        edge_attr:  [E, edge_dim]
        h_in:       [B, N, gru_hidden]
        """
        B, N, _ = obs_nodes.shape
        x = obs_nodes.reshape(B * N, -1)                           # node_idx = b*N + n
        x = self.node_encoder(x)
        x = self.gat1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.gelu(x)
        h_flat = h_in.reshape(B * N, -1)
        h_next_flat = self.gru(x, h_flat)
        h_next = h_next_flat.reshape(B, N, -1)
        logits = self.policy_head(h_next_flat).reshape(B, N, N_ACTIONS)
        velocity_aux = self.velocity_head(h_next_flat).reshape(B, N, 2)
        return logits, velocity_aux, h_next


class MeanAggActor(nn.Module):
    """Edge-aware mean-aggregation baseline.

    This keeps the v18d observation and edge schema intact, including local
    geometry, neighbor z, and d_desired. The only architectural change versus
    GATGRUActor is replacing attention with an edge-MLP message followed by a
    degree-normalized mean over incoming messages.
    """

    def __init__(self, node_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                 gat_hidden: int = 64, gat_heads: int = 4, gru_hidden: int = 128,
                 stop_action_bias: float = 0.0,
                 action_log_std_init: float = -1.9):
        super().__init__()
        del gat_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = gru_hidden
        self.action_mode = "continuous"

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden), nn.GELU(),
        )
        self.msg1 = nn.Sequential(
            nn.Linear(gat_hidden + edge_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden),
        )
        self.self1 = nn.Linear(gat_hidden, gat_hidden)
        self.msg2 = nn.Sequential(
            nn.Linear(gat_hidden + edge_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden),
        )
        self.self2 = nn.Linear(gat_hidden, gat_hidden)
        self.gru = nn.GRUCell(gat_hidden, gru_hidden)
        self.policy_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.GELU(),
            nn.Linear(gru_hidden, N_ACTIONS),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2), nn.GELU(),
            nn.Linear(gru_hidden // 2, 2),
        )
        with torch.no_grad():
            nn.init.orthogonal_(self.velocity_head[-1].weight, gain=0.01)
            nn.init.zeros_(self.velocity_head[-1].bias)
        self.action_log_std = nn.Parameter(torch.full((2,), float(action_log_std_init)))
        with torch.no_grad():
            self.policy_head[-1].bias[0] += float(stop_action_bias)

    @staticmethod
    def _mean_messages(x: torch.Tensor, edge_index: torch.Tensor,
                       edge_attr: torch.Tensor, msg_net: nn.Module) -> torch.Tensor:
        src, dst = edge_index
        if src.numel() == 0:
            return torch.zeros_like(x)
        msg_in = torch.cat([x[src], edge_attr], dim=-1)
        msg = msg_net(msg_in)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        deg = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=x.dtype, device=x.device))
        return out / deg.clamp_min(1.0)

    def forward(self, obs_nodes: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, h_in: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = obs_nodes.shape
        x = self.node_encoder(obs_nodes.reshape(B * N, -1))
        x = F.gelu(self.self1(x) + self._mean_messages(x, edge_index, edge_attr, self.msg1))
        x = F.gelu(self.self2(x) + self._mean_messages(x, edge_index, edge_attr, self.msg2))
        h_flat = h_in.reshape(B * N, -1)
        h_next_flat = self.gru(x, h_flat)
        h_next = h_next_flat.reshape(B, N, -1)
        logits = self.policy_head(h_next_flat).reshape(B, N, N_ACTIONS)
        velocity_aux = self.velocity_head(h_next_flat).reshape(B, N, 2)
        return logits, velocity_aux, h_next


class CentralCritic(nn.Module):
    """Centralized critic — DeepSets over per-agent features (borrowed pattern).

    feature_dim is the dimensionality of the per-agent global feature vector
    constructed in train_mappo.build_critic_features. Output is per-agent value
    (the centralized critic broadcasts a shared value across the team after
    aggregation, matching the v2 DeepSetsCritic interface).
    """

    def __init__(self, feature_dim: int, hidden: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden * 3 + 1, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor,
                log_extent: torch.Tensor | None = None) -> torch.Tensor:
        """features: [B, N, feature_dim] -> values: [B, N]."""
        z = self.phi(features)
        mean = z.mean(dim=1, keepdim=True).expand_as(z)
        maxv = z.max(dim=1, keepdim=True).values.expand_as(z)
        if log_extent is None:
            center = features[..., :2].mean(dim=1)
            rms = (features[..., :2] - center.unsqueeze(1)).norm(dim=-1).mean(dim=1).clamp_min(1e-8)
            log_extent = torch.log(rms)
        le = log_extent.view(-1, 1, 1).expand(features.shape[0], features.shape[1], 1)
        return self.rho(torch.cat([z, mean, maxv, le], dim=-1)).squeeze(-1)


def reset_hidden_for_done(h: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """Zero per-env recurrent state where the env terminated this step.
    h: [B, N, H], done: [B] bool (or compatible).
    """
    if done.dim() == 1:
        mask = done.view(-1, 1, 1)
    else:
        mask = done.view(-1, 1, 1) if done.dim() == 2 and done.shape[1] == 1 else done
    return torch.where(mask.expand_as(h), torch.zeros_like(h), h)
