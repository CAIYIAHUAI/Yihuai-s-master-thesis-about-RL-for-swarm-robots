"""Tolstaya-style K-tap graph-filter actor baselines.

These actors match the GATGRUActor forward signature so existing BC/eval
infrastructure can swap them in via cfg.actor_type. The pure variant uses only
the normalized communication topology; the edge-aware variant gates neighbor
features with an MLP over edge_attr for a fairer comparison under the v18d
observation schema.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from observations import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_ACTIONS


class TolstayaActor(nn.Module):
    def __init__(self, node_dim: int = NODE_FEATURE_DIM, edge_dim: int = EDGE_FEATURE_DIM,
                 gat_hidden: int = 128, gat_heads: int = 4, gru_hidden: int = 256,
                 stop_action_bias: float = 0.0,
                 action_log_std_init: float = -1.9,
                 k_taps: int = 3,
                 use_edge_attr: bool = True):
        super().__init__()
        del gat_heads
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = gru_hidden
        self.k_taps = int(k_taps)
        self.use_edge_attr = bool(use_edge_attr)
        self.action_mode = "continuous"

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden), nn.GELU(),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, gat_hidden), nn.GELU(),
            nn.Linear(gat_hidden, gat_hidden),
        )
        self.filter1 = nn.ModuleList([nn.Linear(gat_hidden, gat_hidden)
                                      for _ in range(self.k_taps + 1)])
        self.filter2 = nn.ModuleList([nn.Linear(gat_hidden, gat_hidden)
                                      for _ in range(self.k_taps + 1)])
        self.to_hidden = nn.Linear(gat_hidden, gru_hidden)
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

    def _propagate(self, x: torch.Tensor, edge_index: torch.Tensor,
                   edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        if src.numel() == 0:
            return x
        if self.use_edge_attr:
            gate = torch.sigmoid(self.edge_gate(edge_attr))
            msg = x[src] * gate
        else:
            msg = x[src]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        deg = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], 1, dtype=x.dtype, device=x.device))
        return (out + x) / deg.clamp_min(1.0)

    def _ktap_filter(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        taps = [x]
        h = x
        for _ in range(self.k_taps):
            h = self._propagate(h, edge_index, edge_attr)
            taps.append(h)
        y = torch.zeros_like(x)
        for tap, layer in zip(taps, layers):
            y = y + layer(tap)
        return F.gelu(y)

    def forward(self, obs_nodes: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, h_in: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del h_in
        B, N, _ = obs_nodes.shape
        x = self.node_encoder(obs_nodes.reshape(B * N, -1))
        x = self._ktap_filter(x, edge_index, edge_attr, self.filter1)
        x = self._ktap_filter(x, edge_index, edge_attr, self.filter2)
        features = self.to_hidden(x)
        h_next = torch.zeros(B, N, self.hidden, dtype=features.dtype, device=features.device)
        logits = self.policy_head(features).reshape(B, N, N_ACTIONS)
        velocity_aux = self.velocity_head(features).reshape(B, N, 2)
        return logits, velocity_aux, h_next
