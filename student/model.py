"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 5.0,          # widened: pendulum can spike hard
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim

        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # LayerNorm on hidden state — critical for long-horizon stability
        self.hidden_norm = nn.LayerNorm(hidden_dim) if self.use_gru else None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, obs_dim),
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm, act_norm, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.encoder(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            hidden = self.hidden_norm(hidden)
            feat = hidden
        # Skip connection: concatenate original input features into head
        raw_delta = self.head(torch.cat([feat, self.input_proj(x)], dim=-1))
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
