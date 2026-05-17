"""Student world model with separated embeddings, residual blocks, and multi-layer GRU."""
from __future__ import annotations
import torch
from torch import nn

class ResidualBlock(nn.Module):
    """Preserves low-level dynamics across deep layers to prevent gradient decay."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 2,
        use_gru: bool = True,
        delta_limit: float = 5.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.num_layers = int(num_layers)

        # 1. Separate Embeddings
        self.obs_emb = nn.Sequential(nn.Linear(obs_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU())
        self.act_emb = nn.Sequential(nn.Linear(act_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU())
        
        # 2. Residual Encoder
        self.encoder = ResidualBlock(hidden_dim)
        
        # 3. Multi-Layer GRU
        if self.use_gru:
            # batch_first=True allows us to pass (B, L, D) tensors easily
            self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
            self.hidden_norm = nn.LayerNorm(hidden_dim)
        else:
            self.gru = None

        # 4. Residual Head
        self.head = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, obs_dim)
        )

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        # nn.GRU expects hidden state of shape (num_layers, batch_size, hidden_dim)
        return torch.zeros(self.num_layers, batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # Embed separately, then concatenate
        obs_feat = self.obs_emb(obs_norm)
        act_feat = self.act_emb(act_norm)
        feat = torch.cat([obs_feat, act_feat], dim=-1)
        
        feat = self.encoder(feat)
        
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            # predict_next passes 2D (B, D) tensors. nn.GRU needs 3D sequence (B, 1, D)
            feat, hidden = self.gru(feat.unsqueeze(1), hidden)
            feat = feat.squeeze(1) # Back to (B, D)
            feat = self.hidden_norm(feat)
            
        raw_delta = self.head(feat)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
