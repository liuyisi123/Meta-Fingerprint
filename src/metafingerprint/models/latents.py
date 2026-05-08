"""Structured latent encoders."""
from __future__ import annotations

import torch
from torch import nn


class MLPPosterior(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-8, 6)


class TemporalPosterior(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, hidden_dim, 5, padding=2), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU())
        self.attn = nn.Sequential(nn.Linear(hidden_dim, max(hidden_dim // 2, 1)), nn.Tanh(), nn.Linear(max(hidden_dim // 2, 1), 1))
        self.posterior = MLPPosterior(hidden_dim, latent_dim, hidden_dim, dropout)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.conv(H.transpose(1, 2)).transpose(1, 2)
        w = torch.softmax(self.attn(feats).squeeze(-1), dim=1)
        return self.posterior(torch.sum(feats * w.unsqueeze(-1), dim=1))


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
    if not sample:
        return mu
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class StructuredLatentEncoder(nn.Module):
    def __init__(self, feature_dim: int, z_id_dim: int = 64, z_bp_dim: int = 64, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.structural = MLPPosterior(feature_dim, z_id_dim, hidden_dim, dropout)
        self.dynamic = TemporalPosterior(feature_dim, z_bp_dim, hidden_dim, dropout)

    def forward(self, H: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        h_bar = H.mean(dim=1)
        mu_id, logvar_id = self.structural(h_bar)
        mu_bp, logvar_bp = self.dynamic(H)
        return {
            "z_id": reparameterize(mu_id, logvar_id, sample),
            "z_bp": reparameterize(mu_bp, logvar_bp, sample),
            "mu_id": mu_id,
            "logvar_id": logvar_id,
            "mu_bp": mu_bp,
            "logvar_bp": logvar_bp,
            "h_bar": h_bar,
        }
