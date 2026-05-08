"""AdaIN waveform decoder and task heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdaIN1d(nn.Module):
    def __init__(self, channels: int, z_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(z_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:channels].fill_(1.0)
            self.affine.bias[channels:].zero_()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(z).chunk(2, dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return gamma.unsqueeze(-1) * (x - mean) / std + beta.unsqueeze(-1)


class AdaINConvBlock(nn.Module):
    def __init__(self, channels: int, z_id_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 5, padding=2)
        self.adain = AdaIN1d(channels, z_id_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.adain(self.conv(x), z_id)))


class WaveformDecoder(nn.Module):
    def __init__(self, z_bp_dim: int, z_id_dim: int, seq_len: int = 1250, channels: int = 128, base_len: int = 64, num_blocks: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len, self.channels, self.base_len = seq_len, channels, base_len
        self.fc = nn.Sequential(nn.Linear(z_bp_dim, channels * base_len), nn.GELU())
        self.blocks = nn.ModuleList([AdaINConvBlock(channels, z_id_dim, dropout) for _ in range(num_blocks)])
        self.out = nn.Sequential(nn.Conv1d(channels, channels // 2, 5, padding=2), nn.GELU(), nn.Conv1d(channels // 2, 1, 1))

    def forward(self, z_bp: torch.Tensor, z_id: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        length = int(seq_len or self.seq_len)
        x = self.fc(z_bp).view(z_bp.shape[0], self.channels, self.base_len)
        if self.base_len != length:
            x = F.interpolate(x, size=length, mode="linear", align_corners=False)
        for block in self.blocks:
            x = block(x, z_id)
        return self.out(x).squeeze(1)


class FeatureDecoder(nn.Module):
    def __init__(self, z_dim: int, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, feature_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiskClassifier(nn.Module):
    def __init__(self, z_dim: int, num_classes: int = 3, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_classes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
