"""HTD-SSM encoder with bounded fractional ECG-PPG delay."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ode import NeuralODEFrontEnd


def fractional_delay(ppg_features: torch.Tensor, delays_sec: torch.Tensor, fs: float) -> torch.Tensor:
    if ppg_features.ndim != 3 or delays_sec.ndim != 2:
        raise ValueError("ppg_features must be [B,L,C] and delays_sec [B,L]")
    bsz, length, channels = ppg_features.shape
    lag = delays_sec * fs
    lag_int = torch.floor(lag).long()
    gamma = (lag - lag_int.to(lag.dtype)).clamp(0, 1)
    base = torch.arange(length, device=ppg_features.device).view(1, length).expand(bsz, length)
    idx0 = base - lag_int
    idx1 = idx0 - 1
    mask0 = (idx0 >= 0) & (idx0 < length)
    mask1 = (idx1 >= 0) & (idx1 < length)
    g0 = torch.gather(ppg_features, 1, idx0.clamp(0, length - 1).unsqueeze(-1).expand(-1, -1, channels)) * mask0.unsqueeze(-1).to(ppg_features.dtype)
    g1 = torch.gather(ppg_features, 1, idx1.clamp(0, length - 1).unsqueeze(-1).expand(-1, -1, channels)) * mask1.unsqueeze(-1).to(ppg_features.dtype)
    return (1 - gamma).unsqueeze(-1) * g0 + gamma.unsqueeze(-1) * g1


class HTDSSMEncoder(nn.Module):
    def __init__(self, fs: float = 125.0, local_dim: int = 64, encoder_dim: int = 128, ode_hidden_dim: int = 96, ode_steps: int = 2, ssm_hidden_dim: int = 128, tau_min: float = 0.10, tau_max: float = 0.40, dropout: float = 0.1, spectral_norm: bool = True) -> None:
        super().__init__()
        self.fs, self.tau_min, self.tau_max = fs, tau_min, tau_max
        self.ssm_hidden_dim = ssm_hidden_dim
        self.ecg_frontend = NeuralODEFrontEnd(local_dim, ode_hidden_dim, ode_steps, dropout, spectral_norm)
        self.ppg_frontend = NeuralODEFrontEnd(local_dim, ode_hidden_dim, ode_steps, dropout, spectral_norm)
        self.delay_gate = nn.Linear(local_dim, 1)
        self.delta_gate = nn.Linear(local_dim, ssm_hidden_dim)
        self.ecg_in = nn.Linear(local_dim, ssm_hidden_dim)
        self.ppg_in = nn.Linear(local_dim, ssm_hidden_dim)
        self.log_neg_a = nn.Parameter(torch.zeros(ssm_hidden_dim))
        self.out_proj = nn.Linear(ssm_hidden_dim, encoder_dim)
        self.skip_proj = nn.Linear(2 * local_dim, encoder_dim)
        self.norm = nn.LayerNorm(encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def estimate_delay(self, g_ecg: torch.Tensor) -> torch.Tensor:
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.delay_gate(g_ecg).squeeze(-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"expected [B,2,L], got {tuple(x.shape)}")
        g_ecg = self.ecg_frontend(x[:, 0])
        g_ppg = self.ppg_frontend(x[:, 1])
        tau = self.estimate_delay(g_ecg)
        p_del = fractional_delay(g_ppg, tau, self.fs)
        u = self.ecg_in(g_ecg) + self.ppg_in(p_del)
        delta = (1.0 / self.fs) * (0.5 + torch.sigmoid(self.delta_gate(g_ecg)))
        a = -F.softplus(self.log_neg_a).view(1, -1) - 1e-4
        h = torch.zeros(x.shape[0], self.ssm_hidden_dim, device=x.device, dtype=x.dtype)
        ys = []
        for k in range(x.shape[-1]):
            a_bar = torch.exp(delta[:, k, :] * a)
            b_fac = torch.expm1(delta[:, k, :] * a) / a
            h = a_bar * h + b_fac * u[:, k, :]
            y = self.out_proj(h) + self.skip_proj(torch.cat([g_ecg[:, k, :], p_del[:, k, :]], dim=-1))
            ys.append(y)
        return self.norm(self.dropout(torch.stack(ys, dim=1))), tau
