"""Spectrally normalized Neural-ODE style morphology front end."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrizations


def maybe_spectral_norm(module: nn.Module, enabled: bool = True) -> nn.Module:
    return parametrizations.spectral_norm(module) if enabled and isinstance(module, nn.Linear) else module


class ODEFunction(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, spectral_norm: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(maybe_spectral_norm(nn.Linear(dim, hidden_dim), spectral_norm), nn.Tanh(), nn.Dropout(dropout), maybe_spectral_norm(nn.Linear(hidden_dim, dim), spectral_norm))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NeuralODEFrontEnd(nn.Module):
    """Fixed-step RK4 continuous-depth block applied to each sample."""

    def __init__(self, out_dim: int, hidden_dim: int, ode_steps: int = 2, dropout: float = 0.0, spectral_norm: bool = True) -> None:
        super().__init__()
        self.ode_steps = max(1, int(ode_steps))
        self.input_proj = maybe_spectral_norm(nn.Linear(1, out_dim), spectral_norm)
        self.func = ODEFunction(out_dim, hidden_dim, dropout, spectral_norm)
        self.norm = nn.LayerNorm(out_dim)

    def _rk4(self, h: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.func(h); k2 = self.func(h + 0.5 * dt * k1); k3 = self.func(h + 0.5 * dt * k2); k4 = self.func(h + dt * k3)
        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [B,L], got {tuple(x.shape)}")
        h = self.input_proj(x.unsqueeze(-1))
        dt = 1.0 / self.ode_steps
        for _ in range(self.ode_steps):
            h = self._rk4(h, dt)
        return self.norm(h)
