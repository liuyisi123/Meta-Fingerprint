"""Top-level Meta-Fingerprint model."""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .decoder import FeatureDecoder, RiskClassifier, WaveformDecoder
from .htd_ssm import HTDSSMEncoder
from .latents import StructuredLatentEncoder


class MetaFingerprintModel(nn.Module):
    """HTD-SSM -> TC-CVAE latents -> AdaIN waveform/risk heads."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.encoder = HTDSSMEncoder(c.fs, c.local_dim, c.encoder_dim, c.ode_hidden_dim, c.ode_steps, c.ssm_hidden_dim, c.tau_min, c.tau_max, c.dropout, c.use_spectral_norm)
        hidden = max(2 * c.encoder_dim, 128)
        self.latents = StructuredLatentEncoder(c.encoder_dim, c.z_id_dim, c.z_bp_dim, hidden, c.dropout)
        z_total = c.z_id_dim + c.z_bp_dim
        self.feature_decoder = FeatureDecoder(z_total, c.encoder_dim, hidden, c.dropout)
        self.wave_decoder = WaveformDecoder(c.z_bp_dim, c.z_id_dim, c.seq_len, c.decoder_channels, c.decoder_base_len, dropout=c.dropout)
        self.classifier = RiskClassifier(z_total, c.num_classes, max(c.decoder_channels, 64), c.dropout)

    @property
    def structural_encoder(self) -> nn.Module:
        return self.latents.structural

    def structural_parameters(self):
        return self.structural_encoder.parameters()

    def non_structural_parameters(self):
        ids = {id(p) for p in self.structural_parameters()}
        return (p for p in self.parameters() if id(p) not in ids)

    def encode(self, x: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        H, tau = self.encoder(x)
        latent = self.latents(H, sample=self.training if sample is None else sample)
        latent["H"] = H
        latent["tau"] = tau
        return latent

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> dict[str, torch.Tensor]:
        latent = self.encode(x, sample=sample)
        z = torch.cat([latent["z_id"], latent["z_bp"]], dim=-1)
        return {**latent, "z": z, "recon_h_bar": self.feature_decoder(z), "waveform": self.wave_decoder(latent["z_bp"], latent["z_id"], x.shape[-1]), "logits": self.classifier(z)}
