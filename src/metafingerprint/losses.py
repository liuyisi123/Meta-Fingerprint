"""Losses for Meta-Fingerprint."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .config import TrainingConfig

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    wave: torch.Tensor
    pheno: torch.Tensor
    dis: torch.Tensor
    feature_recon: torch.Tensor
    mi: torch.Tensor
    tc: torch.Tensor
    mkl: torch.Tensor
    consistency: torch.Tensor
    lambda_dis_eff: float

    def detached(self) -> dict[str, float]:
        return {"loss_total": float(self.total.detach().cpu()), "loss_wave": float(self.wave.detach().cpu()), "loss_pheno": float(self.pheno.detach().cpu()), "loss_dis": float(self.dis.detach().cpu()), "loss_feature_recon": float(self.feature_recon.detach().cpu()), "loss_mi": float(self.mi.detach().cpu()), "loss_tc": float(self.tc.detach().cpu()), "loss_mkl": float(self.mkl.detach().cpu()), "loss_consistency": float(self.consistency.detach().cpu()), "lambda_dis_eff": float(self.lambda_dis_eff)}


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, delta: float = 5.0) -> torch.Tensor:
    if mask is None:
        mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    mask = mask.bool().to(pred.device)
    if int(mask.sum()) == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[mask], target.to(pred.device)[mask], delta=delta, reduction="mean")


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    labels = labels.to(logits.device)
    if mask is None:
        mask = labels >= 0
    mask = mask.bool().to(logits.device) & (labels >= 0)
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask], reduction="mean")


def _log_normal(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * (_LOG_2PI + logvar + (x - mu) ** 2 / torch.exp(logvar))


def _logmeanexp(v: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(v, dim=dim) - torch.log(torch.tensor(v.shape[dim], device=v.device, dtype=v.dtype))


def tc_decomposition(z_id: torch.Tensor, z_bp: torch.Tensor, mu_id: torch.Tensor, logvar_id: torch.Tensor, mu_bp: torch.Tensor, logvar_bp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.cat([z_id, z_bp], dim=-1)
    mu = torch.cat([mu_id, mu_bp], dim=-1)
    logvar = torch.cat([logvar_id, logvar_bp], dim=-1)
    log_q_zCx = _log_normal(z, mu, logvar).sum(dim=-1)
    log_qz = _logmeanexp(_log_normal(z[:, None, :], mu[None, :, :], logvar[None, :, :]).sum(dim=-1), dim=1)
    log_qid = _logmeanexp(_log_normal(z_id[:, None, :], mu_id[None, :, :], logvar_id[None, :, :]).sum(dim=-1), dim=1)
    log_qbp = _logmeanexp(_log_normal(z_bp[:, None, :], mu_bp[None, :, :], logvar_bp[None, :, :]).sum(dim=-1), dim=1)
    log_prod = log_qid + log_qbp
    log_pz = _log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(dim=-1)
    return F.relu((log_q_zCx - log_qz).mean()), F.relu((log_qz - log_prod).mean()), F.relu((log_prod - log_pz).mean())


def temporal_consistency_loss(z_id: torch.Tensor, patient_id: torch.Tensor | None, margin: float = 0.9) -> torch.Tensor:
    if patient_id is None:
        return z_id.sum() * 0.0
    patient_id = patient_id.to(z_id.device)
    z = F.normalize(z_id, dim=-1)
    losses = []
    for pid in torch.unique(patient_id.detach()):
        idx = torch.nonzero(patient_id == pid, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        sims = z[idx] @ z[idx].T
        tri = torch.triu(torch.ones_like(sims, dtype=torch.bool), diagonal=1)
        losses.append(F.relu(margin - sims[tri]).mean())
    return torch.stack(losses).mean() if losses else z_id.sum() * 0.0


def compute_disentanglement_loss(outputs: dict[str, torch.Tensor], patient_id: torch.Tensor | None, cfg: TrainingConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature_recon = F.mse_loss(outputs["recon_h_bar"], outputs["h_bar"])
    mi, tc, mkl = tc_decomposition(outputs["z_id"], outputs["z_bp"], outputs["mu_id"], outputs["logvar_id"], outputs["mu_bp"], outputs["logvar_bp"])
    consistency = temporal_consistency_loss(outputs["z_id"], patient_id, cfg.consistency_margin)
    dis = feature_recon + cfg.alpha_mi * mi + cfg.beta_tc * tc + cfg.gamma_mkl * mkl + cfg.lambda_consistency * consistency
    return dis, {"feature_recon": feature_recon, "mi": mi, "tc": tc, "mkl": mkl, "consistency": consistency}


def compute_total_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], cfg: TrainingConfig, global_step: int = 0) -> LossBreakdown:
    wave = masked_huber_loss(outputs["waveform"], batch["y"], batch.get("abp_mask"), cfg.huber_delta)
    pheno = masked_cross_entropy(outputs["logits"], batch["label"], batch.get("label_mask")) if "label" in batch else outputs["logits"].sum() * 0.0
    pid = batch.get("patient_id")
    if torch.is_tensor(pid):
        pid = pid.to(outputs["waveform"].device)
    dis, terms = compute_disentanglement_loss(outputs, pid, cfg)
    warm = min(1.0, float(global_step + 1) / float(max(1, cfg.dis_warmup_steps)))
    ldis = cfg.lambda_dis * warm
    total = cfg.lambda_wave * wave + cfg.lambda_pheno * pheno + ldis * dis
    return LossBreakdown(total, wave, pheno, dis, terms["feature_recon"], terms["mi"], terms["tc"], terms["mkl"], terms["consistency"], ldis)


def structural_anchor_loss(current: torch.nn.Module, reference_state: dict[str, torch.Tensor]) -> torch.Tensor:
    loss: torch.Tensor | None = None
    for name, p in current.named_parameters():
        if name in reference_state:
            term = (p - reference_state[name].to(p.device)).pow(2).mean()
            loss = term if loss is None else loss + term
    if loss is None:
        first = next(current.parameters())
        return first.sum() * 0.0
    return loss
