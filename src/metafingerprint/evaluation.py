"""Evaluation helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import compute_total_loss
from .metrics import classification_metrics, waveform_metrics
from .utils import move_to_device


def _maybe_unscale(values: np.ndarray, loader: DataLoader) -> np.ndarray:
    ds = getattr(loader, "dataset", None)
    stats = getattr(ds, "stats", None)
    if bool(getattr(ds, "normalize_abp", False)) and stats is not None and stats.abp_mean is not None and stats.abp_std is not None:
        return values * float(stats.abp_std) + float(stats.abp_mean)
    return values


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, training_cfg: Any, device: torch.device, compute_dtw: bool = False) -> dict[str, Any]:
    model.eval()
    preds: list[np.ndarray] = []; trues: list[np.ndarray] = []; logits_all: list[np.ndarray] = []; labels_all: list[np.ndarray] = []; taus: list[np.ndarray] = []
    losses: list[float] = []; wave_losses: list[float] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        out = model(batch["x"], sample=False)
        lb = compute_total_loss(out, batch, training_cfg, global_step=training_cfg.dis_warmup_steps)
        losses.append(float(lb.total.detach().cpu())); wave_losses.append(float(lb.wave.detach().cpu()))
        am = batch["abp_mask"].detach().cpu().numpy().astype(bool)
        if am.any():
            preds.append(out["waveform"].detach().cpu().numpy()[am]); trues.append(batch["y"].detach().cpu().numpy()[am])
        lm = batch["label_mask"].detach().cpu().numpy().astype(bool)
        if lm.any():
            logits_all.append(out["logits"].detach().cpu().numpy()[lm]); labels_all.append(batch["label"].detach().cpu().numpy()[lm])
        taus.append(out["tau"].detach().cpu().numpy())
    metrics: dict[str, Any] = {"loss": float(np.mean(losses)) if losses else float("nan"), "wave_loss": float(np.mean(wave_losses)) if wave_losses else float("nan")}
    if preds:
        pred = _maybe_unscale(np.concatenate(preds, axis=0), loader); true = _maybe_unscale(np.concatenate(trues, axis=0), loader)
        metrics.update(waveform_metrics(pred, true, compute_dtw=compute_dtw))
    if logits_all:
        metrics.update({f"cls_{k}": v for k, v in classification_metrics(np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)).items()})
    if taus:
        tau = np.concatenate(taus, axis=0)
        metrics.update({"tau_mean_ms": float(tau.mean() * 1000), "tau_min_ms": float(tau.min() * 1000), "tau_max_ms": float(tau.max() * 1000)})
    return metrics
