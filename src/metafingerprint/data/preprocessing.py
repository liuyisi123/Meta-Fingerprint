"""Preprocessing helpers for continuous signals."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if orig_fs == target_fs:
        return x.copy()
    if orig_fs > target_fs:
        x = cheby_antialias_filter(x, orig_fs, cutoff_hz=min(45.0, 0.45 * target_fs), order=6)
    gcd = np.gcd(int(round(orig_fs)), int(round(target_fs)))
    return signal.resample_poly(x, int(round(target_fs)) // gcd, int(round(orig_fs)) // gcd).astype(np.float32)


def cheby_antialias_filter(x: np.ndarray, fs: float, cutoff_hz: float = 45.0, order: int = 6) -> np.ndarray:
    cutoff = min(cutoff_hz / (0.5 * fs), 0.95)
    b, a = signal.cheby1(order, 0.05, cutoff, btype="low")
    return signal.filtfilt(b, a, np.asarray(x, dtype=np.float32)).astype(np.float32)


def make_windows(ecg: np.ndarray, ppg: np.ndarray, abp: np.ndarray | None = None, fs: float = 125.0, window_sec: float = 10.0, stride_sec: float | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    ecg = np.asarray(ecg, dtype=np.float32)
    ppg = np.asarray(ppg, dtype=np.float32)
    if ecg.shape != ppg.shape:
        raise ValueError("ecg and ppg must have same length")
    length = int(round(window_sec * fs))
    stride = int(round((stride_sec or window_sec) * fs))
    starts = np.arange(0, len(ecg) - length + 1, stride)
    signals = np.zeros((len(starts), 2, length), dtype=np.float32)
    y = np.zeros((len(starts), length), dtype=np.float32) if abp is not None else None
    for i, s in enumerate(starts):
        signals[i, 0] = ecg[s : s + length]
        signals[i, 1] = ppg[s : s + length]
        if y is not None:
            y[i] = np.asarray(abp, dtype=np.float32)[s : s + length]
    return signals, y


def basic_signal_quality(signals: np.ndarray, abp: np.ndarray | None = None) -> np.ndarray:
    signals = np.asarray(signals)
    mask = np.isfinite(signals).all(axis=(1, 2)) & ((signals.max(axis=2) - signals.min(axis=2)) > 1e-4).all(axis=1)
    if abp is not None:
        abp = np.asarray(abp)
        pp = abp.max(axis=1) - abp.min(axis=1)
        mask &= np.isfinite(abp).all(axis=1) & (abp.min(axis=1) > 20) & (abp.max(axis=1) < 260) & (pp > 5)
    return mask


def bp_scalar_labels(abp: np.ndarray) -> np.ndarray:
    """Three-class acute phenotype for ICD-10 cohorts (MC-MED Task-B).

    Classes: 0 = Hypotension, 1 = Normal, 2 = Hypertension.
    Matches the collapsed ICD-10 label space where pre-hypertension
    cannot be distinguished from normotension.
    """
    sbp = np.asarray(abp).max(axis=1)
    dbp = np.asarray(abp).min(axis=1)
    map_est = dbp + (sbp - dbp) / 3.0
    labels = np.ones(len(sbp), dtype=np.int64)
    labels[(sbp < 90) | (map_est < 65)] = 0
    labels[(sbp >= 130) | (dbp >= 80)] = 2
    return labels


def bp_scalar_labels_4class(abp: np.ndarray) -> np.ndarray:
    """Four-class acute phenotype for ABP-equipped cohorts (Settings A-C Task-B).

    Uses 2017 ACC/AHA SBP thresholds:
      0 = Hypotension  : SBP < 90 mmHg or MAP < 65 mmHg
      1 = Normal       : SBP 90-119 mmHg and DBP < 80 mmHg
      2 = Pre-HTN      : SBP 120-129 mmHg and DBP < 80 mmHg
      3 = Hypertension : SBP >= 130 mmHg or DBP >= 80 mmHg

    Apply to VitalDB, MIMIC-III-Ext-PPG, UCI, and RWW cohorts which
    all carry continuous invasive or CNAP-referenced ABP reference.
    """
    sbp = np.asarray(abp).max(axis=1)
    dbp = np.asarray(abp).min(axis=1)
    map_est = dbp + (sbp - dbp) / 3.0
    labels = np.ones(len(sbp), dtype=np.int64)
    labels[(sbp < 90) | (map_est < 65)] = 0
    labels[(sbp >= 120) & (sbp < 130) & (dbp < 80)] = 2
    labels[(sbp >= 130) | (dbp >= 80)] = 3
    return labels


def save_npz_windows(path: str | Path, signals: np.ndarray, abp: np.ndarray | None = None, labels: np.ndarray | None = None, patient_id: Iterable[int | str] | None = None, domain: Iterable[int | str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"signals": np.asarray(signals, dtype=np.float32)}
    if abp is not None:
        payload["abp"] = np.asarray(abp, dtype=np.float32)
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=np.int64)
    payload["patient_id"] = np.asarray(list(patient_id)) if patient_id is not None else np.arange(payload["signals"].shape[0])
    if domain is not None:
        payload["domain"] = np.asarray(list(domain))
    np.savez_compressed(path, **payload)
