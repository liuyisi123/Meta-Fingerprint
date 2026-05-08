"""Synthetic ECG/PPG/ABP generator for demos and tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocessing import bp_scalar_labels, save_npz_windows


def _gaussian(t: np.ndarray, c: float, w: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - c) / w) ** 2)


def _pulse(t: np.ndarray, onset: float, rise: float, decay: float) -> np.ndarray:
    x = np.maximum(t - onset, 0.0)
    p = (1 - np.exp(-x / max(rise, 1e-4))) * np.exp(-x / max(decay, 1e-4))
    return p / max(float(p.max()), 1e-6)


def generate_patient_window(rng: np.random.Generator, fs: float, seq_len: int, heart_rate: float, delay_sec: float, sbp: float, dbp: float, compliance: float, motion_noise: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(seq_len, dtype=np.float32) / fs
    rr = 60.0 / heart_rate
    beats = np.arange(-rr, seq_len / fs + rr, rr) + rng.normal(0, 0.025, size=int(seq_len / fs / rr) + 3)
    ecg = np.zeros(seq_len, dtype=np.float32)
    ppg = np.zeros(seq_len, dtype=np.float32)
    abp = np.full(seq_len, dbp, dtype=np.float32)
    pp = max(sbp - dbp, 10.0)
    for bt in beats:
        ecg += 0.10 * _gaussian(t, bt - 0.18, 0.035) - 0.15 * _gaussian(t, bt - 0.025, 0.010)
        ecg += _gaussian(t, bt, 0.012) - 0.25 * _gaussian(t, bt + 0.030, 0.012) + 0.28 * _gaussian(t, bt + 0.25, 0.08)
        onset = bt + delay_sec + rng.normal(0, 0.006)
        ppg += _pulse(t, onset, 0.055, 0.23 + 0.12 * compliance)
        abp += pp * _pulse(t, onset - 0.03, 0.035, 0.30 + 0.10 * compliance)
        abp -= 0.08 * pp * _gaussian(t, onset + 0.22, 0.030)
    ecg = (ecg - ecg.mean()) / max(float(ecg.std()), 1e-6)
    ppg = (ppg - ppg.min()) / max(float(ppg.max() - ppg.min()), 1e-6)
    ppg += 0.03 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
    ecg += motion_noise * rng.normal(size=seq_len)
    ppg += motion_noise * rng.normal(size=seq_len)
    abp += 1.5 * rng.normal(size=seq_len)
    return ecg.astype(np.float32), ppg.astype(np.float32), abp.astype(np.float32)


def make_synthetic_dataset(output_dir: str | Path, n_patients: int = 48, windows_per_patient: int = 12, fs: float = 125.0, seq_len: int = 1250, seed: int = 42) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals, abps, pids, domains = [], [], [], []
    for pid in range(n_patients):
        vascular = rng.uniform(0, 1)
        base_hr = rng.normal(73, 9)
        delay = np.clip(rng.normal(0.16 + 0.08 * vascular, 0.025), 0.10, 0.40)
        compliance = np.clip(1 - vascular + rng.normal(0, 0.10), 0.2, 1.5)
        dbp = np.clip(rng.normal(70 + 10 * vascular, 8), 45, 105)
        sbp = np.clip(dbp + rng.normal(42 + 18 * vascular, 8), 85, 190)
        domain = "synthetic-source" if pid < int(0.8 * n_patients) else "synthetic-shift"
        for _ in range(windows_per_patient):
            ecg, ppg, abp = generate_patient_window(rng, fs, seq_len, float(np.clip(base_hr + rng.normal(0, 4), 45, 125)), float(delay + rng.normal(0, 0.010)), float(np.clip(sbp + rng.normal(0, 7), 80, 205)), float(np.clip(dbp + rng.normal(0, 4), 35, 120)), float(compliance))
            signals.append(np.stack([ecg, ppg]))
            abps.append(abp)
            pids.append(pid)
            domains.append(domain)
    signals_arr = np.stack(signals).astype(np.float32)
    abp_arr = np.stack(abps).astype(np.float32)
    labels = bp_scalar_labels(abp_arr)
    patient_id = np.asarray(pids, dtype=np.int64)
    patients = np.arange(n_patients)
    rng.shuffle(patients)
    n_train, n_val = int(0.6 * n_patients), int(0.2 * n_patients)
    split_sets = {"train": set(patients[:n_train]), "val": set(patients[n_train:n_train+n_val]), "test": set(patients[n_train+n_val:])}
    paths: dict[str, Path] = {}
    for name, ids in split_sets.items():
        mask = np.array([pid in ids for pid in patient_id])
        path = output_dir / f"{name}.npz"
        save_npz_windows(path, signals_arr[mask], abp_arr[mask], labels[mask], patient_id[mask], np.asarray(domains)[mask])
        paths[name] = path
    return paths
