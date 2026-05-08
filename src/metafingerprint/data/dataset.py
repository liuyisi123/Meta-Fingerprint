"""NPZ datasets, patient-level splits, and episode sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    abp_mean: float | None = None
    abp_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"input_mean": self.input_mean.tolist(), "input_std": self.input_std.tolist(), "abp_mean": self.abp_mean, "abp_std": self.abp_std}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizationStats":
        return cls(np.asarray(d["input_mean"], dtype=np.float32), np.asarray(d["input_std"], dtype=np.float32), d.get("abp_mean"), d.get("abp_std"))


def _layout(signals: np.ndarray) -> np.ndarray:
    if signals.ndim != 3:
        raise ValueError(f"signals must be 3-D, got {signals.shape}")
    if signals.shape[1] == 2:
        out = signals
    elif signals.shape[2] == 2:
        out = np.transpose(signals, (0, 2, 1))
    else:
        raise ValueError("signals must be [N,2,L] or [N,L,2]")
    return np.asarray(out, dtype=np.float32)


def _squeeze_wave(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 2:
        raise ValueError(f"wave arrays must be [N,L] or [N,1,L], got {x.shape}")
    return x


def _signals_from_npz(raw: Any) -> np.ndarray:
    files = set(raw.files)
    if "signals" in files:
        return _layout(raw["signals"])
    if "ecg" in files and "ppg" in files:
        ecg = _squeeze_wave(raw["ecg"])
        ppg = _squeeze_wave(raw["ppg"])
        if ecg.shape != ppg.shape:
            raise ValueError(f"ecg and ppg shapes differ: {ecg.shape} vs {ppg.shape}")
        return np.stack([ecg, ppg], axis=1).astype(np.float32)
    raise KeyError("NPZ must contain either 'signals' [N,2,L] or both 'ecg' and 'ppg'.")


def _numeric_ids(values: np.ndarray, prefix: str) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.number):
        return raw.astype(np.int64), {}
    mapping: dict[str, int] = {}
    ids = np.zeros(len(raw), dtype=np.int64)
    for i, item in enumerate(raw.astype(str)):
        key = f"{prefix}:{item}"
        if key not in mapping:
            mapping[key] = len(mapping)
        ids[i] = mapping[key]
    return ids, mapping


class NPZWindowDataset(Dataset):
    """Pre-windowed ECG/PPG dataset stored as ``.npz``.

    Accepted input formats:
      * ``signals`` shaped ``[N,2,L]`` or ``[N,L,2]``;
      * or separate ``ecg`` and ``ppg`` arrays shaped ``[N,L]``.

    Optional keys are ``abp``/``targets``, ``labels``/``label``,
    ``patient_id``, ``domain``, and ``split``.
    """

    def __init__(
        self,
        path: str | Path,
        normalize_inputs: bool = True,
        normalize_abp: bool = False,
        stats: NormalizationStats | None = None,
        mmap_mode: str | None = None,
        indices: Sequence[int] | None = None,
        split: str | None = None,
    ) -> None:
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=True, mmap_mode=mmap_mode)
        signals = _signals_from_npz(raw)
        n_all, _, seq_len = signals.shape
        mask = np.ones(n_all, dtype=bool)
        if split is not None:
            if "split" not in raw.files:
                raise ValueError("Requested split but NPZ has no 'split' key.")
            mask &= np.asarray(raw["split"]).astype(str) == split
        if indices is not None:
            idx_mask = np.zeros(n_all, dtype=bool)
            idx_mask[np.asarray(indices, dtype=int)] = True
            mask &= idx_mask
        self.signals = signals[mask]
        self.n, self.c, self.seq_len = self.signals.shape
        abp_key = "abp" if "abp" in raw.files else "targets" if "targets" in raw.files else None
        if abp_key is None:
            self.abp = np.zeros((self.n, self.seq_len), dtype=np.float32)
            self.abp_mask = np.zeros(self.n, dtype=bool)
        else:
            abp = _squeeze_wave(np.asarray(raw[abp_key], dtype=np.float32))[mask]
            if abp.shape != (self.n, self.seq_len):
                raise ValueError(f"abp must be {(self.n, self.seq_len)}, got {abp.shape}")
            self.abp = abp
            self.abp_mask = np.isfinite(abp).all(axis=1)
        label_key = "labels" if "labels" in raw.files else "label" if "label" in raw.files else None
        if label_key is not None:
            self.labels = np.asarray(raw[label_key], dtype=np.int64).reshape(-1)[mask]
            self.label_mask = self.labels >= 0
        else:
            self.labels = np.full(self.n, -1, dtype=np.int64)
            self.label_mask = np.zeros(self.n, dtype=bool)
        patient_raw = raw["patient_id"] if "patient_id" in raw.files else np.arange(n_all)
        self.patient_id, self.patient_mapping = _numeric_ids(np.asarray(patient_raw)[mask], "patient")
        domain_raw = raw["domain"] if "domain" in raw.files else np.zeros(n_all, dtype=np.int64)
        self.domain, self.domain_mapping = _numeric_ids(np.asarray(domain_raw)[mask], "domain")
        self.normalize_inputs = normalize_inputs
        self.normalize_abp = normalize_abp
        self.stats = stats or self._compute_stats()
        self.indices_by_patient: dict[int, np.ndarray] = {}
        for pid in np.unique(self.patient_id):
            self.indices_by_patient[int(pid)] = np.flatnonzero(self.patient_id == pid)
        self.patient_ids = sorted(self.indices_by_patient)

    def _compute_stats(self) -> NormalizationStats:
        mean = self.signals.mean(axis=(0, 2)).astype(np.float32)
        std = np.maximum(self.signals.std(axis=(0, 2)).astype(np.float32), 1e-6)
        if self.normalize_abp and self.abp_mask.any():
            valid = self.abp[self.abp_mask]
            return NormalizationStats(mean, std, float(valid.mean()), float(max(valid.std(), 1e-6)))
        return NormalizationStats(mean, std)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self.signals[index].copy()
        y = self.abp[index].copy()
        if self.normalize_inputs:
            x = (x - self.stats.input_mean[:, None]) / self.stats.input_std[:, None]
        if self.normalize_abp and self.stats.abp_mean is not None and self.stats.abp_std is not None:
            y = (y - self.stats.abp_mean) / self.stats.abp_std
        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "abp_mask": torch.tensor(bool(self.abp_mask[index])),
            "label_mask": torch.tensor(bool(self.label_mask[index])),
            "patient_id": torch.tensor(int(self.patient_id[index]), dtype=torch.long),
            "domain": torch.tensor(int(self.domain[index]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def collate_windows(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples = list(samples)
    return {k: torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k]) else [s[k] for s in samples] for k in samples[0]}


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last, collate_fn=collate_windows)


def patient_level_indices(patient_id: Sequence[int], train_fraction: float = 0.60, val_fraction: float = 0.20, test_fraction: float = 0.20, seed: int = 42) -> dict[str, np.ndarray]:
    pids = np.asarray(patient_id)
    unique = np.unique(pids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(round(len(unique) * train_fraction))
    n_val = int(round(len(unique) * val_fraction))
    train_pids = unique[:n_train]
    val_pids = unique[n_train : n_train + n_val]
    test_pids = unique[n_train + n_val :]
    return {
        "train": np.flatnonzero(np.isin(pids, train_pids)),
        "val": np.flatnonzero(np.isin(pids, val_pids)),
        "test": np.flatnonzero(np.isin(pids, test_pids)),
    }


def create_dataloaders(path: str | Path, data_cfg: Any, training_cfg: Any) -> dict[str, DataLoader]:
    """Create dataloaders from a directory or a single NPZ.

    If ``path`` is a directory, it should contain ``train.npz`` and optional
    ``val.npz``/``test.npz``. If it is one NPZ, an existing ``split`` key is used;
    otherwise a patient-level split is generated from ``patient_id``.
    """

    path = Path(path)
    datasets: dict[str, NPZWindowDataset] = {}
    if path.is_dir():
        train = NPZWindowDataset(path / "train.npz", data_cfg.normalize_inputs, data_cfg.normalize_abp)
        datasets["train"] = train
        for split in ["val", "test"]:
            f = path / f"{split}.npz"
            if f.exists():
                datasets[split] = NPZWindowDataset(f, data_cfg.normalize_inputs, data_cfg.normalize_abp, stats=train.stats)
    else:
        with np.load(path, allow_pickle=True) as raw:
            n = _signals_from_npz(raw).shape[0]
            if "split" in raw.files:
                split_indices = {s: None for s in ["train", "val", "test"]}
            else:
                patient = raw["patient_id"] if "patient_id" in raw.files else np.arange(n)
                patient_num, _ = _numeric_ids(patient, "patient")
                split_indices = patient_level_indices(patient_num, data_cfg.train_fraction, data_cfg.val_fraction, data_cfg.test_fraction, getattr(data_cfg, "split_seed", 42))
        if split_indices.get("train") is None:
            train = NPZWindowDataset(path, data_cfg.normalize_inputs, data_cfg.normalize_abp, split="train")
            datasets["train"] = train
            for s in ["val", "test"]:
                datasets[s] = NPZWindowDataset(path, data_cfg.normalize_inputs, data_cfg.normalize_abp, stats=train.stats, split=s)
        else:
            train = NPZWindowDataset(path, data_cfg.normalize_inputs, data_cfg.normalize_abp, indices=split_indices["train"])
            datasets["train"] = train
            for s in ["val", "test"]:
                datasets[s] = NPZWindowDataset(path, data_cfg.normalize_inputs, data_cfg.normalize_abp, stats=train.stats, indices=split_indices[s])
    return {s: make_loader(ds, training_cfg.batch_size, shuffle=(s == "train"), num_workers=training_cfg.num_workers) for s, ds in datasets.items()}


class EpisodeSampler:
    def __init__(self, dataset: NPZWindowDataset, tasks_per_batch: int = 4, support_size: int = 4, query_size: int = 4, seed: int = 42) -> None:
        self.dataset = dataset
        self.tasks_per_batch = tasks_per_batch
        self.support_size = support_size
        self.query_size = query_size
        self.rng = np.random.default_rng(seed)

    def sample(self) -> list[dict[str, dict[str, Any]]]:
        tasks = []
        replace_patients = len(self.dataset.patient_ids) < self.tasks_per_batch
        for pid in self.rng.choice(self.dataset.patient_ids, self.tasks_per_batch, replace=replace_patients):
            idxs = self.dataset.indices_by_patient[int(pid)]
            need = self.support_size + self.query_size
            chosen = self.rng.choice(idxs, need, replace=len(idxs) < need)
            tasks.append({
                "support": collate_windows([self.dataset[int(i)] for i in chosen[: self.support_size]]),
                "query": collate_windows([self.dataset[int(i)] for i in chosen[self.support_size :]]),
            })
        return tasks


WindowDataset = NPZWindowDataset
