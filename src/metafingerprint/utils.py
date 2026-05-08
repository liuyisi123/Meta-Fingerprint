"""Shared utilities."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Respect explicit CPU-thread limits set by CI/smoke scripts.  We do not
    # force these values for normal training because multi-threaded BLAS can be
    # beneficial on larger machines.
    if "TORCH_NUM_THREADS" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["TORCH_NUM_THREADS"])))
    if "TORCH_INTEROP_THREADS" in os.environ:
        torch.set_num_interop_threads(max(1, int(os.environ["TORCH_INTEROP_THREADS"])))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _asdict(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) else obj


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, scheduler=None, config=None, epoch=None, metrics=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = _asdict(config)
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
