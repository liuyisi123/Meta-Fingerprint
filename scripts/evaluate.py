#!/usr/bin/env python
"""Evaluate a checkpoint on train/val/test data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.config import load_config
from metafingerprint.data import create_dataloaders
from metafingerprint.evaluation import evaluate_model
from metafingerprint.models import MetaFingerprintModel
from metafingerprint.utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else ROOT / "configs" / "default.yaml")
    device = resolve_device(args.device)
    loaders = create_dataloaders(args.data, cfg.data, cfg.training)
    model = MetaFingerprintModel(cfg.model).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate_model(model, loaders[args.split], cfg.training, device, compute_dtw=False)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
