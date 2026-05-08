#!/usr/bin/env python
"""Train Meta-Fingerprint on a dataset directory or NPZ file."""
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
from metafingerprint.models import MetaFingerprintModel
from metafingerprint.train import Trainer
from metafingerprint.utils import count_parameters, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Dataset directory with train/val/test NPZ files, or one NPZ with split/patient_id.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--output", default=str(ROOT / "runs" / "default"))
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.training.device = args.device
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    seed_everything(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    loaders = create_dataloaders(args.data, cfg.data, cfg.training)
    model = MetaFingerprintModel(cfg.model)
    print(f"Model parameters: {count_parameters(model, trainable_only=False):,}")
    trainer = Trainer(model, cfg, device, args.output)
    result = trainer.fit(loaders["train"], loaders.get("val"))
    print(result)


if __name__ == "__main__":
    main()
