#!/usr/bin/env python
"""Adapt only the structural branch on support data, then evaluate query data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.adaptation import adapt_structural_branch
from metafingerprint.config import load_config
from metafingerprint.data import NPZWindowDataset, make_loader
from metafingerprint.evaluation import evaluate_model
from metafingerprint.models import MetaFingerprintModel
from metafingerprint.utils import load_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support", required=True, help="Support/calibration NPZ.")
    p.add_argument("--query", required=True, help="Query/evaluation NPZ.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--inner-steps", type=int, default=None)
    p.add_argument("--inner-lr", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else ROOT / "configs" / "default.yaml")
    device = resolve_device(args.device)
    model = MetaFingerprintModel(cfg.model).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    support_ds = NPZWindowDataset(args.support, cfg.data.normalize_inputs, cfg.data.normalize_abp)
    query_ds = NPZWindowDataset(args.query, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=support_ds.stats)
    support_loader = make_loader(support_ds, min(cfg.training.batch_size, len(support_ds)), shuffle=True, num_workers=0)
    query_loader = make_loader(query_ds, cfg.training.batch_size, shuffle=False, num_workers=0)
    adapted = adapt_structural_branch(model, support_loader, cfg.training, device, steps=args.inner_steps, lr=args.inner_lr)
    metrics = evaluate_model(adapted, query_loader, cfg.training, device, compute_dtw=False)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
