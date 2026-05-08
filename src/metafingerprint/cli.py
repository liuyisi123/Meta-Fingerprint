"""Command-line entry points."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import numpy as np
import torch

from .adaptation import adapt_structural_branch
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig, load_config
from .data import NPZWindowDataset, NormalizationStats, create_dataloaders, make_loader, make_synthetic_dataset
from .evaluation import evaluate_model
from .models import MetaFingerprintModel
from .train import Trainer
from .utils import load_checkpoint, resolve_device, save_checkpoint, save_json, seed_everything


def _load_stats(path: str | None) -> NormalizationStats | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return NormalizationStats.from_dict(json.load(f))


def _config_from_checkpoint_or_file(ckpt: dict[str, Any], config_path: str | None) -> ExperimentConfig:
    if config_path:
        return load_config(config_path)
    if "config" in ckpt:
        raw = ckpt["config"]
        return ExperimentConfig(ModelConfig(**raw.get("model", {})), TrainingConfig(**raw.get("training", {})), DataConfig(**raw.get("data", {})))
    return load_config(None)


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Meta-Fingerprint on NPZ windows or a dataset directory.")
    p.add_argument("--data", default=None, help="Directory with train/val/test.npz or one NPZ with split/patient_id.")
    p.add_argument("--train", default=None, help="Training NPZ; used when --data is not supplied.")
    p.add_argument("--val", default=None, help="Validation NPZ for --train mode.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="runs/metafingerprint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None: cfg.training.epochs = args.epochs
    if args.batch_size is not None: cfg.training.batch_size = args.batch_size
    if args.lr is not None: cfg.training.lr = args.lr
    if args.device is not None: cfg.training.device = args.device
    seed_everything(cfg.training.seed)
    if args.data:
        loaders = create_dataloaders(args.data, cfg.data, cfg.training)
        train_loader = loaders["train"]
        val_loader = loaders.get("val")
        train_ds = getattr(train_loader, "dataset", None)
        if hasattr(train_ds, "seq_len"): cfg.model.seq_len = train_ds.seq_len
    else:
        if not args.train:
            raise SystemExit("Supply either --data or --train.")
        train_ds = NPZWindowDataset(args.train, cfg.data.normalize_inputs, cfg.data.normalize_abp)
        val_ds = NPZWindowDataset(args.val, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=train_ds.stats) if args.val else None
        cfg.model.seq_len = train_ds.seq_len
        train_loader = make_loader(train_ds, cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
        val_loader = make_loader(val_ds, cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers) if val_ds is not None else None
    Trainer(MetaFingerprintModel(cfg.model), cfg, resolve_device(cfg.training.device), args.output).fit(train_loader, val_loader)

def evaluate_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--checkpoint", required=True); p.add_argument("--data", required=True); p.add_argument("--config", default=None); p.add_argument("--output", default=None); p.add_argument("--stats", default=None); p.add_argument("--batch-size", type=int, default=None); p.add_argument("--device", default="auto"); p.add_argument("--dtw", action="store_true")
    args = p.parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint, "cpu"); cfg = _config_from_checkpoint_or_file(ckpt, args.config)
    if args.batch_size is not None: cfg.training.batch_size = args.batch_size
    ds = NPZWindowDataset(args.data, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=_load_stats(args.stats)); cfg.model.seq_len = ds.seq_len
    loader = make_loader(ds, cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    model = MetaFingerprintModel(cfg.model); model.load_state_dict(ckpt["model"]); device = resolve_device(args.device); model.to(device)
    metrics = evaluate_model(model, loader, cfg.training, device, compute_dtw=args.dtw); print(metrics)
    if args.output: save_json(metrics, args.output)


def synthetic_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    p.add_argument("--output", default="data/synthetic"); p.add_argument("--patients", type=int, default=48); p.add_argument("--windows-per-patient", type=int, default=12); p.add_argument("--fs", type=float, default=125.0); p.add_argument("--seq-len", type=int, default=1250); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    paths = make_synthetic_dataset(args.output, args.patients, args.windows_per_patient, args.fs, args.seq_len, args.seed)
    print({k: str(v) for k, v in paths.items()})


def calibrate_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Adapt only the structural branch.")
    p.add_argument("--checkpoint", required=True); p.add_argument("--support", required=True); p.add_argument("--query", default=None); p.add_argument("--config", default=None); p.add_argument("--output", default="runs/patient_adapted.pt"); p.add_argument("--stats", default=None); p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint, "cpu"); cfg = _config_from_checkpoint_or_file(ckpt, args.config); device = resolve_device(args.device)
    support_ds = NPZWindowDataset(args.support, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=_load_stats(args.stats)); cfg.model.seq_len = support_ds.seq_len
    support_loader = make_loader(support_ds, min(cfg.training.batch_size, len(support_ds)), shuffle=True)
    model = MetaFingerprintModel(cfg.model); model.load_state_dict(ckpt["model"]); model.to(device)
    adapted = adapt_structural_branch(model, support_loader, cfg.training, device)
    metrics = {}
    if args.query:
        query_ds = NPZWindowDataset(args.query, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=support_ds.stats)
        metrics = evaluate_model(adapted, make_loader(query_ds, cfg.training.batch_size), cfg.training, device); print(metrics)
    save_checkpoint(args.output, adapted, config=cfg, metrics=metrics)


def predict_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Predict waveforms and phenotype logits.")
    p.add_argument("--checkpoint", required=True); p.add_argument("--data", required=True); p.add_argument("--output", default="predictions.npz"); p.add_argument("--config", default=None); p.add_argument("--stats", default=None); p.add_argument("--batch-size", type=int, default=None); p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    ckpt = load_checkpoint(args.checkpoint, "cpu"); cfg = _config_from_checkpoint_or_file(ckpt, args.config)
    if args.batch_size is not None: cfg.training.batch_size = args.batch_size
    ds = NPZWindowDataset(args.data, cfg.data.normalize_inputs, cfg.data.normalize_abp, stats=_load_stats(args.stats)); cfg.model.seq_len = ds.seq_len
    loader = make_loader(ds, cfg.training.batch_size); device = resolve_device(args.device); model = MetaFingerprintModel(cfg.model); model.load_state_dict(ckpt["model"]); model.to(device).eval()
    waves=[]; logits=[]; taus=[]
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device), sample=False)
            waves.append(out["waveform"].cpu().numpy()); logits.append(out["logits"].cpu().numpy()); taus.append(out["tau"].cpu().numpy())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, waveform=np.concatenate(waves), logits=np.concatenate(logits), delay_sec=np.concatenate(taus)); print(f"saved {args.output}")
