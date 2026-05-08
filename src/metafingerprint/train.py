"""Training loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adaptation import first_order_meta_step
from .config import ExperimentConfig, save_config
from .data import EpisodeSampler
from .evaluation import evaluate_model
from .losses import compute_total_loss
from .utils import count_parameters, move_to_device, save_checkpoint, save_json


class Trainer:
    def __init__(self, model: torch.nn.Module, config: ExperimentConfig, device: torch.device, output_dir: str | Path) -> None:
        self.model = model.to(device); self.config = config; self.device = device; self.output_dir = Path(output_dir); self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, config.training.epochs), eta_min=config.training.min_lr)
        self.global_step = 0; self.best_score = float("inf"); self.bad_epochs = 0
        save_config(config, self.output_dir / "config.yaml")

    def train_one_epoch(self, train_loader: DataLoader, epoch: int, episode_sampler: EpisodeSampler | None = None) -> dict[str, float]:
        self.model.train(); totals=[]; waves=[]; phenos=[]; dises=[]; metas=[]
        verbose = getattr(self.config.training, "verbose", True)
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False, disable=not verbose):
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(batch["x"], sample=True)
            loss = compute_total_loss(out, batch, self.config.training, self.global_step)
            loss.total.backward()
            if self.config.training.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
            self.optimizer.step()
            totals.append(float(loss.total.detach().cpu())); waves.append(float(loss.wave.detach().cpu())); phenos.append(float(loss.pheno.detach().cpu())); dises.append(float(loss.dis.detach().cpu()))
            self.global_step += 1
            if self.config.training.meta and episode_sampler is not None and self.global_step % max(1, self.config.training.meta_frequency) == 0:
                metas.append(first_order_meta_step(self.model, episode_sampler.sample(), self.optimizer, self.config.training, self.device, self.global_step)["meta_loss"])
        out = {"train_loss": float(np.mean(totals)), "train_wave": float(np.mean(waves)), "train_pheno": float(np.mean(phenos)), "train_dis": float(np.mean(dises))}
        if metas:
            out["train_meta_loss"] = float(np.mean(metas))
        return out

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        train_ds = getattr(train_loader, "dataset", None)
        if hasattr(train_ds, "stats"):
            save_json(train_ds.stats.to_dict(), self.output_dir / "train_stats.json")
        sampler = None
        if self.config.training.meta and hasattr(train_ds, "indices_by_patient"):
            sampler = EpisodeSampler(train_ds, self.config.training.tasks_per_batch, self.config.training.support_size, self.config.training.query_size, self.config.training.seed)
        history=[]; summary={"num_parameters": count_parameters(self.model, trainable_only=False)}
        for epoch in range(1, self.config.training.epochs + 1):
            row = self.train_one_epoch(train_loader, epoch, sampler)
            self.scheduler.step()
            if val_loader is not None:
                val = evaluate_model(self.model, val_loader, self.config.training, self.device, compute_dtw=False)
                row.update({f"val_{k}": v for k, v in val.items()})
                score = float(row.get(self.config.training.monitor, row.get("val_rmse", row["val_loss"])))
            else:
                score = float(row["train_wave"])
            row["epoch"] = epoch; history.append(row); save_json({"history": history, "summary": summary}, self.output_dir / "history.json")
            if score < self.best_score:
                self.best_score = score; self.bad_epochs = 0
                save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, self.scheduler, self.config, epoch, row)
            else:
                self.bad_epochs += 1
            save_checkpoint(self.output_dir / "last.pt", self.model, self.optimizer, self.scheduler, self.config, epoch, row)
            if self.config.training.save_every and epoch % self.config.training.save_every == 0:
                save_checkpoint(self.output_dir / f"epoch_{epoch:03d}.pt", self.model, self.optimizer, self.scheduler, self.config, epoch, row)
            print(row)
            if self.bad_epochs >= self.config.training.patience:
                print(f"Early stopping after {epoch} epochs. Best {self.config.training.monitor}={self.best_score:.6f}"); break
        summary["best_score"] = self.best_score
        return {"history": history, "summary": summary}
