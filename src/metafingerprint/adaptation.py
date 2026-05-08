"""Structural-subspace calibration and FO-MAML helper."""
from __future__ import annotations

import copy
from typing import Any

import torch

from .losses import compute_total_loss, masked_huber_loss, structural_anchor_loss
from .utils import move_to_device


def freeze_except_structural(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.structural_encoder.parameters():
        p.requires_grad_(True)


def adapt_structural_branch(model: torch.nn.Module, support_loader: Any, training_cfg: Any, device: torch.device, steps: int | None = None, lr: float | None = None) -> torch.nn.Module:
    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    freeze_except_structural(adapted)
    ref = {n: p.detach().clone() for n, p in adapted.structural_encoder.named_parameters()}
    opt = torch.optim.SGD(adapted.structural_encoder.parameters(), lr=lr or training_cfg.inner_lr)
    for _ in range(int(steps or training_cfg.inner_steps)):
        for batch in support_loader:
            batch = move_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            out = adapted(batch["x"], sample=True)
            wave = masked_huber_loss(out["waveform"], batch["y"], batch.get("abp_mask"), training_cfg.huber_delta)
            loss = wave + training_cfg.lambda_reg * structural_anchor_loss(adapted.structural_encoder, ref)
            loss.backward(); opt.step()
    adapted.eval()
    return adapted


def first_order_meta_step(model: torch.nn.Module, tasks: list[dict[str, dict[str, Any]]], optimizer: torch.optim.Optimizer, training_cfg: Any, device: torch.device, global_step: int) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    accum: dict[str, torch.Tensor] = {}
    losses: list[float] = []
    for task in tasks:
        fast = copy.deepcopy(model).to(device)
        freeze_except_structural(fast)
        ref = {n: p.detach().clone() for n, p in fast.structural_encoder.named_parameters()}
        inner = torch.optim.SGD(fast.structural_encoder.parameters(), lr=training_cfg.inner_lr)
        for _ in range(training_cfg.inner_steps):
            support = move_to_device(task["support"], device)
            inner.zero_grad(set_to_none=True)
            s_out = fast(support["x"], sample=True)
            wave = masked_huber_loss(s_out["waveform"], support["y"], support.get("abp_mask"), training_cfg.huber_delta)
            (wave + training_cfg.lambda_reg * structural_anchor_loss(fast.structural_encoder, ref)).backward()
            inner.step()
        for p in fast.parameters():
            p.requires_grad_(True)
        query = move_to_device(task["query"], device)
        q_out = fast(query["x"], sample=True)
        q_loss = compute_total_loss(q_out, query, training_cfg, global_step)
        q_loss.total.backward(); losses.append(float(q_loss.total.detach().cpu()))
        for name, p_fast in fast.named_parameters():
            if p_fast.grad is not None:
                accum[name] = accum.get(name, torch.zeros_like(p_fast.grad.detach())) + p_fast.grad.detach()
    named = dict(model.named_parameters())
    for name, grad in accum.items():
        if name in named and named[name].requires_grad:
            named[name].grad = grad.to(named[name].device) / max(len(tasks), 1)
    if training_cfg.grad_clip_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip_norm)
    optimizer.step()
    return {"meta_loss": float(sum(losses) / max(len(losses), 1))}
