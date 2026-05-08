"""Configuration dataclasses and YAML helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints

import yaml


@dataclass
class ModelConfig:
    seq_len: int = 1250
    fs: float = 125.0
    input_channels: int = 2
    local_dim: int = 64
    encoder_dim: int = 128
    ode_hidden_dim: int = 96
    ode_steps: int = 2
    ssm_hidden_dim: int = 128
    tau_min: float = 0.10
    tau_max: float = 0.40
    z_id_dim: int = 64
    z_bp_dim: int = 64
    decoder_channels: int = 128
    decoder_base_len: int = 64
    num_classes: int = 3
    dropout: float = 0.10
    use_spectral_norm: bool = True


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    min_lr: float = 1.0e-6
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    device: str = "auto"
    amp: bool = False
    lambda_wave: float = 1.0
    lambda_pheno: float = 0.3
    lambda_dis: float = 0.1
    dis_warmup_steps: int = 10_000
    huber_delta: float = 5.0
    alpha_mi: float = 1.0
    beta_tc: float = 4.0
    gamma_mkl: float = 1.0
    lambda_consistency: float = 0.5
    consistency_margin: float = 0.9
    meta: bool = False
    meta_frequency: int = 20
    tasks_per_batch: int = 4
    support_size: int = 4
    query_size: int = 4
    inner_steps: int = 3
    inner_lr: float = 1.0e-2
    lambda_reg: float = 0.1
    first_order: bool = True
    monitor: str = "val_rmse"
    patience: int = 7
    save_every: int = 0
    verbose: bool = True


@dataclass
class DataConfig:
    target_fs: float = 125.0
    window_sec: float = 10.0
    normalize_inputs: bool = True
    normalize_abp: bool = False
    train_fraction: float = 0.60
    val_fraction: float = 0.20
    test_fraction: float = 0.20


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


T = TypeVar("T")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _construct_dataclass(cls: Type[T], values: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        target = hints.get(f.name, f.type)
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[f.name] = _construct_dataclass(target, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    default = asdict(ExperimentConfig())
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
    merged = _deep_update(default, raw)
    return ExperimentConfig(
        model=_construct_dataclass(ModelConfig, merged.get("model", {})),
        training=_construct_dataclass(TrainingConfig, merged.get("training", {})),
        data=_construct_dataclass(DataConfig, merged.get("data", {})),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=True)
