import torch
torch.set_num_threads(1)
from pathlib import Path

from metafingerprint.config import load_config
from metafingerprint.data import NPZWindowDataset, make_loader, make_synthetic_dataset
from metafingerprint.models import MetaFingerprintModel
from metafingerprint.train import Trainer
from metafingerprint.utils import resolve_device, seed_everything


def test_one_epoch_training(tmp_path):
    paths = make_synthetic_dataset(tmp_path / "data", n_patients=6, windows_per_patient=1, seq_len=24, seed=1)
    cfg = load_config(None)
    cfg.model.seq_len = 24
    cfg.model.local_dim = 8
    cfg.model.encoder_dim = 16
    cfg.model.ode_hidden_dim = 16
    cfg.model.ode_steps = 1
    cfg.model.ssm_hidden_dim = 16
    cfg.model.tau_min = 0.02
    cfg.model.tau_max = 0.08
    cfg.model.z_id_dim = 8
    cfg.model.z_bp_dim = 8
    cfg.model.decoder_channels = 16
    cfg.model.decoder_base_len = 8
    cfg.model.use_spectral_norm = False
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.num_workers = 0
    cfg.training.device = "cpu"
    cfg.training.dis_warmup_steps = 1
    seed_everything(0)
    train_ds = NPZWindowDataset(paths["train"])
    val_ds = NPZWindowDataset(paths["val"], stats=train_ds.stats)
    trainer = Trainer(MetaFingerprintModel(cfg.model), cfg, resolve_device("cpu"), tmp_path / "run")
    trainer.fit(make_loader(train_ds, 4, shuffle=True), make_loader(val_ds, 4))
    assert Path(tmp_path / "run" / "best.pt").exists()
