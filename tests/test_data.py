from pathlib import Path

import numpy as np

from metafingerprint.config import load_config
from metafingerprint.data import NPZWindowDataset, create_dataloaders


def test_npz_dataset_ecg_ppg_compat(tmp_path: Path):
    n, l = 10, 64
    path = tmp_path / "toy.npz"
    np.savez_compressed(
        path,
        ecg=np.random.randn(n, l).astype("float32"),
        ppg=np.random.randn(n, l).astype("float32"),
        abp=(90 + np.random.randn(n, l)).astype("float32"),
        label=np.arange(n) % 3,
        patient_id=np.arange(n) // 2,
        split=np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2),
    )
    ds = NPZWindowDataset(path, split="train")
    assert len(ds) == 6
    item = ds[0]
    assert item["x"].shape == (2, l)
    assert item["y"].shape == (l,)
    cfg = load_config("configs/debug.yaml")
    loaders = create_dataloaders(path, cfg.data, cfg.training)
    assert set(loaders) == {"train", "val", "test"}
