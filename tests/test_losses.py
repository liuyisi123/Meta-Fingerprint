import torch

from metafingerprint.config import load_config
from metafingerprint.losses import compute_total_loss, temporal_consistency_loss
from metafingerprint.models import MetaFingerprintModel


def test_loss_backward():
    torch.set_num_threads(1)
    cfg = load_config("configs/debug.yaml")
    model = MetaFingerprintModel(cfg.model)
    batch = {
        "x": torch.randn(4, 2, cfg.model.seq_len),
        "y": torch.randn(4, cfg.model.seq_len) + 90,
        "label": torch.tensor([0, 1, 2, -1]),
        "abp_mask": torch.tensor([True, True, True, True]),
        "label_mask": torch.tensor([True, True, True, False]),
        "patient_id": torch.tensor([1, 1, 2, 3]),
    }
    out = model(batch["x"], sample=True)
    loss = compute_total_loss(out, batch, cfg.training, global_step=0)
    loss.total.backward()
    assert torch.isfinite(loss.total)


def test_temporal_consistency_zero_without_pairs():
    z = torch.randn(3, 8)
    pids = torch.tensor([1, 2, 3])
    loss = temporal_consistency_loss(z, pids, margin=0.9)
    assert loss.item() == 0.0
