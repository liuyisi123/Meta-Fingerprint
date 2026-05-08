import torch

from metafingerprint.config import load_config
from metafingerprint.models import MetaFingerprintModel


def test_forward_shapes():
    torch.set_num_threads(1)
    cfg = load_config("configs/debug.yaml")
    model = MetaFingerprintModel(cfg.model)
    x = torch.randn(3, 2, cfg.model.seq_len)
    out = model(x, sample=False)
    assert out["H"].shape == (3, cfg.model.seq_len, cfg.model.encoder_dim)
    assert out["tau"].shape == (3, cfg.model.seq_len)
    assert out["waveform"].shape == (3, cfg.model.seq_len)
    assert out["logits"].shape == (3, cfg.model.num_classes)
    assert torch.all(out["tau"] >= cfg.model.tau_min)
    assert torch.all(out["tau"] <= cfg.model.tau_max)
