#!/usr/bin/env python
"""Generate synthetic data and run a one-epoch smoke test."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.data import make_synthetic_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "runs" / "smoke"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(out / "data", n_patients=6, windows_per_patient=1, fs=6.4, seq_len=64, seed=42)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--data",
        str(out / "data"),
        "--config",
        str(ROOT / "configs" / "debug.yaml"),
        "--output",
        str(out / "run"),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch-size",
        "4",
    ]
    print("Running:", " ".join(cmd))
    import os
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    env.setdefault("TORCH_INTEROP_THREADS", "1")
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
