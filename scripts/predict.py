#!/usr/bin/env python
"""Save ABP waveform predictions, phenotype logits, and learned delays."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.cli import predict_main

if __name__ == "__main__":
    predict_main()
