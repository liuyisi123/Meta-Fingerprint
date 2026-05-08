#!/usr/bin/env python
"""Generate a synthetic ECG/PPG/ABP dataset for smoke tests."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.cli import synthetic_main

if __name__ == "__main__":
    synthetic_main()
