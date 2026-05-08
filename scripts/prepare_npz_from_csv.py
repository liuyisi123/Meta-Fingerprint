#!/usr/bin/env python
"""Convert synchronized CSV samples to Meta-Fingerprint NPZ windows.

Required CSV columns: ecg, ppg. Optional: abp, patient_id, label, domain.
Rows are samples from synchronized streams. If patient_id exists, windows do not
cross patient boundaries.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metafingerprint.data.preprocessing import (
    basic_signal_quality,
    make_windows,
    resample_signal,
    save_npz_windows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-fs", type=float, default=125.0)
    p.add_argument("--target-fs", type=float, default=125.0)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable the default finite/flat-line/pressure-range window rejection.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    for col in ["ecg", "ppg"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required CSV column: {col}")
    groups = df.groupby("patient_id") if "patient_id" in df.columns else [(0, df)]
    signals_all, abp_all, labels_all, pids_all, domains_all = [], [], [], [], []
    for pid, g in groups:
        ecg = resample_signal(g["ecg"].to_numpy(float), args.source_fs, args.target_fs)
        ppg = resample_signal(g["ppg"].to_numpy(float), args.source_fs, args.target_fs)
        abp = resample_signal(g["abp"].to_numpy(float), args.source_fs, args.target_fs) if "abp" in g.columns else None
        signals, y = make_windows(ecg, ppg, abp, fs=args.target_fs, window_sec=args.window_sec)
        keep = np.ones(signals.shape[0], dtype=bool)
        if not args.no_quality_filter:
            keep = basic_signal_quality(signals, y)
        signals = signals[keep]
        if y is not None:
            y = y[keep]
        if signals.shape[0] == 0:
            continue
        signals_all.append(signals)
        pids_all.extend([pid] * signals.shape[0])
        if y is not None:
            abp_all.append(y)
        if "label" in g.columns:
            length = int(round(args.target_fs * args.window_sec))
            labels = g["label"].to_numpy(int)[length - 1 :: length]
            labels_all.extend(labels[: len(keep)][keep].tolist())
        if "domain" in g.columns:
            domains_all.extend([str(g["domain"].iloc[0])] * signals.shape[0])
    if not signals_all:
        raise SystemExit("No valid windows remained after preprocessing.")
    save_npz_windows(
        args.out,
        np.concatenate(signals_all, axis=0),
        np.concatenate(abp_all, axis=0) if abp_all else None,
        np.asarray(labels_all, dtype=np.int64) if labels_all else None,
        pids_all,
        domains_all if domains_all else None,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
