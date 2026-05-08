#!/usr/bin/env python
"""Render a simple review-safe metric bar plot from an aggregate CSV.

Expected CSV columns:
    method, setting, metric, value

Example:
    python scripts/figures/plot_metric_bar.py \
      --csv artifacts/setting_a_metrics_long.csv \
      --metric rmse \
      --setting Setting-A \
      --out figures/setting_a_rmse.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--metric", required=True)
    p.add_argument("--setting", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    required = {"method", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    data = df[df["metric"].astype(str) == args.metric].copy()
    if args.setting is not None:
        if "setting" not in data.columns:
            raise SystemExit("--setting requires a 'setting' column")
        data = data[data["setting"].astype(str) == args.setting]
    if data.empty:
        raise SystemExit("No rows matched the requested metric/setting.")
    if "seed" in data.columns:
        data = data.groupby("method", as_index=False)["value"].mean()
    data = data.sort_values("value", ascending=True)
    fig, ax = plt.subplots(figsize=(4.8, max(1.8, 0.32 * len(data))))
    ax.barh(data["method"], data["value"], color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.set_xlabel(args.metric)
    ax.set_ylabel("")
    ax.set_title(args.title or (f"{args.setting or 'All settings'}: {args.metric}"))
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
