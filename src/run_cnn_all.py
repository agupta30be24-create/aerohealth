"""Train the 1D-CNN on FD001-FD004 (several seeds) and write results/all_datasets_cnn_runs.csv.

Runs as its own process on purpose: XGBoost, scikit-learn and PyTorch each bundle their own OpenMP runtime on macOS,
and training XGBoost then PyTorch in one process segfaults. Keeping deep models in a separate process mirrors how
02_baselines (trees) and 03_deep_models (deep) are already split. This script never imports XGBoost.

Usage (from repo root or anywhere):  python src/run_cnn_all.py [seed ...]   (default seeds: 42 1 2)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import deep  # imports torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def nasa_score(y, p):
    e = p - y
    return float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)))


def main(seeds):
    out = ROOT / "results" / "all_datasets_cnn_runs.csv"
    rows = []
    for fd in DATASETS:
        d = dict(np.load(ROOT / "data" / "processed" / f"{fd.lower()}_windows.npz"))
        engines = d.get("test_engines", np.arange(1, len(d["y_test"]) + 1))
        pad = np.isin(engines, d.get("padded_engines", []))
        for s in seeds:
            v, p, secs = deep.train_cnn(d, s)
            rows.append({"dataset": fd, "model": "1D-CNN", "seed": s, "val_RMSE": v,
                         "test_RMSE": rmse(d["y_test"], p), "test_NASA": nasa_score(d["y_test"], p),
                         "NASA_per_engine": nasa_score(d["y_test"], p) / len(p),
                         "RMSE_padded_engines": rmse(d["y_test"][pad], p[pad]) if pad.any() else np.nan,
                         "time_s": secs})
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{fd} seed {s}: test RMSE {rows[-1]['test_RMSE']:.2f} ({secs:.0f}s)", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [42, 1, 2])
