#!/usr/bin/env python3
"""
Tune WMSE weights against a dataset and visualize the effect.

This script loads a WMSE weights CSV (as produced by LDS grid search), maps it
onto a dataset to measure per-gene average applied weight, then applies a
post-hoc calibration so that the dataset-level mean weight per gene is ~1. It
then writes a tuned CSV and optionally plots empirical vs original WMSE vs tuned
WMSE for selected genes.

Usage (typical):
  python tune_wmse_weights.py \
      --weights ./weights/wmse/coad/best_smoothings.csv \
      --dataset coad --split train --max-rows 500 \
      --out-csv ./weights/wmse/coad/best_smoothings_tuned.csv \
      --plot-out lds_plots_tuned --plot-n 8

Notes:
- Calibration uses the same mapping logic as training (via get_dataset with
  lds_smoothing_csv) to compute dataset-level means of the applied weights.
- The tuned weights are simply scaled: w_tuned = w / mean_dataset(w[idx]).
  This preserves the WMSE emphasis shape while normalizing its magnitude to a
  stable baseline across genes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def _load_weights_df(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"gene", "bins", "kernel_size", "sigma", "weights_json"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in CSV: {sorted(missing)}")
    # Drop any spurious rows with gene labels like 'Unnamed: 0'
    df = df[df["gene"].astype(str).str.startswith("Unnamed") == False].copy()
    return df


def _attach_and_measure_means(
    weights_csv: str | Path,
    dataset: str,
    split: str,
    *,
    max_rows: int | None,
    meta_data_dir: str | None,
    gene_data_filename: str | None,
    data_dir_override: str | None = None,
    samples_override: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Map weights to dataset and compute per-gene dataset-level mean weight.

    Returns (df_dataset, mean_by_gene).
    df_dataset is the STDataset.df used for mapping.
    """
    from script.configs.dataset_config import get_dataset_cfg
    from script.data_processing.data_loader import get_dataset

    cfg = get_dataset_cfg({"dataset": dataset, "debug": False})
    data_dir = data_dir_override or cfg["data_dir"]
    if samples_override is not None:
        samples = list(samples_override)
    elif split == "train":
        samples = cfg["train_samples"]
    elif split == "val":
        samples = cfg["val_samples"]
    elif split == "test":
        samples = cfg.get("test_samples") or cfg.get("test_samples_all")
        if not samples:
            raise RuntimeError("Dataset config does not define test_samples; provide --split train or val.")
    else:
        raise ValueError("split must be one of: train, val, test")

    dfw = _load_weights_df(weights_csv)
    genes = list(map(str, dfw["gene"].tolist()))

    # Hardcode dataset metadata layout for now
    gdf_name = "gene_log1p.csv"
    meta_dir = "metadata"

    ds = get_dataset(
        data_dir=data_dir,
        genes=genes,
        samples=samples,
        gene_data_filename=gdf_name,
        meta_data_dir=meta_dir,
        lds_smoothing_csv=str(weights_csv),
        max_len=max_rows,
    )
    df = ds.df

    means: Dict[str, float] = {}
    for g in genes:
        wcol = f"{g}_lds_w"
        if wcol not in df.columns:
            continue
        w = pd.to_numeric(df[wcol], errors="coerce").astype(float)
        # Ignore NaN/inf rows when averaging
        w = w.replace([float("inf"), -float("inf")], np.nan).dropna()
        if w.shape[0] == 0:
            continue
        means[g] = float(w.mean())

    return df, means


def _write_tuned_csv(dfw: pd.DataFrame, mean_by_gene: Dict[str, float], out_csv: str | Path) -> pd.DataFrame:
    out_df = dfw.copy()
    tuned_json: List[str] = []
    orig_json: List[str] = []
    scales: List[float] = []
    eps = 1e-12
    for _, r in out_df.iterrows():
        g = str(r["gene"])
        w = np.asarray(json.loads(r["weights_json"]), dtype=float)
        mu = mean_by_gene.get(g, None)
        if mu is None or not np.isfinite(mu) or mu <= 0:
            tuned_w = w.copy()  # leave unchanged if we cannot measure a mean
            scale = 1.0
        else:
            scale = 1.0 / max(mu, eps)
            tuned_w = w * scale
        tuned_json.append(json.dumps([float(x) for x in tuned_w.tolist()]))
        orig_json.append(json.dumps([float(x) for x in w.tolist()]))
        scales.append(float(scale))

    # Keep a copy of originals for reference, but overwrite weights_json for drop-in use.
    out_df["weights_json_orig"] = orig_json
    out_df["weights_json"] = tuned_json
    out_df["calibration_scale"] = scales
    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def _plot_overlays(
    dfw: pd.DataFrame,
    dataset: str,
    *,
    meta_data_dir: str | None,
    gene_data_filename: str | None,
    plot_out: str | Path,
    n: int,
    data_dir_override: str | None = None,
    samples_override: List[str] | None = None,
):
    """Plot empirical vs original WMSE vs tuned WMSE for up to n genes."""
    from script.configs.dataset_config import get_dataset_cfg
    from script.data_processing.data_loader import label_dataset
    from script.data_processing.lds import LDS, LDSParams

    cfg = get_dataset_cfg({"dataset": dataset, "debug": False})
    data_dir = data_dir_override or cfg["data_dir"]
    samples = samples_override or cfg["train_samples"]
    # Hardcode dataset metadata layout for now
    ds = label_dataset(
        data_dir, samples,
        "gene_log1p.csv",
        meta_data_dir="metadata",
    )
    lds = LDS(kernel_type="gaussian", dataset=ds)
    out_dir = Path(plot_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    head = dfw.head(max(0, int(n)))
    for _, r in head.iterrows():
        g = str(r["gene"])
        bins = int(r["bins"])
        ks = int(r["kernel_size"])
        sigma = float(r["sigma"])
        # tuned_df stores tuned weights in 'weights_json' and originals in 'weights_json_orig'
        if "weights_json_orig" in r:
            w_orig = np.asarray(json.loads(r["weights_json_orig"]), dtype=float)
            w_tuned = np.asarray(json.loads(r["weights_json"]), dtype=float)
        else:
            # Fallback: no tuning performed, plot original twice
            w_orig = np.asarray(json.loads(r["weights_json"]), dtype=float)
            w_tuned = w_orig.copy()

        # Empirical histogram and bin centres
        emp_hist, edges = lds._empirical(g, bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        # Normalize to probabilities for visualization
        eps = 1e-12
        p = emp_hist.astype(float)
        p = p / max(p.sum(), eps)
        q1 = w_orig.astype(float) / max(float(np.sum(w_orig)), eps)
        q2 = w_tuned.astype(float) / max(float(np.sum(w_tuned)), eps)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.step(centres, p, where="mid", label="empirical")
        plt.step(centres, q1, where="mid", label="wmse")
        plt.step(centres, q2, where="mid", label="wmse_tuned")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.title(f"{g} (bins={bins}, ks={ks}, sigma={sigma})")
        plt.legend()
        plt.savefig(out_dir / f"{g}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[OK] wrote overlay plots to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Tune WMSE weights against a dataset and plot overlays.")
    ap.add_argument("--weights", required=True, help="Path to best_smoothings.csv")
    ap.add_argument("--dataset", required=True, help="Dataset key (e.g., coad)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split for calibration")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows when building dataset")
    ap.add_argument("--meta-data-dir", default=None, help="Override meta_data_dir token (optional)")
    ap.add_argument("--gene-data-filename", default=None, help="Override gene_data_filename (optional)")
    ap.add_argument("--data-dir", default=None, help="Override dataset root directory (optional)")
    ap.add_argument("--samples", nargs="*", default=None, help="Optional explicit list of sample IDs to use for split")
    ap.add_argument("--out-csv", default=None, help="Where to write tuned CSV (default adds _tuned.csv next to input)")
    ap.add_argument("--plot-out", default=None, help="Directory to write empirical vs wmse vs tuned plots")
    ap.add_argument("--plot-n", type=int, default=0, help="Number of genes to plot")
    args = ap.parse_args()

    weights_csv = args.weights
    try:
        dfw = _load_weights_df(weights_csv)
    except Exception as e:
        _eprint(f"[ERROR] failed to load weights CSV: {e}")
        sys.exit(2)

    print(f"[INFO] Calibrating against dataset: dataset={args.dataset}, split={args.split}")
    try:
        df_ds, means = _attach_and_measure_means(
            weights_csv, args.dataset, args.split,
            max_rows=args.max_rows,
            meta_data_dir=args.meta_data_dir,
            gene_data_filename=args.gene_data_filename,
            data_dir_override=args.data_dir,
            samples_override=args.samples,
        )
    except Exception as e:
        _eprint(f"[ERROR] dataset mapping failed: {e}")
        sys.exit(3)

    # Build output CSV path
    out_csv = args.out_csv
    if not out_csv:
        p = Path(weights_csv)
        out_csv = str(p.with_name(p.stem + "_tuned.csv"))

    tuned_df = _write_tuned_csv(dfw, means, out_csv)
    print(f"[OK] wrote tuned weights CSV: {out_csv}")

    # Optional plots
    if args.plot_out and args.plot_n > 0:
        try:
            _plot_overlays(tuned_df, args.dataset,
                           meta_data_dir=args.meta_data_dir,
                           gene_data_filename=args.gene_data_filename,
                           plot_out=args.plot_out,
                           n=args.plot_n,
                           data_dir_override=args.data_dir,
                           samples_override=args.samples)
        except Exception as e:
            _eprint(f"[WARN] plot generation failed: {e}")

    print("[DONE] WMSE tuning completed.")


if __name__ == "__main__":
    main()
