#!/usr/bin/env python3
"""
WMSE weights tester

Quick sanity checks for a WMSE weights CSV and (optionally) dataset mapping.

Usage:
  python wmse_tester.py --weights path/to/best_smoothings.csv

Optional dataset checks (requires dataset present locally):
  python wmse_tester.py --weights path/to/best_smoothings.csv \
                        --dataset coad [--split train] [--max-rows 500]

Optional overlay plots for a few genes:
  python wmse_tester.py --weights path/to/best_smoothings.csv \
                        --dataset coad --plot-out lds_plots_verify --plot-n 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import numpy as np


def _eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def check_csv(weights_csv: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(weights_csv)
    required = {"gene", "bins", "kernel_size", "sigma", "weights_json"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in CSV: {sorted(missing)}")

    bad: List[str] = []
    for _, r in df.iterrows():
        gene = str(r["gene"]) if "gene" in r else "?"
        try:
            w = torch.tensor(json.loads(r["weights_json"]))
        except Exception as e:
            _eprint(f"[ERROR] {gene}: failed to parse weights_json: {e}")
            bad.append(gene)
            continue
        bins = int(r["bins"]) if pd.notna(r["bins"]) else None
        if w.ndim != 1 or (bins is not None and len(w) != bins):
            _eprint(f"[ERROR] {gene}: weights length {w.numel()} != bins {bins}")
            bad.append(gene)
            continue
        if not torch.isfinite(w).all():
            _eprint(f"[ERROR] {gene}: weights contain non-finite values")
            bad.append(gene)
            continue
        if (w < 0).any():
            _eprint(f"[ERROR] {gene}: weights contain negative values")
            bad.append(gene)
            continue
        if float(w.sum()) <= 0:
            _eprint(f"[ERROR] {gene}: weights sum to zero")
            bad.append(gene)
            continue
    return df, bad


def map_to_dataset_and_summarize(
    df_weights: pd.DataFrame,
    dataset_name: str,
    split: str,
    max_rows: int | None,
    weight_csv_path: str,
    meta_data_dir: str | None = None,
    gene_data_filename: str | None = None,
) -> Tuple[List[Tuple[str, float, float, float, float]], List[Tuple[str, str]]]:
    """Attach weights to a dataset and compute per-gene stats.

    Returns (stats, problems):
      stats: list of (gene, mean, min, max, nonzero_frac)
      problems: list of (gene, reason)
    """
    # Lazy import project modules so the script can run CSV-only checks without project deps
    from script.configs.dataset_config import get_dataset_cfg
    from script.data_processing.data_loader import get_dataset

    cfg = get_dataset_cfg({"dataset": dataset_name, "debug": False})
    data_dir = cfg["data_dir"]
    if split == "train":
        samples = cfg["train_samples"]
    elif split == "val":
        samples = cfg["val_samples"]
    elif split == "test":
        samples = cfg.get("test_samples") or cfg.get("test_samples_all")
        if not samples:
            raise RuntimeError("Dataset config does not define test_samples; provide --split train or val.")
    else:
        raise ValueError("split must be one of: train, val, test")

    gdf_name = gene_data_filename or cfg.get("gene_data_filename", "gene_data.csv")
    meta_dir = meta_data_dir or cfg.get("meta_data_dir", "/meta_data/")

    # Use genes from the weights CSV to avoid ambiguity and satisfy lds_smoothing_csv requirements
    genes_csv = list(map(str, df_weights["gene"].tolist()))

    ds = get_dataset(
        data_dir=data_dir,
        genes=genes_csv,
        samples=samples,
        gene_data_filename=gdf_name,
        meta_data_dir=meta_dir,
        lds_smoothing_csv=weight_csv_path,
        max_len=max_rows,
    )
    df = ds.df
    want_genes = set(genes_csv)
    # Report any genes present in weights CSV but not in the dataset frame
    missing_in_data = [g for g in want_genes if g not in df.columns]
    if missing_in_data:
        _eprint(f"[WARN] {len(missing_in_data)} genes from CSV not found in dataset columns (first 10): {missing_in_data[:10]}")

    stats = []
    problems: List[Tuple[str, str]] = []
    for g in sorted(want_genes):
        if g not in df.columns:
            problems.append((g, "missing_gene_column"))
            continue
        wcol = f"{g}_lds_w"
        if wcol not in df.columns:
            problems.append((g, "missing_weight_col"))
            continue
        w = pd.to_numeric(df[wcol], errors="coerce").astype(float)
        if not w.replace([float("inf"), -float("inf")], pd.NA).dropna().shape[0]:
            problems.append((g, "non_finite"))
            continue
        mu = float(w.mean()); mn = float(w.min()); mx = float(w.max()); nz = float((w > 0).mean())
        stats.append((g, mu, mn, mx, nz))
        if mu < 0.5 or mu > 2.0:
            problems.append((g, f"mean_out_of_range:{mu:.3f}"))
        if mn < 0:
            problems.append((g, f"negative_min:{mn:.3f}"))
    return stats, problems


def maybe_generate_plots(
    df_weights: pd.DataFrame,
    dataset_name: str,
    out_dir: str,
    n: int,
    meta_data_dir: str | None = None,
    gene_data_filename: str | None = None,
) -> None:
    from script.configs.dataset_config import get_dataset_cfg
    from script.data_processing.data_loader import label_dataset
    from script.data_processing.lds import LDS, LDSParams

    cfg = get_dataset_cfg({"dataset": dataset_name, "debug": False})
    ds = label_dataset(
        cfg["data_dir"], cfg["train_samples"],
        gene_data_filename or cfg.get("gene_data_filename", "gene_data.csv"),
        meta_data_dir=meta_data_dir or cfg.get("meta_data_dir", "/meta_data/"),
    )
    lds = LDS(kernel_type="gaussian", dataset=ds)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    for _, r in df_weights.head(max(0, int(n))).iterrows():
        g = str(r["gene"]); bins = int(r["bins"]); ks = int(r["kernel_size"]); sigma = float(r["sigma"])
        w = np.asarray(json.loads(r["weights_json"]), dtype=float)
        lds.compare_plot(g, LDSParams(bins=bins, kernel_size=ks, sigma=sigma), w, out)
    print(f"[OK] wrote overlay plots to {out}")


def main():
    ap = argparse.ArgumentParser(description="WMSE weights tester")
    ap.add_argument("--weights", required=True, help="Path to best_smoothings.csv")
    # Optional dataset checks
    ap.add_argument("--dataset", help="Dataset key to validate mapping using dataset config")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split for mapping")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows when building dataset")
    ap.add_argument("--meta-data-dir", default=None, help="Override meta_data_dir token (optional)")
    ap.add_argument("--gene-data-filename", default=None, help="Override gene_data_filename (optional)")
    # Optional plots
    ap.add_argument("--plot-out", default=None, help="Directory to write empirical vs smoothed plots")
    ap.add_argument("--plot-n", type=int, default=0, help="Number of genes to plot (requires --dataset)")
    ap.add_argument("--fail-on-warn", action="store_true", help="Return non-zero exit if any issues are found")
    args = ap.parse_args()

    weights_csv = args.weights
    if not os.path.isfile(weights_csv):
        _eprint(f"[ERROR] weights CSV not found: {weights_csv}")
        sys.exit(2)

    print(f"[INFO] Checking CSV structure: {weights_csv}")
    dfw, bad = check_csv(weights_csv)
    n_genes = dfw.shape[0]
    print(f"[OK] CSV parsed. genes={n_genes}, invalid_rows={len(bad)}")
    if bad:
        _eprint(f"[WARN] invalid entries for genes (first 10): {bad[:10]}")

    any_warn = bool(bad)

    if args.dataset:
        print(f"[INFO] Checking dataset mapping: dataset={args.dataset}, split={args.split}")
        try:
            stats, problems = map_to_dataset_and_summarize(
                dfw, args.dataset, args.split, args.max_rows, weights_csv,
                meta_data_dir=args.meta_data_dir, gene_data_filename=args.gene_data_filename,
            )
        except Exception as e:
            _eprint(f"[ERROR] dataset mapping failed: {e}")
            sys.exit(3)
        # Print a compact summary
        print(f"[OK] Attached weights. genes_with_stats={len(stats)}, issues={len(problems)}")
        if stats:
            head = stats[: min(10, len(stats))]
            print("gene           mean    min     max     nonzero")
            for g, mu, mn, mx, nz in head:
                print(f"{g[:12]:12s}  {mu:6.3f}  {mn:6.3f}  {mx:6.3f}  {nz:7.2%}")
            if len(stats) > len(head):
                print(f"… ({len(stats)-len(head)} more)")
        if problems:
            any_warn = True
            _eprint(f"[WARN] problems found (first 10): {problems[:10]}")

        if args.plot_out and args.plot_n > 0:
            try:
                maybe_generate_plots(
                    dfw, args.dataset, args.plot_out, args.plot_n,
                    meta_data_dir=args.meta_data_dir, gene_data_filename=args.gene_data_filename,
                )
            except Exception as e:
                any_warn = True
                _eprint(f"[WARN] plot generation failed: {e}")

    if any_warn and args.fail_on_warn:
        sys.exit(1)
    print("[DONE] WMSE weights check completed.")


if __name__ == "__main__":
    main()
