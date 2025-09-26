#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import pandas as pd


# ---- Configuration (edit these) ---------------------------------------------
META_DIR_NAME = "metadata"      # Name of the metadata directory in each sample
CSV_NAME      = "gene_log1p.csv"  # CSV filename inside the metadata directory


def is_visible_dir(path: str) -> bool:
    name = os.path.basename(path.rstrip(os.sep))
    return os.path.isdir(path) and not name.startswith('.') and not name.startswith('_')


def main() -> int:
    cwd = os.getcwd()
    entries = [os.path.join(cwd, d) for d in os.listdir(cwd)]
    subdirs = [p for p in entries if is_visible_dir(p)]

    if not subdirs:
        print("No subdirectories found to scan.")
        return 0

    loaded = 0
    missing = 0
    errors = 0

    for d in sorted(subdirs):
        sample = os.path.basename(d)
        csv_path = os.path.join(d, META_DIR_NAME, CSV_NAME)
        if not os.path.isfile(csv_path):
            print(f"[WARN] Missing CSV for '{sample}': {csv_path}")
            missing += 1
            continue

        try:
            df = pd.read_csv(csv_path)

            # normalize/clean
            df["tile"] = df["tiles"]
            # avoid writing an index column
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[ERROR] Failed reading '{csv_path}': {e}")
            errors += 1
            continue

        loaded += 1
        print(f"[OK] {sample}: '{CSV_NAME}' — {df.shape[0]} rows, {df.shape[1]} cols")

    print("\nSummary:")
    print(f"  Loaded:  {loaded}")
    print(f"  Missing: {missing}")
    print(f"  Errors:  {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

