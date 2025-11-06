import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, "..")

import re
import yaml
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from script.main_utils import parse_args, parse_yaml_config, setup_dump_env


EVAL_ROOT = "../evaluation"
OUT_DIR = os.path.join(EVAL_ROOT, "boxplots")


def _sanitize_token(s: str) -> str:
    return (
        str(s)
        .replace("\\", "/")
        .rstrip("/")
        .replace("/", "__")
        .replace(" ", "_")
    )[:128]


def _require_keys(cfg: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"missing required config keys: {missing}")


def _load_all_forward_metrics(eval_root: str) -> pd.DataFrame:
    if not os.path.isdir(eval_root):
        raise FileNotFoundError(f"eval_root not found or not a dir: {eval_root}")
    frames: List[pd.DataFrame] = []
    for name in sorted(os.listdir(eval_root)):
        p = os.path.join(eval_root, name)
        if not os.path.isdir(p):
            continue
        csvp = os.path.join(p, "forward_metrics.csv")
        if os.path.isfile(csvp):
            df = pd.read_csv(csvp)
            df["_encoder_dir"] = name
            frames.append(df)
    if not frames:
        raise RuntimeError("no forward_metrics.csv found under ../evaluation/*")
    df = pd.concat(frames, ignore_index=True)
    if "encoder_type" not in df.columns:
        raise RuntimeError("forward_metrics.csv missing 'encoder_type' column")
    if "run_name" not in df.columns:
        raise RuntimeError("forward_metrics.csv missing 'run_name' column")
    return df


def _apply_filters(
    df: pd.DataFrame,
    include_projects: List[str] | None,
    include_encoders: List[str] | None,
    include_run_name_regex: str | None,
    exclude_run_name_regex: str | None,
) -> pd.DataFrame:
    out = df
    if include_projects:
        out = out[out["project"].astype(str).isin([str(x) for x in include_projects])]
    if include_encoders:
        out = out[out["encoder_type"].astype(str).isin([str(x) for x in include_encoders])]
    if include_run_name_regex:
        rx = re.compile(include_run_name_regex)
        out = out[out["run_name"].astype(str).map(lambda s: bool(rx.search(s)))]
    if exclude_run_name_regex:
        rx = re.compile(exclude_run_name_regex)
        out = out[~out["run_name"].astype(str).map(lambda s: bool(rx.search(s)))]
    if out.empty:
        raise RuntimeError("no rows left after filters")
    return out


def _collect_values_by_encoder(df: pd.DataFrame, genes: List[str], skip_non_finite: bool) -> Dict[str, List[float]]:
    cols = [f"pearson_{g}" for g in genes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing pearson columns for genes: {missing}")
    values: Dict[str, List[float]] = {}
    for enc, gdf in df.groupby("encoder_type"):
        per_cols: List[np.ndarray] = []
        for c in cols:
            v = pd.to_numeric(gdf[c], errors="coerce").to_numpy()
            bad_mask = ~np.isfinite(v)
            bad_count = int(bad_mask.sum())
            if bad_count:
                if not skip_non_finite:
                    idx = np.where(bad_mask)[0]
                    examples = idx[:5].tolist()
                    try:
                        runs = gdf["run_name"].astype(str).iloc[idx[:5]].tolist()
                    except Exception:
                        runs = []
                    raise RuntimeError(
                        f"non-finite pearson for {c} in encoder {enc}: {bad_count}/{len(v)} bad; idx: {examples}; runs: {runs}"
                    )
                v = v[np.isfinite(v)]
            if v.size > 0:
                per_cols.append(v.astype(float))
        if not per_cols:
            if skip_non_finite:
                continue
            raise RuntimeError(f"no values collected for encoder {enc}; check filters and genes")
        flat = np.concatenate(per_cols, axis=0)
        values[str(enc)] = flat.tolist()
    if not values:
        raise RuntimeError("no values collected; check filters and genes")
    return values


def _ensure_out_path(path_no_ext: str, ext: str) -> str:
    base = f"{path_no_ext}.{ext}"
    if not os.path.exists(base):
        return base
    i = 2  # e.g. hvg2.png
    while True:
        cand = f"{path_no_ext}{i}.{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def _plot_box(values: Dict[str, List[float]], title: str, out_path: str) -> None:
    encoders = sorted(values.keys())
    if not encoders:
        raise RuntimeError("no encoder groups to plot")
    data = [values[e] for e in encoders]
    means = [float(np.mean(v)) if len(v) else float("nan") for v in data]

    fig, ax = plt.subplots(figsize=(8, 4.5))  # figsize (8, 4.5) chosen for 16:9
    ax.boxplot(data, labels=encoders, showfliers=False)
    ax.scatter(range(1, len(encoders) + 1), means, color="black", s=20, zorder=3)
    ax.set_ylim(-1.0, 1.0)  # enforce Pearson range [-1, 1]
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Encoder type")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    if not isinstance(raw_cfg, dict):
        raise RuntimeError("config must be a mapping")

    setup_dump_env()

    _require_keys(raw_cfg, ["gene_sets", "log_to_wandb"]) 

    gene_sets = raw_cfg.get("gene_sets")
    if not isinstance(gene_sets, dict) or not gene_sets:
        raise ValueError("gene_sets must be a non-empty mapping")
    for name, gl in gene_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gene_sets keys must be non-empty strings")
        if not isinstance(gl, list) or not all(isinstance(g, str) for g in gl):
            raise ValueError(f"gene_sets['{name}'] must be a list of strings")

    include_projects = raw_cfg.get("include_projects") or None
    include_encoders = raw_cfg.get("include_encoders") or None
    include_run_name_regex = raw_cfg.get("include_run_name_regex") or None
    exclude_run_name_regex = raw_cfg.get("exclude_run_name_regex") or None

    log_to_wandb = bool(raw_cfg.get("log_to_wandb"))
    run = None
    if log_to_wandb:
        for key in ("run_name", "group", "job_type", "tags", "project"):
            if key not in raw_cfg:
                raise ValueError(f"Missing required parameter '{key}' for W&B logging")
        wb_cfg = {k: v for k, v in raw_cfg.items() if k not in ("project", "metric", "method", "run_name", "group", "job_type", "tags")}
        original_run_name = raw_cfg.get("run_name")
        run = wandb.init(
            project=raw_cfg["project"],
            name=raw_cfg["run_name"],
            group=raw_cfg["group"],
            job_type=raw_cfg["job_type"],
            tags=raw_cfg["tags"],
            config=wb_cfg,
        )
        raw_cfg = dict(raw_cfg)
        raw_cfg.update(dict(run.config))
        if original_run_name is not None:
            raw_cfg["run_name"] = original_run_name

    df = _load_all_forward_metrics(EVAL_ROOT)
    df = _apply_filters(
        df,
        include_projects=include_projects,
        include_encoders=include_encoders,
        include_run_name_regex=include_run_name_regex,
        exclude_run_name_regex=exclude_run_name_regex,
    )

    saved_paths: List[str] = []
    skip_non_finite = bool(raw_cfg.get("skip_non_finite", False))
    for set_name, genes in gene_sets.items():
        vals = _collect_values_by_encoder(df, genes, skip_non_finite)
        title = f"Pearson by encoder — {set_name}"
        fname = _sanitize_token(set_name)
        out_base = os.path.join(OUT_DIR, fname)
        out_path = _ensure_out_path(out_base, "png")
        _plot_box(vals, title, out_path)
        saved_paths.append(out_path)
        if run is not None:
            run.log({f"boxplot/{set_name}": wandb.Image(out_path)})

    if run is not None:
        run.finish()

    # Print saved files for quick inspection
    for p in saved_paths:
        print(p)


if __name__ == "__main__":
    main()
