from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import wandb
import numpy as np
import pandas as pd


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


def _load_forward_metrics_recursive(scan_root: str) -> pd.DataFrame:
    if not isinstance(scan_root, str) or not scan_root.strip():
        raise ValueError(f"scan_root invalid: {scan_root!r}")
    if not os.path.isdir(scan_root):
        raise FileNotFoundError(f"scan_root not found or not a dir: {scan_root}")
    frames: List[pd.DataFrame] = []
    for dirpath, dirnames, filenames in os.walk(scan_root):
        if "/debug/" in (dirpath + "/"):
            continue
        if "forward_metrics.csv" in filenames:
            csvp = os.path.join(dirpath, "forward_metrics.csv")
            df = pd.read_csv(csvp)
            df["_src_dir"] = dirpath
            frames.append(df)
    if not frames:
        raise RuntimeError(f"no forward_metrics.csv found under {scan_root}/**")
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


def _collect_values_by_encoder(
    df: pd.DataFrame, genes: List[str], skip_non_finite: bool, group_key: str = "encoder_type"
) -> Dict[str, List[float]]:
    cols = [f"pearson_{g}" for g in genes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing pearson columns for genes: {missing}")
    if group_key not in df.columns:
        raise RuntimeError(f"group_key column missing: {group_key}")
    values: Dict[str, List[float]] = {}
    for enc, gdf in df.groupby(group_key):
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
            raise RuntimeError(
                f"no values collected for encoder {enc}; check filters and genes"
            )
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


def _plot_violin(values: Dict[str, List[float]], title: str, out_path: str) -> None:
    encoders = sorted(values.keys())
    if not encoders:
        raise RuntimeError("no encoder groups to plot")
    data = [values[e] for e in encoders]
    means = [float(np.mean(v)) if len(v) else float("nan") for v in data]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.violinplot(data, showmeans=False, showextrema=True, showmedians=False)
    ax.set_xticks(range(1, len(encoders) + 1))
    ax.set_xticklabels(encoders)
    ax.scatter(range(1, len(encoders) + 1), means, color="black", s=20, zorder=3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Encoder type")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


__all__ = [
    "_sanitize_token",
    "_require_keys",
    "_load_forward_metrics_recursive",
    "_apply_filters",
    "_collect_values_by_encoder",
    "_ensure_out_path",
    "_plot_box",
    "_plot_violin",
]


def _validate_gene_sets(gene_sets: Dict[str, List[str]]) -> None:
    if not isinstance(gene_sets, dict) or not gene_sets:
        raise ValueError("gene_sets must be a non-empty mapping")
    for name, gl in gene_sets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gene_sets keys must be non-empty strings")
        if not isinstance(gl, list) or not all(isinstance(g, str) for g in gl):
            raise ValueError(f"gene_sets['{name}'] must be a list of strings")


def _maybe_init_wandb_and_update_cfg(raw_cfg: Dict[str, Any]):
    run = None
    if bool(raw_cfg.get("log_to_wandb")):
        for key in ("run_name", "group", "job_type", "tags", "project"):
            if key not in raw_cfg:
                raise ValueError(
                    f"Missing required parameter '{key}' for W&B logging"
                )
        wb_cfg = {
            k: v
            for k, v in raw_cfg.items()
            if k
            not in (
                "project",
                "metric",
                "method",
                "run_name",
                "group",
                "job_type",
                "tags",
            )
        }
        original_run_name = raw_cfg.get("run_name")
        run = wandb.init(
            project=raw_cfg["project"],
            name=raw_cfg["run_name"],
            group=raw_cfg["group"],
            job_type=raw_cfg["job_type"],
            tags=raw_cfg["tags"],
            config=wb_cfg,
        )
        new_cfg = dict(raw_cfg)
        new_cfg.update(dict(run.config))
        if original_run_name is not None:
            new_cfg["run_name"] = original_run_name
        return run, new_cfg
    return None, raw_cfg


def _plot_all_sets(
    df: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    plot_box: bool,
    plot_violin: bool,
    skip_non_finite: bool,
    run,
    out_dir: str,
    group_key: str = "encoder_type",
) -> List[str]:
    saved_paths: List[str] = []
    for set_name, genes in gene_sets.items():
        vals = _collect_values_by_encoder(df, genes, skip_non_finite, group_key=group_key)
        title = f"Pearson by encoder — {set_name}"
        fname = _sanitize_token(set_name)
        if plot_box:
            out_base_box = os.path.join(out_dir, f"{fname}__box")
            out_path_box = _ensure_out_path(out_base_box, "png")
            _plot_box(vals, title, out_path_box)
            saved_paths.append(out_path_box)
            if run is not None:
                run.log({f"boxplot/{set_name}": wandb.Image(out_path_box)})
        if plot_violin:
            out_base_violin = os.path.join(out_dir, f"{fname}__violin")
            out_path_violin = _ensure_out_path(out_base_violin, "png")
            _plot_violin(vals, title, out_path_violin)
            saved_paths.append(out_path_violin)
            if run is not None:
                run.log({f"violinplot/{set_name}": wandb.Image(out_path_violin)})
    if not saved_paths:
        raise RuntimeError("no plots saved")
    return saved_paths
from __future__ import annotations
