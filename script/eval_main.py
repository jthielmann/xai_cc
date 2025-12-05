import os
import shutil
import sys
from random import random

import pandas as pd

sys.path.insert(0, '..')

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from typing import Any, Dict, List

import yaml
import wandb

from script.configs.dataset_config import get_dataset_cfg
from script.xai_auto_config import build_auto_xai_config
from script.evaluation.eval_pipeline import EvalPipeline
from script.evaluation.gather_results import gather_forward_metrics
from script.main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, setup_dump_env, \
    read_config_parameter, compute_genes_id
from script.boxplot_helpers import (
    _maybe_init_wandb_and_update_cfg,
    _load_forward_metrics_recursive,
    _plot_all_sets,
)


def _resolve_relative(path: str, source_path=None) -> str:
    if not path:
        raise ValueError("_resolve_relative received empty path")
    if os.path.isabs(path):
        return path
    bases = []
    if source_path:
        bases.append(os.path.dirname(os.path.abspath(source_path)))
    bases.append(os.getcwd())
    for base in bases:
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(bases[0], path)) if bases else path


def _prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    if "dataset" not in merged:
        mc = merged.get("model_config") or {}
        ds = mc.get("dataset")
        if not ds:
            raise KeyError("'dataset' missing; provide it in config or ensure model_config.dataset exists")
        merged["dataset"] = ds
    merged.update(get_dataset_cfg(merged))

    # Determine base evaluation directory with gene_set/encoder_type subfolders
    enc = merged.get("encoder_type") or (merged.get("model_config") or {}).get("encoder_type")
    if not isinstance(enc, str) or not enc.strip():
        raise ValueError("encoder_type missing; required to build eval output path")
    enc_token = _sanitize_token(enc)
    genes = (merged.get("model_config") or {}).get("genes")
    project = merged.get("model_config").get("project")
    if not genes:
        raise ValueError("model_config.genes missing; required to build gene_set path")
    gs_token = _sanitize_token(compute_genes_id(genes))
    base_root = "../evaluation/debug" if bool(merged.get("debug", False)) else "../evaluation"
    label = merged.get("eval_label")
    if isinstance(label, str) and label.strip():
        label = _sanitize_token(label)
    else:
        label = None
    base = os.path.join(base_root, label, gs_token, enc_token) if label else os.path.join(base_root, gs_token, enc_token)
    if isinstance(project, str) and project.strip():
        eval_path = os.path.join(base, _sanitize_token(project.replace(" ", "")))
    else:
        raise ValueError("model_config.project missing; required for unique eval path")
    merged["eval_path"] = eval_path
    # Maintain legacy 'out_path' for downstream helpers that expect it
    merged["out_path"] = eval_path

    os.makedirs(eval_path, exist_ok=True)
    ensure_free_disk_space(eval_path)
    dump_cfg = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    with open(os.path.join(eval_path, "config"), "w") as handle:
        yaml.safe_dump(dump_cfg, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return merged


def _sanity_check_config(config: Dict[str, Any]):
    if config.get("model_state_path") and not config.get("encoder_state_path") and not config.get("gene_head_state_path"):
        return
    if not config.get("model_state_path") and config.get("encoder_state_path") and config.get("gene_head_state_path"):
        return
    raise RuntimeError(
        f"corrupted config:\n"
        f"encoder_state_path {config.get('encoder_state_path', 'empty')}\n"
        f"gene_head_state_path {config.get('gene_head_state_path', 'empty')}\n"
        f"model_state_path {config.get('model_state_path', 'empty')}\n"
    )


def _verify_path(path, models_dir):
    if not os.path.exists(path):
        path = os.path.join(models_dir, path)
        if not os.path.exists(path):
            raise RuntimeError(f"path does not exist {path}")
        return path
    return path

# set and verify config path and model path
def _build_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    models_dir = "../models"

    # Resolve optional model_config_path relative to models_dir if needed
    config_path = config.get("model_config_path")
    if config_path:
        if not os.path.exists(config_path):
            config_path = os.path.join(models_dir, config_path)
            if not os.path.exists(config_path):
                raise RuntimeError(f"invalid config path {config_path}")
            config["model_config_path"] = config_path

    # Verify and resolve weight paths
    encoder_state_path = config.get("encoder_state_path")
    gene_head_state_path = config.get("gene_head_state_path")
    model_state_path = config.get("model_state_path")

    if encoder_state_path and gene_head_state_path:
        config["encoder_state_path"]   = _verify_path(encoder_state_path,   models_dir)
        config["gene_head_state_path"] = _verify_path(gene_head_state_path, models_dir)
        return config

    if model_state_path and not gene_head_state_path and not encoder_state_path:
        config["model_state_path"] = _verify_path(model_state_path, models_dir)
        return config

    raise RuntimeError(
        f"corrupted config:\n"
        f"encoder_state_path {config.get('encoder_state_path', 'empty')}\n"
        f"gene_head_state_path {config.get('gene_head_state_path', 'empty')}\n"
        f"model_state_path {config.get('model_state_path', 'empty')}\n"
    )

def _setup_model_config(config_path: str) -> Dict[str, Any]:
    """Load a model config from the exact given path and fail on error.

    No fallbacks or alternative filenames are attempted.
    """
    p = _resolve_relative(config_path)
    if not os.path.exists(p):
        raise RuntimeError(f"model config path does not exist: {p}")
    raw_cfg = parse_yaml_config(p)
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"invalid or empty model config at: {p}")
    return {k: v for k, v in raw_cfg.items()}


def _sanitize_token(s: str) -> str:
    return (
        str(s)
        .replace("\\", "/")
        .rstrip("/")
        .replace("/", "__")
        .replace(" ", "_")
    )[:128]


def _run_single(raw_cfg: Dict[str, Any], run=None) -> None:
    cfg = _build_cfg(raw_cfg)
    mode = cfg.get("xai_pipeline")
    if isinstance(mode, str):
        m = mode.strip().lower()
        if m == "auto":
            cfg = build_auto_xai_config(cfg)
        elif m == "manual":
            pass
        else:
            raise ValueError("invalid xai_pipeline value; expected 'manual' or 'auto'")
    _sanity_check_config(cfg)
    cfg["model_config"] = _setup_model_config(os.path.join(cfg["model_state_path"], "config"))
    # Always allow gradients through the encoder during evaluation/XAI
    cfg["model_config"]["freeze_encoder"] = False
    # If debug, set concrete caps to reduce compute (keep W&B logging unchanged)
    if bool(cfg.get("debug", False)):
        cfg = dict(cfg)
        cfg["max_len"] = 100              # cap rows per patient file
        cfg["lrp_max_items"] = 4          # few attribution samples
        cfg["lxt_max_items"] = 8          # few heatmaps
        cfg["umap_max_samples"] = 400     # limit UMAP points
        cfg["umap_batch_size"] = 16       # smaller eval batches
        cfg["scatter_max_items"] = 100    # few points for scatter
        cfg["forward_max_tiles"] = 1000   # cap tiles forwarded
        cfg["diff_max_items"] = 200       # cap tiles in triptychs
    setup_dump_env()

    local_run = run
    if local_run is None and bool(cfg.get("log_to_wandb")):
        for key in ("run_name", "group", "job_type", "tags"):
            if key not in cfg:
                raise ValueError(f"Missing required parameter '{key}' in config")
        wb_cfg = {k: v for k, v in cfg.items() if k not in ("project", "metric", "method", "run_name", "group", "job_type", "tags")}
        original_run_name = cfg.get("run_name")
        local_run = wandb.init(
            project=cfg.get("project", "xai"),
            name=cfg["run_name"],
            group=cfg["group"],
            job_type=cfg["job_type"],
            tags=cfg["tags"],
            config=wb_cfg,
        )
        cfg = dict(cfg)
        cfg.update(dict(local_run.config))
        if original_run_name is not None:
            cfg["run_name"] = original_run_name
    cfg = _prepare_cfg(cfg)

    pipeline = EvalPipeline(cfg, run=local_run)
    try:
        pipeline.run()
    finally:
        pipeline.cleanup()

    if local_run:
        local_run.finish()


def _find_model_run_dirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Base directory not found or not a directory: {base_dir}"
        )
    hits: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "config" in filenames and "best_model.pth" in filenames:
            hits.append(os.path.abspath(dirpath))
    hits.sort()
    return hits


def _compute_boxplot_root(cfg: Dict[str, Any]) -> str:
    debug_flag = bool(cfg["debug"])
    base_root = "../evaluation/debug" if debug_flag else "../evaluation"
    label = cfg.get("eval_label")
    if not isinstance(label, str) or not label.strip():
        msp = cfg.get("model_state_path")
        if isinstance(msp, str) and msp.strip():
            label = msp
    label_tok = (
        label.replace("\\", "/").rstrip("/").replace("/", "__").replace(" ", "_")[:128]
        if isinstance(label, str) and label.strip()
        else None
    )
    ms = read_config_parameter(cfg, "model_state_path")
    if not isinstance(ms, str) or not ms.strip():
        raise ValueError("model_state_path must be a string for boxplots root computation")
    ms_path = _verify_path(ms, "../models")
    model_cfg = parse_yaml_config(os.path.join(ms_path, "config")) or {}
    genes = model_cfg.get("genes")
    if not genes:
        raise RuntimeError("model_config.genes missing in model config for boxplots root computation")
    gs_token = _sanitize_token(compute_genes_id(genes))
    root = os.path.join(base_root, label_tok, gs_token) if label_tok else os.path.join(base_root, gs_token)
    return root


def _simple_filter_df(df, include_projects=None, include_encoders=None):
    out = df
    if include_projects:
        out = out[out["project"].astype(str).isin([str(x) for x in include_projects])]
    if include_encoders:
        out = out[out["encoder_type"].astype(str).isin([str(x) for x in include_encoders])]
    if out.empty:
        raise RuntimeError("no rows left after filters")
    return out


def _infer_gene_sets_from_df(df):
    genes: List[str] = []
    for c in df.columns:
        if not c.startswith("pearson_"):
            continue
        g = c[len("pearson_") :].strip()
        if not g:
            continue
        if g.lower().startswith("unnamed"):
            continue
        genes.append(g)
    if not genes:
        raise RuntimeError("no pearson_* columns found to infer genes")
    return {"all": genes}


def _run_boxplots(root: str, cfg: Dict[str, Any]) -> None:
    if not cfg["boxplots"]:
        return
    df = _load_forward_metrics_recursive(root)
    df = _simple_filter_df(
        df,
        include_projects=cfg.get("include_projects"),
        include_encoders=cfg.get("include_encoders"),
    )
    gene_sets = cfg.get("gene_sets")
    if isinstance(gene_sets, dict):
        cleaned = {}
        for name, gl in gene_sets.items():
            vals = []
            for g in gl:
                s = str(g).strip()
                if s and not s.lower().startswith("unnamed"):
                    vals.append(s)
            cleaned[name] = vals
        gene_sets = cleaned
    else:
        gene_sets = _infer_gene_sets_from_df(df)

    group_by = str(cfg.get("group_by", "gene_set+encoder_type+loss")).strip().lower()
    group_col = "__group__"
    if group_by == "encoder_type":
        df[group_col] = df["encoder_type"].astype(str)
    elif group_by == "encoder_type+loss":
        if "loss_name" not in df.columns:
            raise RuntimeError("loss_name column missing in metrics; re-run aggregation")
        df[group_col] = df[["encoder_type", "loss_name"]].astype(str).agg(lambda r: f"{r[0]} ({r[1]})", axis=1)
    elif group_by == "project":
        df[group_col] = df["project"].astype(str)
    elif group_by == "project+encoder_type":
        df[group_col] = df[["project", "encoder_type"]].astype(str).agg(lambda r: f"{r[0]}::{r[1]}", axis=1)
    elif group_by == "gene_set+encoder_type+loss":
        if "gene_set" not in df.columns or "loss_name" not in df.columns:
            raise RuntimeError("gene_set/loss_name column missing in metrics; re-run aggregation")
        df[group_col] = df[["gene_set", "encoder_type", "loss_name"]].astype(str).agg(lambda r: f"{r[0]}::{r[1]} ({r[2]})", axis=1)
    else:
        raise ValueError(f"unsupported group_by: {group_by}")

    # Deduplicate per gene within each group: keep best (max Pearson) per gene
    pearson_cols = [c for c in df.columns if c.startswith("pearson_")]
    if not pearson_cols:
        raise RuntimeError("no pearson_* columns found for boxplots")
    df = df.groupby(group_col, as_index=False)[pearson_cols].max()

    run, cfg2 = _maybe_init_wandb_and_update_cfg(cfg)
    out_dir = os.path.join(root, "boxplots")
    _plot_all_sets(
        df=df,
        gene_sets=gene_sets,
        plot_box=bool(cfg2.get("plot_box", True)),
        plot_violin=bool(cfg2.get("plot_violin", False)),
        skip_non_finite=bool(cfg2.get("skip_non_finite", False)),
        run=run,
        out_dir=out_dir,
        group_key=group_col,
    )
    if run is not None:
        run.finish()


def xor(a, b):
    return (a and not b) or (not a and b)


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)
    flags = {}
    flagnames = ["forward_to_csv", "scatter", "diff", "umap", "lxt", "lrp", "log_to_wandb", "boxplots",
                  "forward_to_csv_simple"]
    for flagname in flagnames:
        if flagname not in raw_cfg:
            raise KeyError(f"Missing required flag '{flagname}' in config for sweep mode")
        else:
            flags[flagname] = bool(raw_cfg[flagname])

    # gather all dirs that contain best_model.pth
    model_dirs = []
    model_dirs_incomplete = []
    base_dir = "../models/" + read_config_parameter(raw_cfg, "model_state_path")
    if not base_dir or not os.path.isdir(base_dir):
        raise RuntimeError(f"model state path {base_dir} not found for config variable 'model_state_path'")
    # single run with best_model.pth and config directly there
    if os.path.exists(base_dir + "/config"):
        model_dirs.append(base_dir)
    # split_genes_by case
    else:
        for run_name in os.listdir(base_dir):
            print(run_name, base_dir)
            exit(0)
            run_dir = os.path.join(base_dir, run_name)
            config_path = os.path.join(run_dir, "config")
            model_path = os.path.join(run_dir, "best_model.pth")
            if xor(os.path.exists(config_path), os.path.exists(model_path)):
                model_dirs_incomplete.append(run_dir)
                continue
            if os.path.exists(os.path.join(run_dir, "config")) and os.path.exists(os.path.join(run_dir, "best_model.pth")):
                model_dirs.append(run_dir)
            else:
                for gene_split_name in os.listdir(run_dir):
                    gene_split_dir = os.path.join(run_dir, gene_split_name)

                    config_path = os.path.join(gene_split_dir, "config")
                    model_path = os.path.join(gene_split_dir, "best_model.pth")

                    if xor(os.path.exists(config_path), os.path.exists(model_path)):
                        model_dirs_incomplete.append(run_dir)
                        continue
                    if os.path.exists(config_path) and os.path.exists(model_path):
                        model_dirs.append(os.path.join(run_dir, gene_split_dir))
                    else:
                        raise RuntimeError(f"model state path {run_dir}/{gene_split_dir} not found")

        if len(model_dirs) == 0:
            raise RuntimeError("no model state paths found")
        if len(model_dirs_incomplete) > 0:
            model_dirs_incomplete_df = pd.DataFrame(model_dirs_incomplete)
            location = f"../evaluation/missing/{base_dir[10:]}/"
            if not os.path.exists(location):
                os.makedirs(location)

            model_dirs_incomplete_df.to_csv(location + "incomplete_runs.csv", index=False)

        for i in range(len(model_dirs)):
            for flagname in flags.keys():
                if flags[flagname] is False:
                    continue
                print(i, "/", len(model_dirs), "->", model_dirs[i], flagname)
                model_dir = model_dirs[i]
                flags[flagname] = bool(raw_cfg.get(flagname))
                per_cfg = dict(raw_cfg)
                model_config = parse_yaml_config(model_dir + "/config")
                per_cfg["model_state_path"] = model_dir
                per_cfg["eval_label"] = flagname
                per_cfg["encoder_type"] = model_config["encoder_type"]
                per_cfg["run_name"] = None
                per_cfg["model_config"] = model_config
                _run_single(per_cfg, run=None)


if __name__ == "__main__":
    main()
