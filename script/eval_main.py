import os
import sys
from random import random

sys.path.insert(0, '..')

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from typing import Any, Dict, Optional, List

import yaml
import wandb

from script.configs.dataset_config import get_dataset_cfg
from script.xai_auto_config import build_auto_xai_config
from script.evaluation.eval_pipeline import EvalPipeline
from script.main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, setup_dump_env, \
    read_config_parameter


def _resolve_relative(path: str, source_path: Optional[str] = None) -> str:
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
    merged.update(get_dataset_cfg(merged))

    # Determine base evaluation directory with encoder_type subfolder
    enc = merged.get("encoder_type") or (merged.get("model_config") or {}).get("encoder_type")
    if not isinstance(enc, str) or not enc.strip():
        raise ValueError("encoder_type missing; required to build eval output path")
    enc_token = _sanitize_token(enc)
    base = os.path.join("../evaluation", enc_token)
    eval_path = os.path.join(base, "debug") if bool(merged.get("debug", False)) else base
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


def _run_single(raw_cfg: Dict[str, Any]) -> None:
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

    run = None
    if bool(cfg.get("log_to_wandb")):
        for key in ("run_name", "group", "job_type", "tags"):
            if key not in cfg:
                raise ValueError(f"Missing required parameter '{key}' in config")
        wb_cfg = {k: v for k, v in cfg.items() if k not in ("project", "metric", "method", "run_name", "group", "job_type", "tags")}
        original_run_name = cfg.get("run_name")
        run = wandb.init(
            project=cfg.get("project", "xai"),
            name=cfg["run_name"],
            group=cfg["group"],
            job_type=cfg["job_type"],
            tags=cfg["tags"],
            config=wb_cfg,
        )
        cfg = dict(cfg)
        cfg.update(dict(run.config))
        if original_run_name is not None:
            cfg["run_name"] = original_run_name
    cfg = _prepare_cfg(cfg)

    EvalPipeline(cfg, run=run).run()

    if run:
        run.finish()


def _find_model_run_dirs(base_dir: str) -> List[str]:
    """Return subdirectories under base_dir that look like model runs.

    A valid run dir must contain a file named 'config' and 'best_model.pth'.
    Raises if the base_dir does not exist or is not a directory.
    Returns absolute paths sorted by name for stable iteration.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found or not a directory: {base_dir}")
    entries = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    out: List[str] = []
    for p in entries:
        if not os.path.isdir(p):
            continue
        cfg = os.path.join(p, "config")
        wts = os.path.join(p, "best_model.pth")
        if os.path.exists(cfg) and os.path.exists(wts):
            out.append(os.path.abspath(p))
    out.sort()
    return out


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    # Project-scope eval: when 'is_sweep' is true, treat model_state_path as a base
    # directory and run eval for each subdir that looks like a model run.
    if bool(raw_cfg.get("is_sweep", False)):
        base_val = read_config_parameter(raw_cfg, "model_state_path")
        if not base_val or not isinstance(base_val, str):
            raise ValueError("is_sweep=true requires 'model_state_path' to be a directory path")
        base_dir = _verify_path(base_val, "../models")
        if not os.path.isdir(base_dir):
            raise RuntimeError(f"is_sweep=true but model_state_path is not a directory: {base_dir}")
        run_dirs = _find_model_run_dirs(base_dir)
        if not run_dirs:
            raise RuntimeError(f"No model run subdirectories with config+best_model.pth under: {base_dir}")
        base_run_name = raw_cfg.get("run_name") or "eval"
        models_root = os.path.dirname(os.path.abspath(base_dir))
        # Insert encoder_type into evaluation output base for per-run skip checks
        # Read encoder_type from each run's stored model config
        debug_flag = bool(raw_cfg.get("debug", False))
        # out_base is computed per-run below since it depends on encoder_type
        for rd in run_dirs:
            model_cfg_path = os.path.join(rd, "config")
            if not os.path.exists(model_cfg_path):
                raise RuntimeError(f"missing model config for run dir: {rd}")
            mc = parse_yaml_config(model_cfg_path) or {}
            enc = mc.get("encoder_type")
            if not isinstance(enc, str) or not enc.strip():
                raise ValueError(f"encoder_type missing in model config: {model_cfg_path}")
            out_base = os.path.join("../evaluation", _sanitize_token(enc))
            tgt_base = os.path.join(out_base, "debug") if debug_flag else out_base
            rel_model = os.path.relpath(rd, models_root)
            tgt = os.path.join(tgt_base, rel_model)
            if os.path.exists(tgt):
                continue
            per_cfg = dict(raw_cfg)
            per_cfg["model_state_path"] = rd
            per_cfg["out_path"] = out_base
            per_cfg["encoder_type"] = enc
            rel = os.path.relpath(rd, base_dir)
            token = _sanitize_token(rel)
            per_cfg["run_name"] = f"{base_run_name}__{token}"[:128]
            _run_single(per_cfg)
        return

    # Detect sweep-style lists for model_state_path and expand into multiple runs.
    ms_param = read_config_parameter(raw_cfg, "model_state_path")
    if isinstance(ms_param, list):
        base_run_name = raw_cfg.get("run_name") or "forward_to_csv"
        for ms in ms_param:
            per_cfg = dict(raw_cfg)
            per_cfg["model_state_path"] = ms
            # Ensure a unique, informative run_name for each model
            token = _sanitize_token(ms)
            per_cfg["run_name"] = f"{base_run_name}__{token}"[:128]
            _run_single(per_cfg)
        return

    # Fallback to single-run behavior
    _run_single(raw_cfg)


if __name__ == "__main__":
    main()
