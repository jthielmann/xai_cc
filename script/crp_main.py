import os
import sys
from random import random

sys.path.insert(0, '..')

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from typing import Any, Dict, Optional

import yaml
import wandb

from script.configs.dataset_config import get_dataset_cfg
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
        try:
            bases.append(os.path.dirname(os.path.abspath(source_path)))
        except Exception:
            pass
    bases.append(os.getcwd())
    for base in bases:
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(bases[0], path)) if bases else path


def _prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dataset defaults and ensure output directory exists, persist config."""
    merged = dict(cfg)
    merged.update(get_dataset_cfg(merged))
    out_root = merged.get("out_path") or merged.get("sweep_dir") or merged.get("model_dir") or "./xai_out"
    os.makedirs(out_root, exist_ok=True)
    ensure_free_disk_space(out_root)
    dump_cfg = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    with open(os.path.join(out_root, "config"), "w") as handle:
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

def _setup_model_config(config_name:str):
    raw_cfg = parse_yaml_config(config_name)
    config = {k: v for k, v in raw_cfg.items()}
    return config


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)
    cfg = _build_cfg(raw_cfg)
    if not bool(cfg.get("xai_pipeline", False)):
        raise ValueError("Config must set 'xai_pipeline: true' when using script/eval_main.py")
    _sanity_check_config(cfg)
    cfg["model_config"] = _setup_model_config("../models/" + cfg["model_state_path"] + "/config")
    setup_dump_env()

    run = None
    if bool(cfg.get("log_to_wandb")):
        # Enforce required W&B identity parameters
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
            config=wb_cfg
        )
        # Merge back W&B-config values without losing our explicit run_name
        cfg = dict(cfg)  # start from original cfg as source of truth
        cfg.update(dict(run.config))
        if original_run_name is not None:
            cfg["run_name"] = original_run_name
    cfg = _prepare_cfg(cfg)

    EvalPipeline(cfg, run=run).run()

    if run:
        run.finish()


if __name__ == "__main__":
    main()
