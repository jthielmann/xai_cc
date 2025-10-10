import os

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Keep thread pools small to reduce runtime conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Headless-safe Matplotlib backend
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import wandb

from script.configs.dataset_config import get_dataset_cfg
from script.evaluation.xai_pipeline import XaiPipeline
from script.main_utils import ensure_free_disk_space, parse_args, parse_yaml_config


def setup_dump_env(dump_dir: Optional[str] = None) -> str:
    """Configure env vars so incidental outputs go under a single dump dir."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = Path.cwd()
    dd = Path(dump_dir or os.environ.get("XAI_DUMP_DIR") or (repo_root / "dump")).resolve()
    os.makedirs(dd, exist_ok=True)

    # W&B local dirs (run files and cache)
    os.environ.setdefault("WANDB_DIR", str(dd / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(dd / "wandb_cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(dd / "wandb_config"))
    # Torch / torchvision cache (pretrained weights, etc.)
    os.environ.setdefault("TORCH_HOME", str(dd / "torch_cache"))
    # Matplotlib cache
    os.environ.setdefault("MPLCONFIGDIR", str(dd / "mpl-cache"))
    # Common ML caches (harmless if unused)
    os.environ.setdefault("HF_HOME", str(dd / "hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(dd / "hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(dd / "hf_cache"))

    # Ensure directories exist
    for env_key in (
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "TORCH_HOME",
        "MPLCONFIGDIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    ):
        try:
            Path(os.environ[env_key]).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return str(dd)


def _flatten_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
    for key, param in raw_cfg.get("parameters", {}).items():
        if isinstance(param, dict) and "value" in param:
            cfg[key] = param["value"]
    return cfg


def _resolve_relative(path: str, base: Optional[str]) -> str:
    if not path or os.path.isabs(path) or not base:
        return path
    base_dir = os.path.dirname(os.path.abspath(base))
    return os.path.normpath(os.path.join(base_dir, path))


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


def _build_cfg(raw_cfg: Dict[str, Any], *, source_path: Optional[str] = None) -> Dict[str, Any]:
    """Flatten raw config, optionally inheriting values from a referenced training config."""
    cfg = _flatten_config(raw_cfg)
    if source_path:
        cfg["_config_path"] = os.path.abspath(source_path)

    base_cfg = None
    base_path = cfg.get("model_config_path")
    if base_path:
        resolved = _resolve_relative(str(base_path), source_path)
        base_raw = parse_yaml_config(resolved)
        base_cfg = _flatten_config(base_raw)
        cfg["_model_config_path"] = resolved
        cfg["model_config_path"] = resolved
        for key, value in base_cfg.items():
            if key not in cfg or cfg[key] in (None, ""):
                cfg[key] = value

    if (cfg.get("model_state_path") in (None, "") and base_cfg):
        candidates = []
        out_root = base_cfg.get("out_path")
        if out_root:
            candidates.extend([
                os.path.join(out_root, "best_model.pth"),
                os.path.join(out_root, "latest.pth"),
                os.path.join(out_root, "model.ckpt"),
            ])
        model_dir = base_cfg.get("model_dir")
        project = base_cfg.get("project")
        if model_dir and project:
            candidates.append(os.path.join(model_dir, project, "best_model.pth"))
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                cfg["model_state_path"] = candidate
                break

    return cfg


def _run_pipeline(cfg: Dict[str, Any]) -> None:
    """Initialize optional logging, finalize config, and execute the XAI pipeline."""
    run = None
    if bool(cfg.get("log_to_wandb", False)):
        wb_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
        run = wandb.init(project=cfg.get("project", "xai"), config=wb_cfg)
        cfg = dict(run.config)

    cfg.setdefault("dump_dir", setup_dump_env(cfg.get("dump_dir")))
    cfg = _prepare_cfg(cfg)

    XaiPipeline(cfg, run=run).run()

    if run:
        run.finish()


def main() -> None:
    args = parse_args()

    # Align with training entrypoint: respect dump_dir hints before heavy imports.
    dump_dir = None
    try:
        tmp = parse_yaml_config(args.config)
        if isinstance(tmp, dict) and tmp.get("dump_dir"):
            dump_dir = str(tmp["dump_dir"])
    except Exception:
        pass
    setup_dump_env(dump_dir)

    raw_cfg = parse_yaml_config(args.config)
    cfg = _build_cfg(raw_cfg, source_path=args.config)

    if not bool(cfg.get("xai_pipeline", False)):
        raise ValueError("Config must set 'xai_pipeline: true' when using script/xai_main.py")

    _run_pipeline(cfg)


if __name__ == "__main__":
    main()
