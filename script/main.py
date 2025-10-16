import os


# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Be conservative with thread pools by default (can be overridden by user env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import torch
from umap import UMAP
import csv, os, random, numpy, torch, yaml, pandas as pd, wandb

import sys
sys.path.insert(0, '..')

from script.gene_list_helpers import prepare_gene_list, had_split_genes, get_full_gene_list
from script.train.lit_train_sae import SAETrainerPipeline
from typing import Dict, Any, List, Union, Optional
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
from main_utils import (
    ensure_free_disk_space,
    parse_args,
    parse_yaml_config,
    read_config_parameter,
    get_sweep_parameter_names,
    make_run_name_from_config,
    setup_dump_env
)


def _prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg); cfg.update(get_dataset_cfg(cfg))
    out = "../models"
    os.makedirs(out, exist_ok=True); ensure_free_disk_space(out)
    with open(os.path.join(out, "config"), "w") as f: yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return cfg

import os, glob
import pandas as pd
from typing import Dict, Any, List, Union

def _train(cfg: Dict[str, Any]) -> None:
    # Flatten only at handoff to training
    flat_cfg = _flatten_params(cfg)

    # Initialize W&B with flattened config
    log_to_wandb = bool(read_config_parameter(cfg, "log_to_wandb"))
    if log_to_wandb:
        run = wandb.init(
            project=read_config_parameter(cfg, "project"),
            name=read_config_parameter(cfg, "run_name"),
            group=read_config_parameter(cfg, "group"),
            job_type=read_config_parameter(cfg, "job_type"),
            tags=read_config_parameter(cfg, "tags"),
            config=flat_cfg,
        )
    else:
        run = None

    # Use flattened config for downstream pipeline
    cfg = dict(run.config) if run else flat_cfg

    # Ensure dump_dir and prepare cfg
    cfg.setdefault("dump_dir", setup_dump_env())

    cfg = _prepare_cfg(cfg)
    cfg = _flatten_params(cfg)

    if bool(cfg.get("train_sae", False)):
        # No gene list inference needed — training uses encoder features only
        print("SAETrainerPipeline debug")
        SAETrainerPipeline(cfg, run=run).run()
    else:
        if cfg.get("genes") is None:
            cfg["genes"] = get_full_gene_list(cfg)
        TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

# Locate the config by name or path
def _resolve_config_path(name: str) -> str:

    if os.path.isfile(name):
        return name
    # search common config roots
    root = "../sweeps/configs"
    cand = os.path.join(root, name)
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"Could not resolve config_name '{name}' to a file path")

def _sweep_run():
    # Ensure dump env in agent subprocess before init
    setup_dump_env()
    run = wandb.init()
    run_config = dict(run.config)
    config_name = run_config.get("config_name")
    if not config_name:
        raise RuntimeError("Sweep run missing 'config_name' in parameters")


    auto_name = make_run_name_from_config(run_config)
    run.config.update({"auto_run_name": auto_name}, allow_val_change=True)

    # Start from base config: top-level value wrappers and fixed parameters.value
    cfg: Dict[str, Any] = {}
    for k, v in raw_cfg.items():
        if k == "parameters":
            continue
        if isinstance(v, dict) and "value" in v and "values" not in v:
            cfg[k] = v["value"]
        else:
            cfg[k] = v
    for k, pv in (raw_cfg.get("parameters", {}) or {}).items():
        if isinstance(pv, dict) and "value" in pv and "values" not in pv:
            cfg[k] = pv["value"]

    # Overlay chosen hyperparameters from the sweep run, excluding meta keys
    exclude = {"metric", "method", "_wandb", "config_name"}
    for k, v in run_config.items():
        if k not in exclude:
            cfg[k] = v

    if cfg.get("encoder_type") in ["dinov3_vits16plus"] or "convnext" in cfg.get("encoder_type"):
        cfg["image_size"] = 384

    # Recompute gene chunks from the resolved config so we can log genes_id deterministically
    # Build a temporary cfg to infer gene list without forcing a previously chosen chunk
    tmp_cfg = dict(cfg)
    chosen_genes = run_config.get("genes")
    if chosen_genes is not None:
        # prevent short-circuit in _prepare_gene_list
        tmp_cfg.pop("genes", None)
    tmp_cfg.update(get_dataset_cfg(tmp_cfg))

    gene_chunks = prepare_gene_list(tmp_cfg)

    if gene_chunks and chosen_genes is not None and had_split_genes(raw_cfg):
        if isinstance(gene_chunks, list) and gene_chunks and isinstance(gene_chunks[0], str):
            gene_chunks = [gene_chunks]
        idx = next(i for i, ch in enumerate(gene_chunks) if ch == chosen_genes)
        run.config.update({"genes_id": str(idx)}, allow_val_change=True)

    base_model_dir = "../models/"
    project = cfg.get("project")
    project_dir = os.path.join(base_model_dir, project)
    os.makedirs(project_dir, exist_ok=True)
    ensure_free_disk_space(project_dir)

    cfg["model_dir"] = project_dir
    cfg["sweep_dir"] = project_dir

    wb_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
    run.config.update(wb_cfg, allow_val_change=True)

    # Set required W&B run identity fields from sweep config
    for key in ("run_name", "group", "job_type", "tags"):
        if key not in cfg:
            raise RuntimeError(f"Missing required parameter '{key}' for sweep run")
    run.name = cfg["run_name"]
    run.group = cfg["group"]
    run.job_type = cfg["job_type"]
    run.tags = cfg["tags"]

    # Ensure dump_dir present and env set before training
    setup_dump_env()
    cfg = _prepare_cfg(cfg)
    if bool(cfg.get("train_sae", False)):
        SAETrainerPipeline(cfg, run=run).run()
    else:
        TrainerPipeline(cfg, run=run).run()
    run.finish()

def _extract_hyperparams(pdict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pdict.items():
        if isinstance(v, dict) and ("values" in v or "distribution" in v):
            out[k] = v
    return out

def _flatten_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for k, v in raw.items():
        if type(v) == str:
            cfg[k] = v
        elif k == "parameters" or "metric":
            continue
        else:
            cfg[k] = v["value"]
    params = raw.get("parameters")
    for pk, pv in params.items():
        cfg[pk] = pv["value"]
    return cfg

def _build_sweep_config(raw_cfg: Dict[str, Any], config_basename: Optional[str] = None) -> Dict[str, Any]:
    """Build a W&B sweep spec from a raw config with a 'parameters' section."""
    params_dict = (raw_cfg.get("parameters", {}) or {}) if isinstance(raw_cfg, dict) else {}
    hyper_params = _extract_hyperparams(params_dict)

    # Add config name so each run can load the base config for fixed parameters
    hyper_params["config_name"] = {"value": config_basename}

    # Minimal fields for dataset resolution and gene discovery
    cfg_for_genes: Dict[str, Any] = {}
    for key in ("dataset", "debug", "gene_data_filename", "meta_data_dir", "sample_ids", "split_genes_by", "genes"):
        val = read_config_parameter(raw_cfg, key)
        cfg_for_genes[key] = val
    cfg_for_genes.update(get_dataset_cfg(cfg_for_genes))
    gene_chunks = prepare_gene_list(dict(cfg_for_genes))

    sweep_config = {
        "name": make_run_name_from_config(raw_cfg, get_sweep_parameter_names(raw_cfg)),
        "method": read_config_parameter(raw_cfg, "method"),
        "metric": read_config_parameter(raw_cfg, "metric"),
        "parameters": hyper_params,
    }

    if "genes" not in hyper_params:
        if gene_chunks:
            sweep_config["parameters"]["genes"] = {"values": gene_chunks}
        else:
            fixed_genes = read_config_parameter(raw_cfg, "genes")
            sweep_config["parameters"]["genes"] = {"value": fixed_genes}


    sweep_param_names = get_sweep_parameter_names(raw_cfg)
    sweep_config.update({"sweep_parameter_names": sweep_param_names}, allow_val_change=True)


    return sweep_config

def _has_sweep_params(config: Dict[str, Any]) -> bool:
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    if not isinstance(params, dict):
        return False
    return any(isinstance(v, dict) and ("values" in v or "distribution" in v) for v in params.values())

def main():
    args = parse_args()

    raw_cfg = parse_yaml_config(args.config)

    xai_pipeline = (read_config_parameter(raw_cfg, "xai_pipeline"))
    if xai_pipeline:
        raise RuntimeError("[info] Detected xai_pipeline config, instead run:\n"
                           "  python script/eval_main.py --config", args.config)

    setup_dump_env()

    debug = read_config_parameter(raw_cfg, "debug")
    project = read_config_parameter(raw_cfg, "project") if not debug else "_debug_" + random.randbytes(4).hex()

    if debug:
        print("python version:", sys.version)
        print("numpy version:", numpy.version.version)
        print("torch version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())

    # Decide sweep vs single run from parameters section only (no global flatten)
    is_sweep = _has_sweep_params(raw_cfg)
    if is_sweep:
        name = make_run_name_from_config(raw_cfg, get_sweep_parameter_names(raw_cfg))
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, name)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f: sweep_id = f.read().strip()
        else:
            os.makedirs(sweep_id_dir, exist_ok=True)
            sweep_config = _build_sweep_config(raw_cfg, os.path.basename(args.config))
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        _train(raw_cfg)

if __name__ == "__main__":
    main()
