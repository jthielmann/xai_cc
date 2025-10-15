import sys
sys.path.insert(0, '..')
import os

from script.gene_list_helpers import prepare_gene_list, had_split_genes, get_full_gene_list
from script.train.lit_train_sae import SAETrainerPipeline

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Be conservative with thread pools by default (can be overridden by user env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
# Use a safer multiprocessing start method to avoid fork-related segfaults in DataLoader workers
try:
    import torch.multiprocessing as mp
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except Exception:
    # Best-effort: if already set or unsupported, ignore
    pass

import csv, os, random, numpy, torch, yaml, pandas as pd, wandb


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
    make_sweep_name_from_space,
    setup_dump_env
)


def _prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg); cfg.update(get_dataset_cfg(cfg))
    out = cfg.get("out_path") or cfg.get("sweep_dir") or cfg.get("model_dir")
    os.makedirs(out, exist_ok=True); ensure_free_disk_space(out)
    with open(os.path.join(out, "config"), "w") as f: yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return cfg

import os, glob
import pandas as pd
from typing import Dict, Any, List, Union

def _train(cfg: Dict[str, Any]) -> None:
    # Sanitize config for W&B: avoid nested sweep keys showing as empty columns
    wb_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
    if cfg.get("log_to_wandb", False):
        run = wandb.init(
            project=cfg.get("project", "xai"),
            name=read_config_parameter(cfg, "run_name"),
            group=read_config_parameter(cfg, "group"),
            job_type=read_config_parameter(cfg, "job_type"),
            tags=read_config_parameter(cfg, "tags"),
            config=wb_cfg,
        )
    else:
        run = None
    # Ensure dump_dir is present in cfg and env
    cfg = dict(run.config) if run else cfg
    cfg.setdefault("dump_dir", setup_dump_env(cfg.get("dump_dir")))
    cfg = _prepare_cfg(cfg)

    # SAE path: only train sparse autoencoder, no gene heads/lr find.
    if bool(cfg.get("train_sae", False)):
        from script.train.lit_train_sae import SAETrainerPipeline
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

# Only log genes_id if the original config requested splitting by chunks

def _sweep_run():
    # Ensure dump env in agent subprocess before init
    setup_dump_env()
    run = wandb.init()
    # Extract hyperparameters chosen by the sweep plus our config name
    rcfg = dict(run.config)

    # Resolve base config from the provided config name
    config_name = rcfg.get("config_name")
    if not config_name:
        raise RuntimeError("Sweep run missing 'config_name' in parameters")

    base_cfg_path = _resolve_config_path(config_name)
    raw_cfg = parse_yaml_config(base_cfg_path)

    sweep_param_names = get_sweep_parameter_names(raw_cfg)
    # expose names to downstream so TrainerPipeline can detect sweep context
    run.config.update({"sweep_parameter_names": sweep_param_names}, allow_val_change=True)
    auto_name = make_run_name_from_config(rcfg, sweep_param_names)
    # record auto-generated suggestion for reference; do not override required run_name
    run.config.update({"auto_run_name": auto_name}, allow_val_change=True)


    # Start from base config (top-level keys except 'parameters')
    cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}

    # Overlay fixed parameters from base config
    for k, p in raw_cfg.get("parameters", {}).items():
        if isinstance(p, dict) and "value" in p:
            cfg[k] = p["value"]

    # Overlay chosen hyperparameters from the sweep run
    # Exclude W&B-internal keys and our helper keys
    exclude = {"parameters", "metric", "method", "_wandb", "config_name"}
    for k, v in rcfg.items():
        if k not in exclude:
            cfg[k] = v

    if cfg.get("encoder_type") in ["dinov3_vits16plus"] or "convnext" in cfg.get("encoder_type"):
        cfg["image_size"] = 384

    # Recompute gene chunks from the resolved config so we can log genes_id deterministically
    # Build a temporary cfg to infer gene list without forcing a previously chosen chunk
    tmp_cfg = dict(cfg)
    chosen_genes = rcfg.get("genes")
    if chosen_genes is not None:
        # prevent short-circuit in _prepare_gene_list
        tmp_cfg.pop("genes", None)
    tmp_cfg.update(get_dataset_cfg(tmp_cfg))
    try:
        gene_chunks = prepare_gene_list(tmp_cfg)
    except Exception:
        gene_chunks = None

    if gene_chunks and chosen_genes is not None and had_split_genes(raw_cfg):
        # normalize chunks to list-of-lists
        if isinstance(gene_chunks, list) and gene_chunks and isinstance(gene_chunks[0], str):
            gene_chunks = [gene_chunks]
        try:
            idx = next(i for i, ch in enumerate(gene_chunks) if ch == chosen_genes)
            run.config.update({"genes_id": str(idx)}, allow_val_change=True)
        except StopIteration:
            pass

    # Ensure output roots are defined (move single-run behavior into sweep runs)
    base_model_dir = None
    if "model_dir" in raw_cfg or "model_dir" in raw_cfg.get("parameters", {}):
        try:
            base_model_dir = read_config_parameter(raw_cfg, "model_dir")
        except Exception:
            base_model_dir = None
    if not base_model_dir:
        base_model_dir = "../models/"

    project = cfg.get("project", "xai")
    project_dir = os.path.join(base_model_dir, project)
    os.makedirs(project_dir, exist_ok=True)
    ensure_free_disk_space(project_dir)

    # Align fields so downstream training has what it expects
    cfg["model_dir"] = project_dir
    cfg["sweep_dir"] = project_dir

    # Proactively surface resolved config at sweep-run start so W&B columns
    # are populated immediately (not only after training begins).
    try:
        wb_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
        run.config.update(wb_cfg, allow_val_change=True)
    except Exception:
        pass

    # Set required W&B run identity fields from sweep config
    for key in ("run_name", "group", "job_type", "tags"):
        if key not in cfg:
            raise RuntimeError(f"Missing required parameter '{key}' for sweep run")
    run.name = cfg["run_name"]
    run.group = cfg["group"]
    run.job_type = cfg["job_type"]
    run.tags = list(cfg["tags"]) if not isinstance(cfg["tags"], list) else cfg["tags"]

    # Ensure dump_dir present and env set before training
    cfg.setdefault("dump_dir", setup_dump_env(cfg.get("dump_dir")))
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

def _build_sweep_config(config):
    hyper_params = _extract_hyperparams(read_config_parameter(config, "parameters"))

    # Add config name so each run can load the base config for fixed parameters
    config_name = os.path.basename(config)
    hyper_params["config_name"] = {"value": config_name}

    config.update(get_dataset_cfg(config))

    sweep_config = {
        "name": make_sweep_name_from_space(config), # Ignores config 'name', it is only used for slurm purposes
        "method": read_config_parameter(config, "method"),
        "metric": read_config_parameter(config, "metric"),
        "parameters": hyper_params
    }
    gene_chunks = prepare_gene_list(config)  # flat list or list-of-lists
    if gene_chunks: # is None if specific gene list was provided in config, e.g. not chunking all genes
        sweep_config["parameters"]["genes"] = {"values": gene_chunks}
    else:
        sweep_config["parameters"]["genes"] = config["genes"]
    return sweep_config

def main():
    args = parse_args()

    raw_cfg = parse_yaml_config(args.config)

    setup_dump_env()

    debug = read_config_parameter(raw_cfg, "debug")
    project = read_config_parameter(raw_cfg, "project") if not debug else "_debug_" + random.randbytes(4).hex()

    if debug:
        print("python version:", sys.version)
        print("numpy version:", numpy.version.version)
        print("torch version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())

    model_dir = "../models" + project
    os.makedirs(model_dir, exist_ok=True)
    ensure_free_disk_space(model_dir)

    config = {k: v for k, v in raw_cfg.items()}
    config["model_dir"]["value"] = model_dir
    is_sweep = any(isinstance(param, dict) and "values" in param for param in config.values())

    if is_sweep:
        name = make_sweep_name_from_space(raw_cfg)
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, name)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f: sweep_id = f.read().strip()
        else:
            os.makedirs(sweep_id_dir, exist_ok=True)
            sweep_config = _build_sweep_config(config)
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        if bool(config.get("xai_pipeline", False)):
            print("[info] Detected xai_pipeline config, instead run:\n"
                  "  python script/eval_main.py --config", args.config)
            return

        _train(config)

if __name__ == "__main__":
    main()
