import os


# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Be conservative with thread pools by default (can be overridden by user env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# Required for deterministic CuBLAS ops on CUDA >= 10.2 when Lightning sets deterministic=True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from umap import UMAP
import csv, os, random, numpy, torch, yaml, pandas as pd, wandb

import sys
sys.path.insert(0, '..')

from script.gene_list_helpers import prepare_gene_list, get_full_gene_list, had_split_genes, _GENE_SETS
from script.train.lit_train_sae import SAETrainerPipeline
from typing import Dict, Any, List
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
from main_utils import (
    ensure_free_disk_space,
    parse_args,
    parse_yaml_config,
    read_config_parameter,
    get_sweep_parameter_names,
    make_run_name_from_config,
    setup_dump_env,
    prepare_cfg,
    compute_genes_id,
)


import os, glob
import pandas as pd
from typing import Dict, Any, List



def _apply_gene_set_inplace(cfg: Dict[str, Any]) -> None:
    gs = cfg.get("gene_set")
    if gs is None:
        return
    if cfg.get("genes") is not None:
        raise ValueError("cannot set both 'genes' and 'gene_set'")
    key = str(gs).strip().lower()
    if key == "common":
        return
    if key not in _GENE_SETS:
        raise ValueError(f"unknown gene_set {gs!r}")
    cfg["genes"] = list(_GENE_SETS[key])


def _validate_learning_rates(cfg: Dict[str, Any]) -> None:
    fixed = cfg.get("global_fix_learning_rate")
    enc_lr = cfg.get("encoder_lr")
    ratio = cfg.get("encoder_lr_ratio")
    if enc_lr is None:
        return
    if ratio is not None:
        raise ValueError(f"encoder_lr={enc_lr} conflicts with encoder_lr_ratio={ratio}; choose one")
    if fixed is None:
        return
    fixed_val = float(fixed)
    if fixed_val >= 0:
        raise ValueError(
            f"encoder_lr={enc_lr} conflicts with global_fix_learning_rate={fixed}; use encoder_lr_ratio"
        )

def _train(cfg: Dict[str, Any]) -> None:
    # Flatten only at handoff to training
    flat_cfg = _flatten_params(cfg)
    _apply_gene_set_inplace(flat_cfg)
    _validate_learning_rates(flat_cfg)

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

    cfg = prepare_cfg(cfg)

    if bool(cfg.get("train_sae", False)):
        # No gene list inference needed — training uses encoder features only
        print("SAETrainerPipeline debug")
        SAETrainerPipeline(cfg, run=run).run()
    else:
        # Use provided genes if available; otherwise infer from dataset
        if cfg.get("genes") is None:
            cfg["genes"] = get_full_gene_list(cfg)
        # If split_genes_by is set, split the provided list into chunks and pick an index
        split_k = cfg.get("split_genes_by")
        genes = cfg.get("genes")
        if split_k:
            k = int(split_k)
            if k <= 0:
                raise ValueError("split_genes_by must be a positive integer.")
            # Do not accept nested lists silently; require a flat list of genes
            if isinstance(genes, list) and genes and isinstance(genes[0], list):
                raise ValueError(
                    "Config 'genes' must be a flat list of gene names when using split_genes_by; "
                    "got a list of lists. Provide a flat list and use 'gene_list_index' to select a chunk."
                )
            tgt = [str(g) for g in genes]
            chunks = [tgt[i:i+k] for i in range(0, len(tgt), k)]
            # Select chunk via 1-based gene_list_index if provided, else first chunk
            idx = int(cfg.get("gene_list_index", 1)) - 1
            idx = max(0, min(idx, len(chunks) - 1)) if chunks else 0
            cfg["genes"] = chunks[idx] if chunks else []
            cfg["genes_id"] = f"c{idx+1:03d}"
        else:
            # No chunking requested; derive a stable id from the provided list
            cfg["genes_id"] = compute_genes_id(cfg.get("genes"))
        TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

 # Locate the config by name or path
def _resolve_config_path(name: str) -> str:
    if os.path.isfile(name):
        return name
    # search common config roots
    roots = ["../sweeps/configs", "../sweeps/to_train"]
    for root in roots:
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    cwd = os.getcwd()
    raise FileNotFoundError(f"Could not resolve config_name '{name}' from cwd={cwd!r}; searched: {roots}")

def _flatten_base_fixed(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts fixed values from a base config that may use {"value": ...}
    wrappers and a 'parameters' section. Only fixed values are retained.
    """
    out: Dict[str, Any] = {}

    # Top-level fields: allow either raw values or {"value": ...}
    for k, v in cfg.items():
        if k in ("parameters", "metric", "method", "name", "sweep_parameter_names"):
            continue
        if isinstance(v, dict) and "value" in v:
            out[k] = v["value"]
        else:
            out[k] = v

    # Fixed values inside the 'parameters' section (ignore sweep search specs)
    params = cfg.get("parameters")
    if isinstance(params, dict):
        for pk, pv in params.items():
            if isinstance(pv, dict) and "value" in pv and not ("values" in pv or "distribution" in pv):
                out[pk] = pv["value"]
    return out

def _sweep_run():
    # Ensure dump env in agent subprocess before init
    setup_dump_env()
    run = wandb.init()
    run_config = dict(run.config)
    config_name = run_config.get("config_name")
    if not config_name:
        raise RuntimeError("Sweep run missing 'config_name' in parameters")

    base_config_path = _resolve_config_path(run_config["config_name"])
    base_config = parse_yaml_config(base_config_path)
    parameter_names = get_sweep_parameter_names(base_config)
    auto_name = make_run_name_from_config(run_config, parameter_names)
    run.name = auto_name

    base_fixed = _flatten_base_fixed(base_config)
    merged = dict(base_fixed)
    merged.update({k: v for k, v in run_config.items() if k != "config_name"})
    merged["name"] = auto_name
    merged["sweep_parameter_names"] = list(parameter_names)

    project = merged.get("project") or read_config_parameter(base_config, "project")
    base_model_dir = "../models/"
    project_dir = os.path.join(base_model_dir, project)
    os.makedirs(project_dir, exist_ok=True)
    ensure_free_disk_space(project_dir)

    merged["model_dir"] = project_dir
    merged["sweep_dir"] = project_dir
    # Tag runs/dirs with an identifier derived from the selected genes
    if merged.get("genes") is not None:
        try:
            merged["genes_id"] = compute_genes_id(merged.get("genes"))
        except Exception:
            merged["genes_id"] = "genes"
    merged = prepare_cfg(merged)
    run.config.update(merged, allow_val_change=True)

    # Ensure dump_dir present and env set before training
    if bool(merged.get("train_sae", False)):
        SAETrainerPipeline(merged, run=run).run()
    else:
        TrainerPipeline(merged, run=run).run()
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
        elif k in ("parameters", "metric"):
            continue
        else:
            cfg[k] = v["value"]
    params = raw.get("parameters")
    for pk, pv in params.items():
        cfg[pk] = pv["value"]
    return cfg

def _build_sweep_config(raw_cfg: Dict[str, Any], config_basename: str = None) -> Dict[str, Any]:
    """Build a W&B sweep spec from a raw config with a 'parameters' section."""
    params_dict = (raw_cfg.get("parameters", {})) if isinstance(raw_cfg, dict) else {}
    hyper_params = _extract_hyperparams(params_dict)

    # Add config name so each run can load the base config for fixed parameters
    hyper_params["config_name"] = {"value": config_basename}

    # Minimal fields for dataset resolution and gene discovery
    cfg_for_genes: Dict[str, Any] = {}
    for key in ("dataset", "debug", "gene_data_filename", "meta_data_dir", "sample_ids", "split_genes_by", "genes", "gene_set"):
        val = read_config_parameter(raw_cfg, key)
        cfg_for_genes[key] = val
    if read_config_parameter(raw_cfg, "gene_set") is not None and read_config_parameter(raw_cfg, "genes") is not None:
        raise ValueError("cannot set both 'genes' and 'gene_set'")
    cfg_for_genes = prepare_cfg(cfg_for_genes)
    _apply_gene_set_inplace(cfg_for_genes)
    gene_chunks = prepare_gene_list(dict(cfg_for_genes))

    sweep_config = {
        "name": raw_cfg["name"],
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
    sweep_config["sweep_parameter_names"] = sweep_param_names


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
        raise RuntimeError("[info] Detected xai_pipeline config (manual/auto). Instead run:\n"
                           f"  python script/eval_main.py --config {args.config}")

    setup_dump_env()

    debug = read_config_parameter(raw_cfg, "debug")
    project = read_config_parameter(raw_cfg, "project") if not debug else "_debug_" + random.randbytes(4).hex()

    if debug:
        print("python version:", sys.version)
        print("numpy version:", numpy.version.version)
        print("torch version:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())

    is_sweep = _has_sweep_params(raw_cfg)
    if is_sweep:
        name = raw_cfg["name"]
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, name)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")
        os.environ["WANDB_RUN_GROUP"] = str(read_config_parameter(raw_cfg, "group"))
        os.environ["WANDB_JOB_TYPE"] = str(read_config_parameter(raw_cfg, "job_type"))
        tags = read_config_parameter(raw_cfg, "tags")
        os.environ["WANDB_TAGS"] = ",".join(map(str, tags))
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
