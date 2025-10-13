import sys
sys.path.insert(0, '..')
import os

from script.gene_list_helpers import prepare_gene_list, had_split_genes
from script.train.lit_train_sae import SAETrainerPipeline

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import torch

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
    run = wandb.init(project=cfg.get("project","xai"), config=wb_cfg) if cfg.get("log_to_wandb", False) else None
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
            cfg["genes"] = prepare_gene_list(cfg)
        TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

# Locate the config by name or path
def _resolve_config_path(name: str) -> str:
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    if os.path.isfile(name):
        return name
    # search common config roots
    search_roots = [
        ".",
        "../sweeps/configs",
        "../sweeps/configs/sweep",
        "../sweeps/configs/single",
    ]
    for root in search_roots:
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    # fallback: recursive search under sweeps/configs
    for root, _dirs, files in os.walk("sweeps/configs"):
        if os.path.basename(name) in files:
            return os.path.join(root, os.path.basename(name))
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

    # Name the run based on chosen hyperparameters (ignore provided config name)
    try:
        sweep_param_names = get_sweep_parameter_names(raw_cfg)
        # expose names to downstream so TrainerPipeline can detect sweep context
        run.config.update({"sweep_parameter_names": sweep_param_names}, allow_val_change=True)
        run_name = make_run_name_from_config(rcfg, sweep_param_names)
        run.name = run_name
        # also surface it in config for filtering if desired
        run.config.update({"auto_run_name": run_name}, allow_val_change=True)
    except Exception:
        pass

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

    # Ensure dump_dir present and env set before training
    cfg.setdefault("dump_dir", setup_dump_env(cfg.get("dump_dir")))
    cfg = _prepare_cfg(cfg)
    if bool(cfg.get("train_sae", False)):
        SAETrainerPipeline(cfg, run=run).run()
    else:
        TrainerPipeline(cfg, run=run).run()
    run.finish()

def main():
    print("main debug")
    args = parse_args()
    # Set up dump env early so any libs honor it
    dump_dir = None
    try:
        # Allow dump_dir override via the config file if top-level key exists
        tmp = parse_yaml_config(args.config)
        if isinstance(tmp, dict) and tmp.get("dump_dir"):
            dump_dir = str(tmp.get("dump_dir"))
    except Exception:
        pass
    setup_dump_env(dump_dir)
    raw_cfg = parse_yaml_config(args.config)
    params = raw_cfg.get("parameters", {})
    is_sweep = any(isinstance(param, dict) and "values" in param for param in params.values())
    if is_sweep:
        # Only include hyperparameters in the sweep, plus a config_name to locate the base config per run
        def _only_hyperparams(pdict: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in pdict.items():
                if isinstance(v, dict) and ("values" in v or "distribution" in v):
                    out[k] = v
            return out

        hyper_params = _only_hyperparams(read_config_parameter(raw_cfg, "parameters"))

        # Add config name so each run can load the base config for fixed parameters
        config_name = os.path.basename(args.config)
        hyper_params["config_name"] = {"value": config_name}

        sweep_config = {
            # Ignore config 'name'; build from hyperparameter space
            "name": make_sweep_name_from_space(raw_cfg),
            "method": read_config_parameter(raw_cfg, "method"),
            "metric": read_config_parameter(raw_cfg, "metric"),
            "parameters": hyper_params,
            # Avoid adding project/description into the sweep config so they don't leak into run.config
        }
        # carry project separately for wandb.sweep(..., project=project)
        # if project isn't in the raw config parameters or top-level, default
        try:
            project = read_config_parameter(raw_cfg, "project") if not read_config_parameter(raw_cfg, "debug") else "_debug_" + random.randbytes(4).hex()
        except Exception:
            project = "xai"
        # Determine model_dir from the base config (not from parameters)
        model_dir = read_config_parameter(raw_cfg, "model_dir") if "model_dir" in raw_cfg or "model_dir" in raw_cfg.get("parameters", {}) else "../models/"
        sweep_dir = os.path.join(model_dir, project)
        if not os.path.exists(sweep_dir): os.makedirs(sweep_dir, exist_ok=True)
        ensure_free_disk_space(sweep_dir)
        print(f"Project: {project}")
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, sweep_config["name"])
        os.makedirs(sweep_id_dir, exist_ok=True)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f: sweep_id = f.read().strip()
            print(f"Loaded existing sweep ID: {sweep_id}")
        else:
            # start from raw_cfg to get non-parameter fields like data_dir, filenames, etc.
            tmp_cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
            # overlay fixed parameter values from the raw config
            for k, v in raw_cfg.get("parameters", {}).items():
                if isinstance(v, dict) and "value" in v:
                    tmp_cfg[k] = v["value"]
            tmp_cfg.update(get_dataset_cfg(tmp_cfg))
            gene_chunks = prepare_gene_list(tmp_cfg)  # flat list or list-of-lists
            if gene_chunks: # is None if specific gene list was provided in config, e.g. not chunking all genes
                # normalize to list-of-lists so the sweep can iterate values
                if isinstance(gene_chunks[0], str):
                    gene_chunks = [gene_chunks]
                # inject as sweep param (hyperparameter)
                sweep_config["parameters"]["genes"] = {"values": gene_chunks}
            else:
                g = tmp_cfg["genes"]
                sweep_config["parameters"]["genes"] = (
                    {"values": g} if g and isinstance(g[0], list) else {"values": [g]}
                )
            # prevent double splitting within the runner; we'll recompute chunks per run
            sweep_config["parameters"].pop("split_genes_by", None)
            sweep_config["method"] = sweep_config.get("method") or "grid"

            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Initialized new sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        # Normalize model_dir for single runs
        val = params.get("model_dir", "../models/")
        if not isinstance(val, dict):
            params["model_dir"] = {"value": str(val)}
        params["model_dir"]["value"] = "../models/"
        cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
        for k, p in params.items():
            if isinstance(p, dict) and "value" in p: cfg[k] = p["value"]

        # If this is an XAI pipeline config, do not run it from here.
        # We keep XAI in a separate entrypoint so it can be executed
        # in a dedicated conda environment without bringing in training deps.
        if bool(cfg.get("xai_pipeline", False)):
            print("[info] Detected xai_pipeline config. Please run:\n"
                  "  python script/xai_main.py --config", args.config)
            return

        # Otherwise: standard training single-run flow
        project = cfg["project"] if not read_config_parameter(raw_cfg, "debug") else "debug_" + random.randbytes(4).hex()
        out_dir = cfg["model_dir"] + project
        os.makedirs(out_dir, exist_ok=True)
        ensure_free_disk_space(out_dir)
        cfg["out_path"] = out_dir
        _train(cfg)

if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    main()
