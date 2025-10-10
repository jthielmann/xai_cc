import os
# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Keep thread pools small to reduce runtime conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Headless-safe Matplotlib backend (prevents some backend segfaults)
os.environ.setdefault("MPLBACKEND", "Agg")
# Optional last-resort for duplicate OpenMP (use only if still crashing)
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch

import csv, sys, os, random, numpy, torch, yaml, pandas as pd, wandb
sys.path.insert(0, '..')

from typing import Dict, Any, List, Union, Optional
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
from script.evaluation.xai_pipeline import XaiPipeline
from main_utils import (
    ensure_free_disk_space,
    parse_args,
    parse_yaml_config,
    read_config_parameter,
    get_sweep_parameter_names,
    make_run_name_from_config,
    make_sweep_name_from_space,
)

from pathlib import Path

def setup_dump_env(dump_dir: Optional[str] = None) -> str:
    """Configure env vars so incidental outputs go under a single dump dir.

    Returns the resolved dump_dir path.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = Path.cwd()
    dd = Path(
        dump_dir
        or os.environ.get("XAI_DUMP_DIR")
        or (repo_root / "dump")
    ).resolve()
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
    for k in [
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "TORCH_HOME",
        "MPLCONFIGDIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
    ]:
        try:
            Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return str(dd)

def _prepare_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg); cfg.update(get_dataset_cfg(cfg))
    out = cfg.get("out_path") or cfg.get("sweep_dir") or cfg.get("model_dir")
    os.makedirs(out, exist_ok=True); ensure_free_disk_space(out)
    with open(os.path.join(out, "config"), "w") as f: yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return cfg

import os, glob
import pandas as pd
from typing import Dict, Any, List, Union

def _prepare_gene_list(cfg: Dict[str, Any]) -> Any:
    if cfg.get("genes") is not None:
        return None

    # Validate mutually exclusive CSV configuration early
    if cfg.get("single_csv_path") and (
        cfg.get("train_csv_path") or cfg.get("val_csv_path") or cfg.get("test_csv_path")
    ):
        raise ValueError(
            "Provide either 'single_csv_path' or split-specific CSVs ('train_csv_path'/'val_csv_path'/'test_csv_path'), not both."
        )

    data_dir = cfg.get("data_dir")
    meta_dir = str(cfg.get("meta_data_dir", "meta_data")).strip("/")
    fname = cfg.get("gene_data_filename")

    files = []
    # Prefer single CSV(s) if provided
    scp = cfg.get("single_csv_path")
    if scp:
        # Resolve relative to data_dir if needed
        path = None
        if not os.path.isabs(scp) and cfg.get("data_dir"):
            cand = os.path.join(cfg["data_dir"], scp)
            if os.path.isfile(cand):
                path = cand
        if path is None and os.path.isfile(scp):
            path = scp
        if path is None:
            raise FileNotFoundError(f"single_csv_path not found: '{scp}'. Tried: '{scp}' and data_dir-joined '{os.path.join(cfg.get('data_dir',''), scp)}'")
        files = [path]
    elif cfg.get("train_csv_path") and os.path.isfile(cfg["train_csv_path"]):
        files = [cfg["train_csv_path"]]
    else:
        sample_ids = cfg.get("sample_ids")
        if sample_ids:
            for sid in sample_ids:
                path = os.path.join(data_dir, sid, meta_dir, fname)
                if os.path.isfile(path):
                    files.append(path)
        else:
            files = glob.glob(os.path.join(data_dir, "*", meta_dir, fname))
            files.sort()

    if not files:
        raise FileNotFoundError(f"No gene data files found under {data_dir}/*/{meta_dir}/{fname}")

    max_files = int(cfg.get("gene_detect_max_files", 50))
    files = files[:max_files]

    order = None
    inter = None
    for i, path in enumerate(files):
        df = pd.read_csv(path, nrows=1)
        cols = [c for c in df.columns if c != "tile" and not str(c).endswith("_lds_w") and pd.api.types.is_numeric_dtype(df[c])]
        s = set(cols)
        if i == 0:
            order = cols[:]
            inter = s
        else:
            inter &= s
        if not inter:
            break

    genes = [c for c in order if c in inter] if order else []
    if not genes:
        raise ValueError("Could not infer any gene columns. Check your CSV headers and dtypes.")

    if cfg.get("split_genes_by"):
        k = int(cfg["split_genes_by"])
        if k <= 0:
            raise ValueError("split_genes_by must be a positive integer.")
        chunks = [genes[i:i+k] for i in range(0, len(genes), k)]
        cfg["genes"] = chunks
        cfg["gene_chunks"] = chunks
    else:
        cfg["genes"] = genes
        cfg["gene_chunks"] = [genes]

    cfg["n_gene_chunks"] = len(cfg["gene_chunks"])
    return cfg["genes"]

def _get_active_chunk_idx(cfg: Dict[str, Any], chunk: Optional[List[str]] = None) -> int:
    chunks = cfg.get("gene_chunks") or []
    if not chunks:
        return 0
    if chunk is None:
        if cfg.get("genes") and isinstance(cfg["genes"][0], list):
            idx = int(cfg.get("gene_list_index", 1)) - 1
            return max(0, min(idx, len(chunks) - 1))
        # if cfg["genes"] is already a single chunk
        chunk = cfg.get("genes")
    try:
        return next(i for i, ch in enumerate(chunks) if ch == chunk)
    except StopIteration:
        raise RuntimeError(f"chunk not found {chunk} in {chunks}")


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
            cfg["genes"] = _prepare_gene_list(cfg)
        TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

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
        gene_chunks = _prepare_gene_list(tmp_cfg)
    except Exception:
        gene_chunks = None

    # Only log genes_id if the original config requested splitting by chunks
    def _had_split_genes(cfg_dict: Dict[str, Any]) -> bool:
        # check top-level
        if "split_genes_by" in cfg_dict and cfg_dict["split_genes_by"] is not None:
            try:
                return int(cfg_dict["split_genes_by"]) > 0
            except Exception:
                return False
        # check parameters.value
        p = cfg_dict.get("parameters", {}).get("split_genes_by")
        if isinstance(p, dict) and "value" in p and p["value"] is not None:
            try:
                return int(p["value"]) > 0
            except Exception:
                return False
        return False

    if gene_chunks and chosen_genes is not None and _had_split_genes(raw_cfg):
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

def log_runtime_banner():
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print(f"[runtime] torch={torch.__version__} cuda={torch.version.cuda} device={dev} bf16_supported={torch.cuda.is_bf16_supported() if dev=='cuda' else False}")

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
            gene_chunks = _prepare_gene_list(tmp_cfg)  # flat list or list-of-lists
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

        # If this is an XAI pipeline config, handle it directly here,
        # preserving any provided out_path and preparing the dump env.
        if bool(cfg.get("xai_pipeline", False)):
            wb_cfg = {k: v for k, v in cfg.items() if k not in ("parameters", "metric", "method")}
            run = wandb.init(project=cfg.get("project", "xai"), config=wb_cfg) if cfg.get("log_to_wandb", False) else None
            cfg = dict(run.config) if run else cfg
            cfg.setdefault("dump_dir", setup_dump_env(cfg.get("dump_dir")))
            cfg = _prepare_cfg(cfg)  # ensures dataset cfg and creates out_path/sweep_dir/model_dir
            XaiPipeline(cfg, run=run).run()
            if run:
                run.finish()
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
