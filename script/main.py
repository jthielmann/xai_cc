import csv, sys, os, random, numpy, torch, yaml, pandas as pd, wandb
from typing import Dict, Any, List, Union, Optional

sys.path.insert(0, '..')
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
from main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, read_config_parameter, get_sweep_parameter_names

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

    data_dir = cfg["data_dir"]
    fname = cfg["gene_data_filename"]

    files = []
    sample_ids = cfg.get("sample_ids")
    if sample_ids:
        for sid in sample_ids:
            path = os.path.join(data_dir, sid, "meta_data", fname)
            if os.path.isfile(path):
                files.append(path)
    else:
        files = glob.glob(os.path.join(data_dir, "*", "meta_data", fname))
        files.sort()

    if not files:
        raise FileNotFoundError(f"No gene data files found under {data_dir}/*/meta_data/{fname}")

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
    run = wandb.init(project=cfg.get("project","xai"), config=cfg) if cfg.get("log_to_wandb", False) else None
    cfg = _prepare_cfg(dict(run.config) if run else cfg)

    if cfg.get("genes") is None:
        cfg["genes"] = _prepare_gene_list(cfg)

    TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

def _sweep_run():
    run = wandb.init()
    chunks = run.config.get("gene_chunks", None)
    if chunks:
        chosen = run.config["genes"]
        idx = next(i for i, ch in enumerate(chunks) if ch == chosen)
        run.config.update({"genes_id": str(idx)}, allow_val_change=True)
    cfg = _prepare_cfg(dict(run.config))
    TrainerPipeline(cfg, run=run).run(); run.finish()

def log_runtime_banner():
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    print(f"[runtime] torch={torch.__version__} cuda={torch.version.cuda} device={dev} bf16_supported={torch.cuda.is_bf16_supported() if dev=='cuda' else False}")

def main():
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)
    params = raw_cfg.get("parameters", {})
    is_sweep = any(isinstance(param, dict) and "values" in param for param in params.values())
    val = params.get("model_dir", "../models/")
    if not isinstance(val, dict): params["model_dir"] = {"value": str(val)}
    params["model_dir"]["value"] = "../models/"
    if is_sweep:
        sweep_config = {
            "name": read_config_parameter(raw_cfg, "name") if not raw_cfg.get("debug") else "debug_" + random.randbytes(4).hex(),
            "method": read_config_parameter(raw_cfg, "method"),
            "metric": read_config_parameter(raw_cfg, "metric"),
            "parameters": read_config_parameter(raw_cfg, "parameters"),
            "project": read_config_parameter(raw_cfg, "project"),
            "description": " ".join(get_sweep_parameter_names(raw_cfg)),
        }
        project = sweep_config["project"] if not read_config_parameter(raw_cfg,"debug") else "_debug_" + random.randbytes(4).hex()
        sweep_dir = sweep_config["parameters"]["model_dir"]["value"] + project
        if not os.path.exists(sweep_dir): os.makedirs(sweep_dir, exist_ok=True)
        ensure_free_disk_space(sweep_dir)
        sweep_config["parameters"]["sweep_dir"] = {"value": sweep_dir}
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
            # overlay parameter values (resolve {"value": ...} into plain values)
            for k, v in sweep_config["parameters"].items():
                tmp_cfg[k] = v["value"] if isinstance(v, dict) and "value" in v else v
            tmp_cfg.update(get_dataset_cfg(tmp_cfg))
            gene_chunks = _prepare_gene_list(tmp_cfg)  # flat list or list-of-lists
            if gene_chunks: # is None if specific gene list was provided in config, e.g. not chunking all genes
            # normalize to list-of-lists so the sweep can iterate values
                if isinstance(gene_chunks[0], str):
                    gene_chunks = [gene_chunks]

                # inject as sweep param
                sweep_config["parameters"]["genes"] = {"values": gene_chunks}

                sweep_config["parameters"]["gene_chunks"] = {"value": gene_chunks}
            else:
                g = tmp_cfg["genes"]
                sweep_config["parameters"]["genes"] = (
                    {"values": g} if g and isinstance(g[0], list) else {"values": [g]}
                )

            swept_names = [
                k for k, v in sweep_config["parameters"].items()
                if isinstance(v, dict) and ("values" in v or "distribution" in v)
            ]
            swept_names.append("genes_id")
            if "genes" in swept_names:
                swept_names.remove("genes")
            sweep_config["parameters"]["sweep_parameter_names"] = {"value": swept_names}

            # prevent double splitting in runs
            sweep_config["parameters"].pop("split_genes_by", None)
            sweep_config["method"] = sweep_config.get("method") or "grid"

            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Initialized new sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
        for k, p in params.items():
            if isinstance(p, dict) and "value" in p: cfg[k] = p["value"]

        project = cfg["project"] if not read_config_parameter(raw_cfg, "debug") else "debug_" + random.randbytes(
            4).hex()
        out_dir = cfg["model_dir"] + project
        os.makedirs(out_dir, exist_ok=True);
        ensure_free_disk_space(out_dir)
        cfg["out_path"] = out_dir

        _train(cfg)

if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    main()