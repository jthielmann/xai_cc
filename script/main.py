import csv, sys, os, random, numpy, torch, yaml, pandas as pd, wandb
from typing import Dict, Any
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

def _prepare_gene_list(cfg: Dict[str, Any]) -> None:
    if cfg.get("genes") is None:
        pts = cfg["train_samples"] + cfg["val_samples"] + cfg.get("test_samples", [])
        df0 = pd.read_csv(os.path.join(cfg["data_dir"], pts[0], "meta_data", cfg["gene_data_filename"]), nrows=1)
        cand = [c for c in df0 if c != "tile" and not str(c).endswith("_lds_w") and pd.api.types.is_numeric_dtype(df0[c])]
        g = set(cand)
        for p in pts[1:]:
            df = pd.read_csv(os.path.join(cfg["data_dir"], p, "meta_data", cfg["gene_data_filename"]), nrows=1)
            g &= set(df.columns)

        k = cfg.get("split_genes_by", None)
        gene_list = [c for c in cand if c in g]
        if not k:
            cfg["genes"] = gene_list
        else:
            cfg["genes"] = [gene_list[i:i + k] for i in range(0, len(gene_list), k)]

def _train(cfg: Dict[str, Any]) -> None:
    run = wandb.init(project=cfg.get("project","xai"), config=cfg) if cfg.get("log_to_wandb", False) else None
    cfg = _prepare_cfg(dict(run.config) if run else cfg); TrainerPipeline(cfg, run=run).run()
    if run: run.finish()

def _sweep_run():
    run = wandb.init(); cfg = _prepare_cfg(dict(run.config))
    TrainerPipeline(cfg, run=run).run(); run.finish()

def main():
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)
    params = raw_cfg.get("parameters", {})
    is_sweep = any(isinstance(param, dict) and "values" in param for param in params.values())
    val = params.get("model_dir", "../models/")
    if not isinstance(val, dict): params["model_dir"] = {"value": str(val)}
    params["model_dir"]["value"] = "../models/"
    if is_sweep:
        params["sweep_parameter_names"] = {"values": [get_sweep_parameter_names(raw_cfg)]}
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
            _prepare_gene_list(sweep_config)
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f: f.write(sweep_id)
            print(f"Initialized new sweep ID: {sweep_id}")
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
        for k, p in params.items():
            if isinstance(p, dict) and "value" in p: cfg[k] = p["value"]
        project = cfg["project"] if not read_config_parameter(raw_cfg,"debug") else "debug_" + random.randbytes(4).hex()
        out_dir = cfg["model_dir"] + project
        os.makedirs(out_dir, exist_ok=True); ensure_free_disk_space(out_dir)
        cfg["out_path"] = out_dir
        _train(cfg)

if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    main()