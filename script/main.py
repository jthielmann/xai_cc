import sys
sys.path.insert(0, '..')

import wandb
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
import os
import random

import numpy
import torch
import yaml
import sys
from main_utils import ensure_free_disk_space, parse_args, parse_yaml_config, read_config_parameter, get_sweep_parameter_names
from typing import Dict, Any


def _train(cfg: Dict[str, Any]) -> None:
    ds_cfg = get_dataset_cfg(cfg)
    cfg.update(ds_cfg)

    # holds run variable in ca
    run = None

    use_wandb = cfg.get("log_to_wandb", False)
    if use_wandb:
        run = wandb.init(
            project=cfg.get("project", "xai"),
            config=cfg
        )
        cfg_for_pipeline = dict(run.config)
    else:
        cfg_for_pipeline = cfg
    with open(cfg_for_pipeline["out_path"] + "/config", "w") as f:
        yaml.safe_dump(cfg_for_pipeline, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    pipeline = TrainerPipeline(cfg_for_pipeline, run=run)
    pipeline.run()

    if run is not None:
        run.finish()


def _sweep_run():
    run = wandb.init()

    cfg = dict(run.config)

    ds_cfg = get_dataset_cfg(cfg)
    cfg.update(ds_cfg)
    pipeline = TrainerPipeline(cfg, run=run)
    pipeline.run()
    run.finish()


def main():
    args = parse_args()

    # Load raw YAML (parameters block may contain 'value' or 'values')
    raw_cfg = parse_yaml_config(args.config)

    params = raw_cfg.get("parameters", {})
    # Detect if any parameter defines multiple 'values'
    is_sweep = any(isinstance(param, dict) and "values" in param for param in params.values())

    if is_sweep:
        # Build sweep config automatically based on 'values'
        params["sweep_parameter_names"] = {"values": [get_sweep_parameter_names(raw_cfg)]}
        sweep_config = {
            "name": read_config_parameter(raw_cfg, "name") if not raw_cfg.get("debug")
            else "debug_" + random.randbytes(4).hex(),
            "method": read_config_parameter(raw_cfg, "method"),
            "metric": read_config_parameter(raw_cfg, "metric"),
            "parameters": read_config_parameter(raw_cfg, "parameters"),
            "project": read_config_parameter(raw_cfg, "project"),
            "description": " ".join(get_sweep_parameter_names(raw_cfg))
        }

        # need the target location to exist to check if there is enough space
        project = sweep_config["project"] if not read_config_parameter(raw_cfg,"debug") else "debug_" + random.randbytes(4).hex()
        sweep_dir = "../models/" + project
        if not os.path.exists(sweep_dir):
            os.makedirs(sweep_dir, exist_ok=True)
        ensure_free_disk_space(sweep_dir)
        sweep_config["parameters"]["sweep_dir"] = {"value": sweep_dir}

        print(f"Project: {project}")
        sweep_id_dir = os.path.join("..", "wandb_sweep_ids", project, sweep_config["name"])
        os.makedirs(sweep_id_dir, exist_ok=True)
        sweep_id_file = os.path.join(sweep_id_dir, "sweep_id.txt")

        # Load or create sweep ID
        if os.path.exists(sweep_id_file):
            with open(sweep_id_file, "r") as f:
                sweep_id = f.read().strip()
            print(f"Loaded existing sweep ID: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project=project)
            with open(sweep_id_file, "w") as f:
                f.write(sweep_id)
            print(f"Initialized new sweep ID: {sweep_id}")

        # Launch agent for the sweep
        wandb.agent(sweep_id, function=_sweep_run, project=project)
    else:
        # Single-run: flatten each parameter 'value' into cfg dict
        cfg = {k: v for k, v in raw_cfg.items() if k != "parameters"}
        for key, param in params.items():
            if isinstance(param, dict) and "value" in param:
                cfg[key] = param["value"]
        ensure_free_disk_space(cfg.get("out_path"))
        _train(cfg)


if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    main()