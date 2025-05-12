import argparse
import wandb
import yaml
import sys

sys.path.insert(0, '..')
from script.configs.dataset_config import get_dataset_cfg
from script.train.lit_train import TrainerPipeline
import os
import random

import numpy
import torch


def parse_args():
    """
    Parse the path to a YAML config file from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run a single training job or a W&B sweep from a YAML config."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the YAML config file defining parameters or sweeps"
    )
    return parser.parse_args()


def _train(raw_cfg: dict):
    run = wandb.init(
        project=raw_cfg.get("project", "xai"),
        config={**raw_cfg},
        mode="online" if raw_cfg.get("log_to_wandb", False) else "disabled"
    )
    # merge ds_cfg if needed…
    pipeline = TrainerPipeline(dict(run.config), run=run)
    pipeline.run()


def _sweep_run():
    # (reinit=True can help if this function gets called multiple times
    #  in the same process, but it's optional)
    run = wandb.init()

    # now wandb.config is populated, so pull it into a plain dict:
    cfg = dict(run.config)

    # merge in any dataset defaults
    ds_cfg = get_dataset_cfg(cfg["dataset"], cfg.get("debug", False))
    cfg.update(ds_cfg)

    # run your training
    pipeline = TrainerPipeline(cfg, run=run)
    pipeline.run()


def read_config_parameter(config: dict, parameter: str):
    if parameter in config:
        return config[parameter]
    if parameter in config["parameters"]:
        param = config["parameters"][parameter]
        if isinstance(param, dict) and "value" in param:
            return param["value"]
        if isinstance(param, dict) and "values" in param:
            return param["values"]
    raise ValueError(f"Parameter '{parameter}' not found in config.")


def get_sweep_parameter_names(config: dict) -> list[str]:
    return [
        name
        for name, param in config.get("parameters", {}).items()
        if isinstance(param, dict) and "values" in param
    ]


def main():
    args = parse_args()

    # Load raw YAML (parameters block may contain 'value' or 'values')
    with open(args.config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    params = raw_cfg.get("parameters", {})
    # Detect if any parameter defines multiple 'values'
    is_sweep = any(isinstance(p, dict) and "values" in p for p in params.values())

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
        # Determine project and sweep directory
        project = sweep_config["project"] if not read_config_parameter(raw_cfg,"debug") else "debug_" + random.randbytes(4).hex()

        print(f"Project: {project}")
        sweep_dir = os.path.join("..", "wandb_sweep_ids", project, sweep_config["name"])
        os.makedirs(sweep_dir, exist_ok=True)
        sweep_id_file = os.path.join(sweep_dir, "sweep_id.txt")

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
        _train(cfg)


if __name__ == "__main__":
    print("python version:", sys.version)
    print("numpy version:", numpy.version.version)
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    main()