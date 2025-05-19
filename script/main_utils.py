import argparse
import shutil
from typing import Dict


def ensure_free_disk_space(path: str, min_gb: int = 20) -> None:
    total, used, free = shutil.disk_usage(path)
    if free < min_gb * 1024**3:
        raise RuntimeError(
            f"Only {free/1024**3:.2f} GB free at {path!r}; need ≥{min_gb} GB."
        )


def parse_yaml_config(path: str) -> Dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


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

