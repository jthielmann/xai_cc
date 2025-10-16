import argparse
import shutil
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Iterable
import os
import yaml


def setup_dump_env() -> str:
    dd = "../dump"
    os.makedirs(dd, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", dd)
    os.environ.setdefault("WANDB_CACHE_DIR", dd)
    os.environ.setdefault("WANDB_CONFIG_DIR", dd)
    os.environ.setdefault("TORCH_HOME", dd)
    os.environ.setdefault("MPLCONFIGDIR", dd)
    os.environ.setdefault("HF_HOME", dd)
    os.environ.setdefault("TRANSFORMERS_CACHE", dd)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", dd)
    return dd

def ensure_free_disk_space(path: str, min_gb: int = 20) -> None:
    p = Path(path).resolve()
    total, used, free = shutil.disk_usage(p)
    if free < min_gb * 1024**3:
        raise RuntimeError(f"Only {free/1024**3:.2f} GB free at {str(p)!r}; need ≥{min_gb} GB.")


def parse_yaml_config(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
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
    """Support reading from either top-level or W&B-style 'parameters' section."""
    if parameter in config:
        val = config[parameter]
        if isinstance(val, dict):
            if "value" in val:
                return val["value"]
            if "values" in val:
                return val["values"]
        return val
    params = config.get("parameters", {}) or {}
    if parameter in params:
        val = params[parameter]
        if isinstance(val, dict):
            if "value" in val:
                return val["value"]
            if "values" in val:
                return val["values"]
        return val
    return None


def get_sweep_parameter_names(config: dict) -> list[str]:
    names: list[str] = []
    for name, param in (config.get("parameters", {}) or {}).items():
        if isinstance(param, dict) and "values" in param:
            names.append(name)
    return names


_KEY_ALIASES: Dict[str, str] = {
    # common training
    "learning_rate": "lr",
    "lr": "lr",
    "weight_decay": "wd",
    "epochs": "ep",
    "batch_size": "bs",
    "optimizer": "opt",
    "scheduler": "sch",
    "dataset": "ds",
    "encoder_type": "enc",
    "encoder_out_dim": "eod",
    "encoder_finetune_layers": "efl",
    "encoder_finetune_layer_names": "efln",
    "middle_layer_features": "ml",
    "image_size": "sz",
    "num_workers": "nw",
    "loss_fn_switch": "loss",
    "freeze_encoder": "fe",
    "one_linear_out_layer": "1l",
    "gene_data_filename": "gdf",
    "bins": "b",
    "genes": "g",
}


def _abbr_key(key: str) -> str:
    if key in _KEY_ALIASES:
        return _KEY_ALIASES[key]
    parts = [p for p in str(key).replace("-", "_").split("_") if p]
    if not parts:
        return str(key)[:3]
    if len(parts) == 1:
        return parts[0][:3]
    return "".join(p[0] for p in parts[:3])


def _abbr_value(val: Any) -> str:
    return str(val)[:20]


def make_run_name_from_config(cfg: Dict[str, Any], param_names: Iterable[str]) -> str:
    keys = list(dict.fromkeys(param_names))
    tokens = []
    for k in keys:
        v = read_config_parameter(cfg, k)
        if v == None:
            raise RuntimeError(f"could not read {k} from {cfg}")
        ak = _abbr_key(k)
        av = _abbr_value(v)
        tokens.append(f"{ak}={av}")
    if not tokens:
        raise RuntimeError("could not infer tokens for make_run_name_from_config")
    name = "-".join(tokens)
    return name[:128]
