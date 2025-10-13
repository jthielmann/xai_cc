import argparse
import shutil
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Iterable
import os


def setup_dump_env() -> str:
    """Configure env vars so incidental outputs go under a single dump dir."""
    dd = Path("../dump")
    os.makedirs(dd, exist_ok=True)

    # W&B local dirs (run files and cache)
    os.environ.setdefault("WANDB_DIR", str(dd / "../dump"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(dd / "../dump"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(dd / "../dump"))
    # Torch / torchvision cache (pretrained weights, etc.)
    os.environ.setdefault("TORCH_HOME", str(dd / "../dump"))
    # Matplotlib cache
    os.environ.setdefault("MPLCONFIGDIR", str(dd / "../dump"))
    # Common ML caches (harmless if unused)
    os.environ.setdefault("HF_HOME", str(dd / "../dump"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(dd / "../dump"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(dd / "../dump"))
    return str(dd)

def ensure_free_disk_space(path: str, min_gb: int = 20) -> None:
    """Best-effort check that there is enough space and inodes, and that the
    current user can actually create a file at the target path.

    Notes:
    - Filesystem free space checks do not account for user quotas (EDQUOT).
      We also do a tiny probe write to detect EDQUOT early.
    - Symlinks are resolved to avoid checking a different mount than the target.
    """
    p = Path(path).resolve()
    # 1) Byte capacity (filesystem-level)
    total, used, free = shutil.disk_usage(p)
    if free < min_gb * 1024**3:
        raise RuntimeError(
            f"Only {free/1024**3:.2f} GB free at {str(p)!r}; need ≥{min_gb} GB."
        )
    # 2) Inodes (filesystem-level; may still not reflect per-user quotas)
    try:
        st = os.statvfs(p)
        free_inodes = getattr(st, "f_favail", 0)
        if free_inodes is not None and free_inodes <= 0:
            raise RuntimeError(f"No free inodes available at {str(p)!r} (inode quota reached).")
    except Exception:
        # statvfs may not be available/accurate on some systems; ignore softly
        pass
    # 3) Probe write to detect EDQUOT/ENOSPC on the actual target dir
    try:
        with tempfile.NamedTemporaryFile(dir=str(p), prefix=".__quota_probe_", delete=True) as tf:
            tf.write(b"x")
            tf.flush()
            os.fsync(tf.fileno())
    except OSError as e:
        # 28: ENOSPC (no space), 122: EDQUOT (quota exceeded)
        if getattr(e, "errno", None) in (28, 122):
            raise RuntimeError(
                f"Cannot write to {str(p)!r}: {e.strerror} (errno {e.errno}). "
                f"Filesystem may have space, but your user quota or inode quota is exhausted."
            ) from e
        raise


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


# --- Naming helpers ---------------------------------------------------------

_KEY_ALIASES: Dict[str, str] = {
    # common training
    "learning_rate": "lr",
    "lr": "lr",
    "weight_decay": "wd",
    "epochs": "ep",
    "batch_size": "bs",
    "optimizer": "opt",
    "scheduler": "sch",
    # data/model
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
    # fallback: take first chars of snake-case parts, max 3 parts
    parts = [p for p in str(key).replace("-", "_").split("_") if p]
    if not parts:
        return str(key)[:3]
    if len(parts) == 1:
        return parts[0][:3]
    return "".join(p[0] for p in parts[:3])


def _abbr_value(val: Any, key: str = "") -> str:
    # lists: summarize by length (esp. genes)
    if isinstance(val, (list, tuple)):
        try:
            n = len(val)
        except Exception:
            n = 0
        return f"n{n}"
    # booleans
    if isinstance(val, bool):
        return "t" if val else "f"
    # ints
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    # floats
    if isinstance(val, float):
        # concise formatting; keep significant digits without trailing zeros
        s = f"{val:g}"
        return s
    # strings (filenames → stem; compact common gene_data_* patterns)
    if isinstance(val, str):
        stem = os.path.splitext(os.path.basename(val))[0]
        stem = stem.replace("gene_data_", "")
        return stem[:20]
    # fallback
    return str(val)[:20]


def make_run_name_from_config(cfg: Dict[str, Any], param_names: Iterable[str]) -> str:
    """Build a compact run name from chosen hyperparameters.

    Only includes parameters that are part of the sweep (param_names).
    Produces tokens like "lr=0.01-bs=32-gdf=ranknorm".
    """
    keys = list(dict.fromkeys(param_names))  # stable unique
    tokens = []
    for k in keys:
        if k in cfg:
            v = cfg[k]
            ak = _abbr_key(k)
            av = _abbr_value(v, key=k)
            tokens.append(f"{ak}={av}")
    name = "-".join(tokens) if tokens else "auto"
    # hard limit to keep names manageable
    return name[:128]


def make_sweep_name_from_space(config: Dict[str, Any]) -> str:
    """Build a generic sweep name from the hyperparameter space (keys only)."""
    keys = get_sweep_parameter_names(config)
    if not keys:
        return "sweep-auto"
    toks = [_abbr_key(k) for k in keys]
    base = "swp-" + "+".join(toks)
    return base[:64]
