import argparse
import shutil
import os
import tempfile
from pathlib import Path
from typing import Dict


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
