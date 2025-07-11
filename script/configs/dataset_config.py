import yaml
from typing import Any, Dict, List


def load_user_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the given path.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Base dataset specifications for reuse
_BASE_CRC_N19: Dict[str, Any] = {
    "data_dir": "../data/CRC-N19/",
    "train_samples_all": [
        "TENX92","TENX91","TENX90","TENX89","TENX70","TENX49",
        "ZEN49","ZEN48","ZEN47","ZEN46","ZEN45","ZEN44"
    ],
    "val_samples_all": [
        "TENX29","ZEN43","ZEN42","ZEN40","ZEN39","ZEN38","ZEN36"
    ],
}

_BASE_CRC_BASE: Dict[str, Any] = {
    "data_dir": "../data/crc_base/Training_Data/",
    "train_samples_all": ["p007","p014","p016","p020","p025"],
    "val_samples_all":   ["p009","p013"],
}

# Consolidated dataset configurations
DATASETS: Dict[str, Dict[str, Any]] = {
    "CRC_N19": {
        **_BASE_CRC_N19,
        "mean": [0.0555, 0.1002, 0.00617],
        "std":  [0.991,  0.9826, 0.9967],
        "weights": None,
    },
    "CRC-N19_2": {
        **_BASE_CRC_N19,
        "mean": [0.5405, 0.2749, 0.5476],
        "std":  [0.2619, 0.2484, 0.2495],
        "weights": None,
    },
    "crc_base": {
        **_BASE_CRC_BASE,
        "mean": [0.331,  0.632,  0.3946],
        "std":  [1.1156, 1.1552, 1.1266],
        "weights": None,
    },
    "pseudospot": {
        **_BASE_CRC_BASE,
        "mean": [0.331,  0.632,  0.3946],
        "std":  [1.1156, 1.1552, 1.1266],
        "weights": None,
    },
}


def get_dataset_cfg(cfg: dict) -> Dict[str, Any]:
    name = cfg["dataset"]
    debug = cfg.get("debug", False)

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")

    ds = DATASETS[name].copy()

    has_train = "train_samples" in cfg
    has_val   = "val_samples"   in cfg

    # If only one of the two keys is given, that’s a mis-configuration.
    if has_train ^ has_val:
        raise AttributeError(
            "Both 'train_samples' and 'val_samples' must be provided together in cfg."
        )

    if has_train and has_val:
        train_all: List[str] = cfg["train_samples"]
        val_all:   List[str] = cfg["val_samples"]
    else:  # neither key present – fall back to defaults in DATASETS
        train_all = ds.pop("train_samples_all")
        val_all   = ds.pop("val_samples_all")

    ds["train_samples"] = train_all[:1] if debug else train_all
    ds["val_samples"]   = val_all[:1]   if debug else val_all

    return ds

def get_dataset_data_dir(name: str) -> str:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")

    return DATASETS[name]["data_dir"]


def get_dataset_cfg_lds(cfg: dict) -> Dict[str, Any]:
    name = cfg["dataset"].get("value")
    debug = cfg.get("debug", False)

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")

    ds = DATASETS[name].copy()

    has_train = "train_samples" in cfg
    has_val   = "val_samples"   in cfg

    # If only one of the two keys is given, that’s a mis-configuration.
    if has_train ^ has_val:
        raise AttributeError(
            "Both 'train_samples' and 'val_samples' must be provided together in cfg."
        )

    if has_train and has_val:
        train_all: List[str] = cfg["train_samples"]
        val_all:   List[str] = cfg["val_samples"]
    else:  # neither key present – fall back to defaults in DATASETS
        train_all = ds.pop("train_samples_all")
        val_all   = ds.pop("val_samples_all")

    ds["train_samples"] = train_all[:1] if debug else train_all
    ds["val_samples"]   = val_all[:1]   if debug else val_all

    return ds
