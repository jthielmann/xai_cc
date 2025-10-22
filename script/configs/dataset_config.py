import os
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

_BASE_COAD: Dict[str, Any] = {
    "data_dir": "/data/cephfs-2/unmirrored/groups/krieger/xai/HEST/hest_coad_visium",
    "train_samples_all": [
         'MISC62', 'ZEN45', 'TENX152', 'MISC73', 'MISC71', 'MISC70', 'MISC68', 'MISC67', 'MISC66', 'MISC65', 'MISC64',
         'MISC63', 'MISC58', 'MISC57', 'MISC51', 'MISC50', 'MISC49', 'MISC48', 'MISC47', 'MISC46', 'MISC44', 'MISC43',
         'MISC41', 'MISC40', 'MISC39', 'MISC38', 'MISC36', 'TENX92', 'TENX91', 'TENX49', 'ZEN47', 'ZEN46', 'ZEN43',
         'ZEN39', 'MISC42'],
    "val_samples_all": ['MISC69', 'MISC56', 'MISC37', 'MISC35', 'MISC34', 'TENX90', 'TENX29', 'ZEN42', 'ZEN38'],
    "test_samples_all": ['MISC72', 'MISC45', 'MISC33', 'TENX89', 'TENX28', 'ZEN44']
}


# Consolidated dataset configurations
DATASETS: Dict[str, Dict[str, Any]] = {
    "CRC_N19": {
        **_BASE_CRC_N19,
        "weights": None,
    },
    "CRC-N19_2": {
        **_BASE_CRC_N19,
        "weights": None,
    },
    "coad": {
        **_BASE_COAD,
        "weights": None,
    },
    "crc_base": {
        **_BASE_CRC_BASE,
        "weights": None,
    },
    "pseudospot": {
        **_BASE_CRC_BASE,
        "weights": None,
    },
}


def get_dataset_cfg(cfg: dict) -> Dict[str, Any]:
    name = cfg["dataset"]
    if type(name) == dict:
        name = name.get("value")
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
        train_all: List[str] = list(cfg["train_samples"])  # type: ignore[index]
        val_all:   List[str] = list(cfg["val_samples"])    # type: ignore[index]
    else:  # neither key present – fall back to defaults in DATASETS
        train_all = list(ds.pop("train_samples_all"))
        val_all   = list(ds.pop("val_samples_all"))

    ds["train_samples"] = train_all[:1] if debug else train_all
    ds["val_samples"]   = val_all[:1]   if debug else val_all

    # Derive test samples
    if "test_samples" in cfg:
        test_all: List[str] = list(cfg["test_samples"])  # explicit override
    elif "test_samples_all" in ds:
        test_all = list(ds.pop("test_samples_all"))
    else:
        # Compute as all patient dirs minus train+val
        data_dir = ds["data_dir"]
        all_dirs = [
            f.name for f in os.scandir(data_dir)
            if f.is_dir() and not f.name.startswith((".", "_"))
        ]
        exclude = set(train_all) | set(val_all)
        test_all = sorted([d for d in all_dirs if d not in exclude])

    ds["test_samples"] = test_all[:1] if debug else test_all

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
