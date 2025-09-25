import yaml
from typing import Any, Dict, List

# Load a YAML configuration file from the given path
def load_user_config(path: str) -> Dict[str, Any]:
        return yaml.safe_load(f)


# takes the output of lds and returns the weights
def get_weights_from_smoothed(smoothed_labels: List[float]) -> List[float]:
    eps, alpha = 1e-4, 0.5
    # for each smoothed count, add eps then raise to alpha and invert
    weights = [
        1.0 / ((count + eps) ** alpha)
        for count in smoothed_labels
    ]
    return weights


def get_wmse_weights(dataset_name: str) -> List[float]:
    if dataset_name == "CRC_N19":
        smoothed_labels = [

        ]
    elif dataset_name == "CRC_N19_2":
        smoothed_labels = [

        ]
    elif dataset_name == "crc_base":
        smoothed_labels = [

        ]
    elif dataset_name == "pseudospot":
        smoothed_labels = [

        ]
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    return get_weights_from_smoothed(smoothed_labels)


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
        "weights": get_wmse_weights("CRC_N19"),
    },
    "CRC-N19_2": {
        **_BASE_CRC_N19,
        "weights": get_wmse_weights("CRC_N19_2"),
    },
    "crc_base": {
        **_BASE_CRC_BASE,
        "weights": get_wmse_weights("crc_base"),
    },
    "pseudospot": {
        **_BASE_CRC_BASE,
        "weights": get_wmse_weights("pseudospot"),
    },
}


def get_dataset_cfg(name: str, debug: bool) -> Dict[str, Any]:
    """
    Retrieve and prepare dataset configuration by name. If `debug` is True,
    only the first train/val sample is selected; otherwise, all.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'")

    ds = DATASETS[name].copy()
    train_all: List[str] = ds.pop("train_samples_all")
    val_all:   List[str] = ds.pop("val_samples_all")

    ds["train_samples"] = train_all[:1] if debug else train_all
    ds["val_samples"]   = val_all[:1]   if debug else val_all

    return ds
