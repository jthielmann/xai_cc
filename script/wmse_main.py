import os
import sys

# Make numba avoid OpenMP/TBB to prevent clashes with PyTorch/MKL on HPC
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
# Be conservative with thread pools by default (can be overridden by user env)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, "..")

from typing import Any, Dict, List

import pandas as pd

from script.main_utils import (
    parse_args,
    parse_yaml_config,
    read_config_parameter,
    ensure_free_disk_space,
    setup_dump_env,
)
from script.data_processing.data_loader import label_dataset
from script.data_processing.lds import LDS, grid_search_lds
from script.configs.dataset_config import get_dataset_cfg


def _require(cfg: Dict[str, Any], key: str):
    val = read_config_parameter(cfg, key)
    if val is None:
        raise ValueError(f"Missing required parameter '{key}' in config")
    return val


def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
    v = _require(cfg, key)
    if isinstance(v, (list, tuple)):
        return list(v)
    # Allow scalar -> list with a single element only if explicitly given as a scalar
    return [v]

def _has_key(cfg: Dict[str, Any], key: str) -> bool:
    return key in cfg and cfg[key] is not None


def main() -> None:
    args = parse_args()
    raw_cfg = parse_yaml_config(args.config)

    setup_dump_env()

    # Required config parameters (no fallbacks for values; derive splits from dataset config)
    dataset_name = _require(raw_cfg, "dataset")
    gene_data_filename = _require(raw_cfg, "gene_data_filename")
    meta_data_dir = _require(raw_cfg, "meta_data_dir")
    bin_space = _require_list(raw_cfg, "bin_space")
    ks_space = _require_list(raw_cfg, "ks_space")
    sigma_space = _require_list(raw_cfg, "sigma_space")
    kernel_type = _require(raw_cfg, "kernel_type")
    weight_dir = _require(raw_cfg, "weight_dir")

    # Resolve dataset paths and splits from dataset config
    ds_cfg = get_dataset_cfg({
        "dataset": dataset_name,
        "debug": bool(read_config_parameter(raw_cfg, "debug")),
    })
    data_dir = ds_cfg["data_dir"]
    train_samples = ds_cfg["train_samples"]

    # Prepare output directory strictly as provided
    os.makedirs(weight_dir, exist_ok=True)
    ensure_free_disk_space(weight_dir)
    out_csv = os.path.join(weight_dir, "best_smoothings.csv")

    # Build label dataset strictly from config
    ds = label_dataset(
        data_dir=data_dir,
        samples=train_samples,
        gene_data_filename=str(gene_data_filename),
        meta_data_dir=str(meta_data_dir),
        max_len=None,
    )

    # Determine genes from dataset automatically; disallow providing 'genes' in config to avoid confusion
    if _has_key(raw_cfg, "genes"):
        raise ValueError("Do not set 'genes' in WMSE config; all genes in the dataset will be used.")

    # Auto-detect gene columns: take all numeric columns from the label dataframe
    df = ds.dataframe
    import pandas as pd  # local import to avoid global dependency at module import
    auto_genes = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not auto_genes:
        raise RuntimeError("No numeric gene columns found in the dataset for WMSE weight generation.")

    # Grid search LDS smoothing and persist results
    lds = LDS(kernel_type=str(kernel_type), dataset=ds)
    df = grid_search_lds(lds, list(auto_genes), list(bin_space), list(ks_space), list(sigma_space))
    df_out = df.drop(columns=["weights"]).copy() if "weights" in df.columns else df
    df_out.to_csv(out_csv, index=False)

    print(f"Saved LDS/WMSE weights to: {out_csv}")


if __name__ == "__main__":
    main()
