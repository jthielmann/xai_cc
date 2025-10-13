#!/usr/bin/env python3
"""Convenience wrapper to generate LDS weights for the COAD dataset.

This script resolves the default LDS sweep config (extension-less file
`sweeps/configs/lds_coad`) relative to the repository root and forwards it to
`script.data_processing.lds`. Use it as:

    python3 -m script.tools.run_lds_coad

Optionally pass `--config` to point at a different config file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tempfile

import yaml

from script.configs.dataset_config import get_dataset_cfg
from script.main import _prepare_gene_list
from script.main_utils import parse_yaml_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config() -> Path:
    return _repo_root() / "sweeps" / "configs" / "lds_coad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LDS weight generation for COAD.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_default_config()),
        help="Path to the LDS config file (extension-less YAML).",
    )
    return parser.parse_args()


def _resolve_scalar(entry):
    if isinstance(entry, dict):
        if "value" in entry:
            return entry["value"]
        if "values" in entry and entry["values"]:
            return entry["values"][0]
    return entry


def _ensure_genes(cfg_path: Path) -> Path:
    cfg = parse_yaml_config(str(cfg_path))
    params = cfg.setdefault("parameters", {})
    if "genes" in params and params["genes"]:
        return cfg_path

    dataset_name = _resolve_scalar(params.get("dataset"))
    if not dataset_name:
        raise ValueError("LDS config must specify 'dataset'.")
    gene_data_filename = _resolve_scalar(params.get("gene_data_filename")) or "gene_data.csv"
    meta_data_dir = _resolve_scalar(params.get("meta_data_dir")) or "meta_data"

    trainer_cfg = {
        "dataset": dataset_name,
        "gene_data_filename": gene_data_filename,
        "meta_data_dir": meta_data_dir,
    }
    trainer_cfg.update(get_dataset_cfg(trainer_cfg))
    genes = _prepare_gene_list(trainer_cfg)
    if not genes:
        raise RuntimeError(f"Failed to infer any genes for dataset '{dataset_name}'.")

    params["genes"] = {"values": genes}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        # Try resolving relative to repo root if the provided value is relative.
        alt_path = (_repo_root() / args.config).resolve()
        if alt_path.is_file():
            cfg_path = alt_path
        else:
            raise FileNotFoundError(f"Config file not found: {args.config!r}")

    orig_cfg_path = cfg_path
    cfg_path_with_genes = _ensure_genes(cfg_path)
    cleanup = cfg_path_with_genes != orig_cfg_path

    # Delegate to script.data_processing.lds by mimicking its CLI invocation.
    sys.argv = ["lds.py", "--config", str(cfg_path_with_genes)]
    from script.data_processing import lds  # late import to avoid side effects when unused

    lds.main()
    if cleanup:
        try:
            Path(cfg_path_with_genes).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
