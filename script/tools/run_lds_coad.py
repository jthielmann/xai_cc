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

    # Delegate to script.data_processing.lds by mimicking its CLI invocation.
    sys.argv = ["lds.py", "--config", str(cfg_path)]
    from script.data_processing import lds  # late import to avoid side effects when unused

    lds.main()


if __name__ == "__main__":
    main()

