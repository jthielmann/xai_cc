#!/usr/bin/env python3
import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _normalize_grid(grid: Dict[str, Any]) -> List[Tuple[str, List[Any]]]:
    items: List[Tuple[str, List[Any]]] = []
    for k, v in grid.items():
        if isinstance(v, list):
            items.append((k, v))
        else:
            items.append((k, [v]))
    return items


def _product(grid: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    items = _normalize_grid(grid)
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def write_configs(
    base_config: Dict[str, Any],
    grid: Dict[str, Any],
    out_dir: Path,
    tag: str,
    dry_run: bool = False,
) -> List[Path]:
    out_dir = out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, params in enumerate(_product(grid)):
        cfg = json.loads(json.dumps(base_config))  # deep copy via json
        # shallow merge: top-level keys from grid override base
        cfg.update(params)
        run_dir = out_dir / f"{i:04d}"
        paths.append(run_dir / "config.json")
        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)

    if not dry_run:
        with (out_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump({"count": len(paths), "paths": [str(p) for p in paths]}, f, indent=2)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate sweep configs (JSON only)")
    ap.add_argument("--grid", required=True, help="Path to JSON grid file {param: [values], ...}")
    ap.add_argument("--base-config", help="Optional JSON base config to start from")
    ap.add_argument("--tag", required=True, help="Sweep tag (directory under out/sweeps)")
    ap.add_argument("--out-dir", default="out/sweeps", help="Output directory root")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files, just print summary")
    args = ap.parse_args(argv)

    grid_path = Path(args.grid)
    with grid_path.open("r", encoding="utf-8") as f:
        grid = json.load(f)

    base_cfg: Dict[str, Any] = {}
    if args.base_config:
        with Path(args.base_config).open("r", encoding="utf-8") as f:
            base_cfg = json.load(f)

    out_dir = Path(args.out_dir)
    paths = write_configs(base_cfg, grid, out_dir, args.tag, dry_run=args.dry_run)

    print(f"Generated {len(paths)} configs under {out_dir / args.tag}")
    if args.dry_run:
        for p in paths[:10]:
            print(f" - {p}")
        if len(paths) > 10:
            print(" - ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

