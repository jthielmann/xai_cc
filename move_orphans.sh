#!/usr/bin/env bash
set -euo pipefail

csv="models_overview_full.csv"
models_root="models"
legacy_root="${models_root}/test_runs"

[[ -f "$csv" ]]
mkdir -p "$legacy_root"

python3 - "$csv" "$models_root" "$legacy_root" <<'PY'
import csv, pathlib, shutil, sys, yaml
csv_path, models_root, legacy_root = sys.argv[1:4]
projects = { (r.get("project") or "").strip()
             for r in csv.DictReader(open(csv_path), delimiter=";")
             if (r.get("project") or "").strip() }
moved = 0
for cfg in pathlib.Path(models_root).rglob("config"):
    if legacy_root in cfg.as_posix():
        continue
    try:
        proj = (yaml.safe_load(cfg.read_text()) or {}).get("project")
    except Exception:
        continue
    proj = (proj or "").strip()
    if not proj or proj in projects:
        continue
    run_dir = cfg.parent
    dest = pathlib.Path(legacy_root) / run_dir.name
    i = 1
    while dest.exists():
        dest = pathlib.Path(legacy_root) / f"{run_dir.name}_{i}"
        i += 1
    shutil.move(run_dir, dest)
    moved += 1
print(f"moved {moved} run(s)")
PY
