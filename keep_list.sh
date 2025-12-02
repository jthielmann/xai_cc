#!/usr/bin/env bash
set -euo pipefail

csv="models_overview_full.csv"
models_root="models"
legacy_root="${models_root}/test_runs"

[[ -f "$csv" ]]

python3 - "$csv" "$models_root" "$legacy_root" <<'PY'
import csv, pathlib, sys, yaml
csv_path, models_root, legacy_root = sys.argv[1:4]
projects = { (r.get("project") or "").strip()
             for r in csv.DictReader(open(csv_path), delimiter=";")
             if (r.get("project") or "").strip() }
kept = []
for cfg in pathlib.Path(models_root).rglob("config"):
    if legacy_root in cfg.as_posix():
        continue
    try:
        proj = (yaml.safe_load(cfg.read_text()) or {}).get("project")
    except Exception:
        proj = ""
    proj = (proj or "").strip()
    if not proj or proj in projects:
        kept.append((cfg.parent.as_posix(), proj))
for path, proj in kept:
    print(f"keep {path} (project={proj})")
print(f"kept {len(kept)} run(s)")
PY
