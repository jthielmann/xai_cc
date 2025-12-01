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
tokens = {p.lower() for p in projects}
kept = []
for cfg in pathlib.Path(models_root).rglob("config"):
    if legacy_root in cfg.as_posix():
        continue
    try:
        proj = (yaml.safe_load(cfg.read_text()) or {}).get("project")
    except Exception:
        proj = ""
    proj = (proj or "").strip()
    path_hit = any(tok and tok in cfg.parent.as_posix().lower() for tok in tokens)
    if not proj or proj in projects or path_hit:
        kept.append((cfg.parent.as_posix(), proj))
for path, proj in kept:
    print(f"keep {path} (project={proj})")
print(f"kept {len(kept)} run(s)")
PY
