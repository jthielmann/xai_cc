#!/usr/bin/env bash
set -euo pipefail

csv="models_overview_full.csv"
models_root="models"
legacy_root="${models_root}/test_runs"

[[ -f "$csv" ]]

python3 - "$csv" "$models_root" "$legacy_root" <<'PY'
import csv, pathlib, sys, yaml
from datetime import datetime, timezone, timedelta
csv_path, models_root, legacy_root = sys.argv[1:4]
projects = { (r.get("project") or "").strip()
             for r in csv.DictReader(open(csv_path), delimiter=";")
             if (r.get("project") or "").strip() }
tokens = {p.lower() for p in projects}
recent = []
cutoff = datetime.now(timezone.utc) - timedelta(days=3)
for cfg in pathlib.Path(models_root).rglob("config"):
    if legacy_root in cfg.as_posix():
        continue
    try:
        proj = (yaml.safe_load(cfg.read_text()) or {}).get("project")
    except Exception:
        continue
    proj = (proj or "").strip()
    path_hit = any(tok and tok in cfg.parent.as_posix().lower() for tok in tokens)
    if not proj or proj in projects or path_hit:
        continue
    run_dir = cfg.parent
    mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
    if mtime >= cutoff:
        recent.append(run_dir.as_posix())
    dest = pathlib.Path(legacy_root) / run_dir.name
    i = 1
    while dest.exists():
        dest = pathlib.Path(legacy_root) / f"{run_dir.name}_{i}"
        i += 1
    print(f"would mv {run_dir} -> {dest}")
if recent:
    print("\nRecent (<=3 days) candidates:")
    for r in sorted(recent):
        print(r)
PY
