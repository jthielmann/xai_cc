#!/usr/bin/env bash

set -u

if [[ ! -f "boxplot_main.py" ]]; then
  echo "error: run from script/ (missing boxplot_main.py)" >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "usage: bash run_boxplots_recursive.sh <boxplot_cfg_yaml>" >&2
  exit 1
fi

CFG="$1"
if [[ ! -f "$CFG" ]]; then
  echo "error: config not found: $CFG" >&2
  exit 1
fi

BASE_ROOT="../evaluation"
if [[ ! -d "$BASE_ROOT" ]]; then
  echo "error: base_root not found: $BASE_ROOT" >&2
  exit 1
fi

dirs_list=$(find "$BASE_ROOT" -type f -name forward_metrics.csv -print0 | xargs -0 -n1 dirname | sort -u)
if [[ -z "$dirs_list" ]]; then
  echo "error: no forward_metrics.csv under $BASE_ROOT" >&2
  exit 1
fi

failures=()
while IFS= read -r d; do
  [[ -z "$d" ]] && continue
  name=$(python - "$BASE_ROOT" "$d" <<'PY'
import os, sys
root=os.path.abspath(sys.argv[1]); d=os.path.abspath(sys.argv[2])
print(os.path.relpath(d, root))
PY
)
  echo "==> boxplots for: $name"

  tmp_yaml=$(mktemp)
  out_src_root=$(python - "$BASE_ROOT" "$name" <<'PY'
import os, sys
base, rel = sys.argv[1:3]
label = rel.replace("\\", "/").rstrip("/").replace("/", "__").replace(" ", "_")[:128]
print(os.path.abspath(os.path.join(base, label, "boxplots")))
PY
)
  absd=$(python - "$d" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)

  python - "$CFG" "$name" "$absd" "$tmp_yaml" <<'PY'
import sys, yaml, os
cfg_path, label, scan_root, out_path = sys.argv[1:5]
with open(cfg_path,'r') as f: base = yaml.safe_load(f) or {}
base['debug'] = False
base['eval_label'] = str(label)
base['scan_root'] = str(scan_root)
with open(out_path,'w') as f:
    yaml.safe_dump(base, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
PY

  if python boxplot_main.py --config "$tmp_yaml"; then
    echo "ok: $name"
  else
    code=$?
    echo "fail($code): $name" >&2
    failures+=("$name:$code")
  fi

  if [[ -d "$out_src_root" ]]; then
    mkdir -p "$d/boxplots"
    shopt -s nullglob
    for f in "$out_src_root"/*.png; do
      mv -f "$f" "$d/boxplots/"
    done
    shopt -u nullglob
    rmdir "$out_src_root" 2>/dev/null || true
  fi

  rm -f "$tmp_yaml"
done <<< "$dirs_list"

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "" >&2
  echo "failed dirs:" >&2
  for f in "${failures[@]}"; do echo "  $f" >&2; done
  exit 1
fi

echo "all boxplots succeeded"
exit 0

