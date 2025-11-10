#!/usr/bin/env bash
set -euo pipefail

# Base directory containing trained model run folders (each with 'config' and 'best_model.pth').
MODELS_BASE="${MODELS_BASE:-../models}"
# Python executable to use.
PYTHON_BIN="${PYTHON_BIN:-python}"
# Directory for temporary generated configs.
CONFIG_TMP_DIR="${CONFIG_TMP_DIR:-/tmp}"

# Patterns to match run directories (substring match). Override by passing args.
if [ "$#" -gt 0 ]; then
  PATTERNS=("$@")
else
  PATTERNS=(icms2up icms2down icms3up icms3down)
fi

if [ ! -d "$MODELS_BASE" ]; then
  echo "models base missing: $MODELS_BASE" >&2
  exit 1
fi
if [ ! -f "script/crp_main.py" ]; then
  echo "missing entrypoint: script/crp_main.py" >&2
  exit 1
fi

found=0
base_dir="${MODELS_BASE%/}"

while IFS= read -r -d '' cfg_path; do
  run_dir="$(dirname "$cfg_path")"
  [ -f "$run_dir/best_model.pth" ] || continue

  matched=0
  for pat in "${PATTERNS[@]}"; do
    case "$run_dir" in
      *"$pat"*) matched=1; break ;;
    esac
  done
  [ "$matched" -eq 1 ] || continue

  found=1
  rel="$run_dir"
  if [[ "$rel" == "$base_dir"* ]]; then
    rel="${rel#${base_dir}/}"
  fi

  tmp_cfg="$(mktemp "${CONFIG_TMP_DIR%/}/crp_XXXXXX.yaml")"
  cat >"$tmp_cfg" <<EOF
project: xai
name: crp
xai_pipeline: manual
run_name: crp_$(basename "$rel")
group: crp_icms
job_type: xai
tags: [crp, icms]
log_to_wandb: true
debug: false
dataset: coad
model_state_path: ${rel}
target_layer: encoder
crp: true
pcx: false
EOF

  "$PYTHON_BIN" script/crp_main.py --config "$tmp_cfg"
  rm -f "$tmp_cfg"
done < <(find "$MODELS_BASE" -type f -name config -print0)

if [ "$found" -eq 0 ]; then
  echo "no matching model runs found under $MODELS_BASE" >&2
  exit 1
fi

