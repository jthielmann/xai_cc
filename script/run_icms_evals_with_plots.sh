#!/usr/bin/env bash

set -u

if [[ ! -f "eval_main.py" ]]; then
  echo "error: run from script/ (missing eval_main.py)" >&2
  exit 1
fi

CONFIG_DIR="../sweeps/configs"
if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "error: configs dir not found: $CONFIG_DIR" >&2
  exit 1
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/eval_icms*)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "error: no icms configs matched in $CONFIG_DIR" >&2
  exit 1
fi

failures=()
for cfg in "${configs[@]}"; do
  if [[ "$cfg" == *_debug ]]; then
    echo "skip debug: $cfg"
    continue
  fi
  if grep -qiE '^[[:space:]]*debug:[[:space:]]*true[[:space:]]*($|#)' "$cfg"; then
    echo "skip debug: $cfg"
    continue
  fi
  echo "==> running: $cfg"
  tmp_cfg=$(mktemp)
  printf "%s\n%s\n%s\n%s\n" "$(cat "$cfg")" "boxplots: true" "plot_violin: true" "log_to_wandb: true" >"$tmp_cfg"
  if python eval_main.py --config "$tmp_cfg"; then
    echo "ok: $cfg"
  else
    code=$?
    echo "fail($code): $cfg" >&2
    failures+=("$cfg:$code")
  fi
  rm -f "$tmp_cfg"
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "" >&2
  echo "failed configs:" >&2
  for f in "${failures[@]}"; do echo "  $f" >&2; done
  exit 1
fi

echo "all icms configs succeeded"
exit 0
