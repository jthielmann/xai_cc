#!/usr/bin/env bash
set -u

if [[ ! -f "crp_main.py" ]]; then
  echo "error: run from script/ (missing crp_main.py)" >&2
  exit 1
fi

CONFIG_DIR="../sweeps/configs"
if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "error: configs dir not found: $CONFIG_DIR" >&2
  exit 1
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/crp_icms*)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "error: no crp_icms* configs matched in $CONFIG_DIR" >&2
  exit 1
fi

ROOT="$(cd .. && pwd)"
MODELS_DIR="$ROOT/models"
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

  base_name=$(grep -E '^[[:space:]]*model_state_path:' "$cfg" | sed -E 's/.*:[[:space:]]*"?([^"[:space:]]+)"?.*/\1/' || true)
  if [[ -z "$base_name" ]]; then
    echo "skip (no model_state_path): $cfg"
    continue
  fi
  base_path="$MODELS_DIR/$base_name"
  if [[ ! -d "$base_path" ]]; then
    echo "skip (missing base path): $base_path"
    continue
  fi

  mapfile -t runs < <(find "$base_path" -type f -name config -printf '%h\n' 2>/dev/null | sort -u)
  if [[ ${#runs[@]} -eq 0 ]]; then
    mapfile -t runs < <(find "$base_path" -type f -name config -exec dirname {} \; 2>/dev/null | sort -u)
  fi
  if [[ ${#runs[@]} -eq 0 ]]; then
    echo "skip (no runs under): $base_path"
    continue
  fi

  for run_dir in "${runs[@]}"; do
    if [[ ! -f "$run_dir/best_model.pth" ]]; then
      continue
    fi
    rel="${run_dir#${MODELS_DIR}/}"
    echo "==> running: $cfg :: $rel"
    tmp_cfg=$(mktemp)
    printf "%s\n" "$(cat "$cfg")" \
      "run_name: crp_$(basename "$rel")" \
      "group: crp_icms" \
      "job_type: xai" \
      "tags: [crp, icms]" \
      "log_to_wandb: ${CRP_WANDB:-true}" \
      "model_state_path: ${rel}" >"$tmp_cfg"
    if [[ -n "${CRP_MAX_ITEMS:-}" ]]; then
      echo "crp_max_items: ${CRP_MAX_ITEMS}" >>"$tmp_cfg"
    fi
    if python crp_main.py --config "$tmp_cfg"; then
      echo "ok: $cfg :: $rel"
    else
      code=$?
      echo "fail($code): $cfg :: $rel" >&2
      failures+=("$cfg::$rel:$code")
    fi
    rm -f "$tmp_cfg"
  done
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "" >&2
  echo "failed configs/runs:" >&2
  for f in "${failures[@]}"; do echo "  $f" >&2; done
  exit 1
fi

echo "all crp icms configs succeeded"
exit 0
