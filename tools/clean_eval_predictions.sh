#!/usr/bin/env bash
set -euo pipefail

# hard-code evaluation dir relative to repo root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="${SCRIPT_DIR}/../evaluation"

if [ ! -d "$ROOT" ]; then
  echo "error: evaluation dir not found: $ROOT" 1>&2
  exit 3
fi

mapfile -t PDIRS < <(find "$ROOT" -type d -name predictions -print)
for d in "${PDIRS[@]:-}"; do
  [ -z "$d" ] && continue
  echo "removing dir: $d"
  rm -rf -- "$d"
done

mapfile -t FMCSV < <(find "$ROOT" -type f -name forward_metrics.csv -print)
for f in "${FMCSV[@]:-}"; do
  [ -z "$f" ] && continue
  echo "removing file: $f"
  rm -f -- "$f"
done

echo "done: $ROOT"
