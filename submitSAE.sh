#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submitSAE.sh path/to/config.yaml
# If no config is provided, uses the example config.

CONFIG="${1:-sweeps/configs/sae_example.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# Pick a Python interpreter (prefer python3)
if command -v python3 >/dev/null 2>&1; then
  PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
else
  echo "Error: Python interpreter not found on PATH. Install Python 3 or add it to PATH." >&2
  exit 127
fi

# Single call: the Python script iterates all encoders from the config
"$PY_BIN" -m script.main -c "$CONFIG"

echo "\nDone. SAE outputs are under ./sae/<encoder>/"
