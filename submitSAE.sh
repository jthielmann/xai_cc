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

# Single call: the Python script iterates all encoders from the config
python -m script.main -c "$CONFIG"

echo "\nDone. SAE outputs are under ./sae/<encoder>/"
