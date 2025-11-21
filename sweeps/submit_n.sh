#!/usr/bin/env bash
set -euo pipefail

n="$1"
config="$2"

./submit.sh --config "$config"
sleep 120
for ((i = 2; i <= n; i++)); do

  ./submit.sh --config "$config"
done
