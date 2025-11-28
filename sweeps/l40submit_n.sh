#!/usr/bin/env bash
set -euo pipefail

n="$1"
config="$2"

./l40submit.sh "$config"
sleep 120
for ((i = 2; i <= n; i++)); do

  ./l40submit.sh "$config"
done
