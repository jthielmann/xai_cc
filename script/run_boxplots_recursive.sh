#!/usr/bin/env bash

set -u

if [[ ! -f "boxplot_main.py" ]]; then
  echo "error: run from script/ (missing boxplot_main.py)" >&2
  exit 1
fi

if [[ $# -ne 0 ]]; then
  echo "usage: bash run_boxplots_recursive.sh" >&2
  exit 1
fi

SCAN_ROOT="../evaluation"
if [[ ! -d "$SCAN_ROOT" ]]; then
  echo "error: scan_root not found: $SCAN_ROOT" >&2
  exit 1
fi

abs_scan=$(python - "$SCAN_ROOT" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)

tmp_yaml=$(mktemp)
python - "$abs_scan" "$tmp_yaml" <<'PY'
import os, sys, yaml
scan_root, out_path = sys.argv[1:3]
cfg = {
    'debug': False,
    'eval_label': 'results',
    'scan_root': str(scan_root),
    'gene_sets': 'all',
    'plot_box': True,
    'plot_violin': True,
    'log_to_wandb': False,
}
with open(out_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
PY

if python boxplot_main.py --config "$tmp_yaml"; then
  echo "ok: results"
  rm -f "$tmp_yaml"
  echo "all boxplots succeeded"
  exit 0
else
  code=$?
  echo "fail($code): results" >&2
  rm -f "$tmp_yaml"
  exit 1
fi
