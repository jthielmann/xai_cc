#!/bin/bash
# submit_xai.sh — submit an XAI (LRP/CRP/PCX) job using script/eval_main.py
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./submit_xai.sh <config_path>" >&2
  exit 1
fi

cfg="$1"

# Resolve repo root so we can call helper regardless of CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pick a Python interpreter (prefer python3)
if command -v python3 >/dev/null 2>&1; then
  PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
else
  echo "Error: Python interpreter not found on PATH. Install Python 3 or add it to PATH." >&2
  exit 127
fi

# Derive job name from config
name_result=$($PY_BIN "${REPO_ROOT}/script/cli/print_job_name.py" "$cfg" || true)
if [[ -z "${name_result:-}" ]]; then
  name_result="xai"
fi
name="${name_result%.*}"

# Absolute config path
cfg_abs="$(readlink -f "$cfg" 2>/dev/null || $PY_BIN -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$cfg")"
mkdir -p logs

tmp="$(mktemp -t "${name}_xai_sbatch_XXXXXX.sh")"
cat > "$tmp" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00

echo "[$(date)] starting XAI job on $(hostname) — job: \
${SLURM_JOB_NAME:-xai} (${SLURM_JOB_ID:-local})"
mkdir -p logs
shopt -s expand_aliases
source ~/.bashrc

# Activate your XAI env here if you use one, e.g.:
# conda activate xai-crp

# Resolve Python on the compute node
PY_BIN_NODE="$(command -v python3 || command -v python)"
if [[ -z "$PY_BIN_NODE" ]]; then
  echo "Error: Python interpreter not found on compute node PATH." >&2
  exit 127
fi

srun --ntasks=1 --gpus=1 --cpus-per-gpu=4 \
     "$PY_BIN_NODE" "${REPO_ROOT}/script/xai_main.py" --config "${cfg_abs}"
EOF

chmod +x "$tmp"
echo "Submitting XAI job '${name}' with config: ${cfg_abs}"
sbatch "$tmp"
rm -f "$tmp"

