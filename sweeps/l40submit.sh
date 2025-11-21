#!/bin/bash
# submit.sh — generate a tailored SLURM script and submit it
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./submit.sh <config_path>"
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

# Derive a safer, more descriptive job name from the config; fail if it cannot be derived
name=""
name_result=$($PY_BIN "${REPO_ROOT}/script/cli/print_job_name.py" "$cfg")
rc=$?
if [[ $rc -ne 0 || -z "$name_result" ]]; then
  echo "Error: failed to derive job name from config '$cfg' (exit $rc)." >&2
  exit 1
fi
name="${name_result%.*}"

# Get absolute path to cfg
cfg_abs="$(readlink -f "$cfg" 2>/dev/null || $PY_BIN -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$cfg")"
mkdir -p logs

tmp="$(mktemp -t "${name}_sbatch_XXXXXX.sh")"
cat > "$tmp" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00

echo "[\$(date)] starting run on \$(hostname) — job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
mkdir -p logs
shopt -s expand_aliases
source ~/.bashrc
lit
script

# Resolve Python on the compute node
PY_BIN="\$(command -v python3 || command -v python)"
if [[ -z "\$PY_BIN" ]]; then
  echo "Error: Python interpreter not found on compute node PATH." >&2
  exit 127
fi

srun --ntasks=1 --gpus=1 --cpus-per-gpu=4 \\
     "\$PY_BIN" "${REPO_ROOT}/script/main.py" --config "${cfg_abs}"
EOF

chmod +x "$tmp"
echo "Submitting job '${name}' with config: ${cfg_abs}"
sbatch "$tmp"
rm -f "$tmp"
