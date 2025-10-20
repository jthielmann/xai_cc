#!/bin/bash
# submit_eval.sh — submit an evaluation/XAI job (e.g., LXT) using script/eval_main.py
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./submit_eval.sh <config_path>" >&2
  exit 1
fi

cfg="$1"

# Resolve repo root so we can call helper regardless of CWD
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Derive job name from config (uses top-level 'project' key)
name_result=$(python "${REPO_ROOT}/script/cli/print_job_name.py" "$cfg" || true)
if [[ -z "${name_result:-}" ]]; then
  echo "Warning: failed to derive job name from config; defaulting to 'eval'" >&2
  name_result="eval"
fi
name="${name_result%.*}"

# Absolute config path
cfg_abs="$(readlink -f "$cfg" 2>/dev/null || python -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$cfg")"
mkdir -p logs

tmp="$(mktemp -t "${name}_eval_sbatch_XXXXXX.sh")"
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

echo "[$(date)] starting eval job on $(hostname) — job: ${SLURM_JOB_NAME:-eval} (${SLURM_JOB_ID:-local})"
mkdir -p logs
shopt -s expand_aliases
source ~/.bashrc

# Activate your environment if needed, e.g.:
# conda activate xai

srun --ntasks=1 --gpus=1 --cpus-per-gpu=4 \
     python "${REPO_ROOT}/script/eval_main.py" --config "${cfg_abs}"
EOF

chmod +x "$tmp"
echo "Submitting eval job '${name}' with config: ${cfg_abs}"
sbatch "$tmp"
rm -f "$tmp"
