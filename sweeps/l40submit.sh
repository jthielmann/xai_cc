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

# Derive a safer, more descriptive job name from the config
name="$(python "${REPO_ROOT}/script/cli/print_job_name.py" "$cfg" 2>/dev/null || basename "$cfg")"
name="${name%.*}"

# Get absolute path to cfg
cfg_abs="$(readlink -f "$cfg" 2>/dev/null || python -c 'import os,sys;print(os.path.abspath(sys.argv[1]))' "$cfg")"
mkdir -p logs

tmp="$(mktemp -t "${name}_sbatch_XXXXXX.sh")"
cat > "$tmp" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=96:00:00

echo "[\$(date)] starting run on \$(hostname) — job: \${SLURM_JOB_NAME} (\${SLURM_JOB_ID})"
mkdir -p logs
shopt -s expand_aliases
source ~/.bashrc
lit
script

srun --ntasks=1 --gpus=1 --cpus-per-gpu=4 \\
     python main.py --config "${cfg_abs}"
EOF

chmod +x "$tmp"
echo "Submitting job '${name}' with config: ${cfg_abs}"
sbatch "$tmp"
rm -f "$tmp"
