#!/bin/bash
# submit.sh — generate a tailored SLURM script and submit it
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./submit.sh <config_path>"
  exit 1
fi

cfg="$1"
name="$(basename "$cfg")"
name="${name%.*}"

# Get absolute path to cfg (fallback if readlink -f is unavailable(on m)
cfg_abs="$(readlink -f "$cfg")"
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
echo "Submitting job '${name}'"
sbatch "$tmp"
rm -f "$tmp"