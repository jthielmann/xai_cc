#!/bin/bash
#SBATCH --job-name=crc_norm
#SBATCH --output=logs/crc_norm_%j_%t.out
#SBATCH --error=logs/crc_norm_%j_%t.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3           # 3 GPUs on the node
#SBATCH --cpus-per-gpu=4       # 4 CPUs for each GPU  → 12 CPUs total
#SBATCH --mem=128G
#SBATCH --time=48:00:00


echo "[$(date)] starting run on $(hostname)"
shopt -s expand_aliases
source ~/.bashrc
lit
script

for gpu in 0 1 2; do
  echo "[$(date)] launching GPU $gpu"
  srun --exclusive \
       --ntasks=1 \
       --gpus=1 \
       --cpus-per-gpu=4 \
       --gpu-bind=single:mask_gpu:$gpu \
       python main.py --config "$SLURM_SUBMIT_DIR/config" &
  sleep 30
done

wait
echo "[$(date)] all runs completed"