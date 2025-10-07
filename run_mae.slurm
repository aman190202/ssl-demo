#!/bin/bash
#SBATCH --job-name=mae_galaxy10
#SBATCH --output=logs/mae_galaxy10_%j.out
#SBATCH --error=logs/mae_galaxy10_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

# -----------------------------
# Environment setup
# -----------------------------
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate /oscar/scratch/aagar133/ssl/env
# -----------------------------
# Run MAE pretraining
# -----------------------------

python torch_pretrain.py \
  --run_dir runs/mae_galaxy10 \
  --epochs 1600 \
  --save_every 5 \
  --vis_every 10 \
  --use_wandb \
  --wandb_project mae_galaxy10 \
  --wandb_run_name mae_galaxy10_run2 \
  --wandb_mode online