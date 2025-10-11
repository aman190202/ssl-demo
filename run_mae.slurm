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


# add before python in run_mae.slurm
SCRATCH=/oscar/scratch/aagar133/ssl
mkdir -p $SCRATCH/wandb $SCRATCH/.cache/wandb $SCRATCH/.config/wandb
export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb

# Mitigate CUDA memory fragmentation per PyTorch guidance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# -----------------------------
# Run MAE pretraining
# -----------------------------


python torch_pretrain.py \
  --dataset matthieulel/galaxy10_decals \
  --run_dir runs/mae_vits16_384d12_m0p70_e200 \
  --img_size 256 --patch_size 16 \
  --num_workers 4 \
  --batch_size 64 --accum_steps 1 \
  --epochs 200 --warmup_epochs 10 \
  --lr_schedule cosine --lr 4e-4 --weight_decay 0.05 \
  --mask_ratio 0.70 \
  --emb_dim 384 --enc_depth 12 --enc_heads 6 \
  --dec_dim 384 --dec_depth 4 \
  --knn_k 20 --knn_t 0.07 --eval_every 1 \
  --save_every 10 --vis_every 10 \
  --use_wandb --wandb_project mae_galaxy10 \
  --wandb_run_name mae_vits16_384d12_m0p70_e200

python linear_probe.py \
  --encoder_path runs/mae_fast_reliable_192d8_random_masking/encoder_epoch_004_best.pth \
  --save_dir runs/linear_probe_fix \
  --dataset matthieulel/galaxy10_decals \
  --img_size 256 --patch_size 16 \
  --batch_size 256 --num_workers 2 \
  --probe_mode lbfgs --lbfgs_max_iter 120 \
  --l2norm --epochs 500

pip install matplotlib

python collect_metrics.py --roots runs --overlay-val


