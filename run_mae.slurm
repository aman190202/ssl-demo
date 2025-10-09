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

echo "Latest_with_probe_inserted"
python torch_pretrain.py \
  --dataset matthieulel/galaxy10_decals \
  --run_dir runs/mae_mask60 \
  --img_size 256 --patch_size 16 \
  --batch_size 96 \
  --epochs 400 --warmup_epochs 40 \
  --lr 1.5e-4 --weight_decay 0.05 --lr_schedule cosine \
  --mask_ratio 0.60 \
  --emb_dim 512 --enc_depth 16 --enc_heads 8 \
  --dec_dim 512 --dec_depth 8 \
  --knn_k 20 --knn_t 0.07 \
  --save_every 10 --use_wandb --wandb_project mae_galaxy10 \
  --wandb_run_name mae_mask60_512_16_512_16



# python probe_galaxy.py \
#   --ckpt_path runs/mae_galaxy10_fix/encoder_epoch_015.pth \
#   --respect_ckpt_meta \
#   --schedule onecycle --probe_batch_size 96 --max_lr 0.05 --nesterov \
#   --probe_epochs 1000 --num_workers 1 --emb_dim 384 --enc_depth 12 --enc_heads 6
