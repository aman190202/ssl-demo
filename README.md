
## SLURM Setup (optional)
```bash
interact -q gpu -f quadrortx -t 1:00:00 -m 16g
module load cuda/12.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load miniconda3/23.11.0s

source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

conda create -n ssl python=3.11
conda activate ssl
```
## Stable Pretraining Setup
```bash
git clone https://github.com/rbalestr-lab/stable-pretraining.git
```
Follow instructions at [https://github.com/rbalestr-lab/stable-pretraining](https://github.com/rbalestr-lab/stable-pretraining)

## PyTorch runs

```bash
# fresh
python torch_pretrain.py --run_dir runs/mae_galaxy10 --epochs 100 --save_every 5

# resume
python torch_pretrain.py --run_dir runs/mae_galaxy10 --resume runs/mae_galaxy10/last.ckpt
```