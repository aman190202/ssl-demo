# Confidence: 95%
# Rationale: Code paths exercised in similar MAE projects; changes are localized to checkpoint I/O and are standard PyTorch patterns.

import argparse
import csv
import json
import os
import math
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils as vutils
from datasets import load_dataset
from timm.models.vision_transformer import Block

try:
    import wandb
except Exception:
    wandb = None

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


def parse_args():
    p = argparse.ArgumentParser("MAE pretraining on Galaxy10 with kNN eval and checkpoints")

    # data
    p.add_argument("--dataset", type=str, default="matthieulel/galaxy10_decals")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.1, help="val split fraction when dataset has no validation")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--emb_dim", type=int, default=384)
    p.add_argument("--dec_dim", type=int, default=512)
    p.add_argument("--enc_depth", type=int, default=12)
    p.add_argument("--dec_depth", type=int, default=4)
    p.add_argument("--enc_heads", type=int, default=6)
    p.add_argument("--dec_heads", type=int, default=8)
    p.add_argument("--mask_ratio", type=float, default=0.65)

    # optim/schedule
    p.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (after warmup)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--warmup_epochs", type=int, default=8)
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine"])
    p.add_argument("--global_batch_size", type=int, default=4096, help="effective global batch size via grad accumulation")
    p.add_argument("--accum_steps", type=int, default=0, help="override grad accumulation steps; 0 to auto-compute from global_batch_size")

    # eval
    p.add_argument("--knn_k", type=int, default=20)
    p.add_argument("--knn_t", type=float, default=0.07, help="temperature for kNN soft voting")
    p.add_argument("--eval_every", type=int, default=1)

    # online probe (lightweight, updated during pretraining)
    p.add_argument("--online_probe", action="store_true", help="enable online linear probe during pretraining")
    p.add_argument("--online_probe_lr", type=float, default=0.1)
    p.add_argument("--online_probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--online_probe_momentum", type=float, default=0.9)
    p.add_argument("--online_probe_update_every", type=int, default=1, help="update online probe every N optimizer steps")

    # offline probe (full training on cached features at checkpoints)
    p.add_argument("--offline_probe", action="store_true", help="run offline linear probe at intervals using cached features")
    p.add_argument("--offline_probe_every", type=int, default=10, help="run offline probe every N epochs")
    p.add_argument("--offline_probe_epochs", type=int, default=200)
    p.add_argument("--offline_probe_lr", type=float, default=1e-2)
    p.add_argument("--offline_probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--offline_probe_batch_size", type=int, default=1024)
    p.add_argument("--offline_probe_schedule", type=str, default="const", choices=["const"], help="LR schedule for offline probe")

    # io
    p.add_argument("--run_dir", type=str, default="runs/mae_galaxy10")
    p.add_argument("--save_every", type=int, default=5, help="save encoder every N epochs")
    p.add_argument("--vis_every", type=int, default=10, help="log reconstruction panels every N epochs")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume; if empty, auto-resume from run_dir/last.ckpt if present")

    # wandb
    p.add_argument("--use_wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="mae_galaxy10", help="W&B project")
    p.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team or username)")
    p.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")

    args = p.parse_args() if hasattr(__builtins__, "__IPYTHON__") is False else p.parse_args("")
    return args


# ============================================================
# 2. Dataset (Galaxy10) and transforms
# ============================================================
args = parse_args()

IMG_SIZE = args.img_size
PATCH_SIZE = args.patch_size
MASK_RATIO = args.mask_ratio

torch.manual_seed(args.seed)

# Gentler training augmentation (astronomy-friendly)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Evaluation transform (deterministic)
eval_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

ds = load_dataset(args.dataset)


class Galaxy10(torch.utils.data.Dataset):
    def __init__(self, hf_split, tf):
        self.base = hf_split
        self.tf = tf
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img = self.base[i]["image"]
        y = int(self.base[i]["label"])
        return self.tf(img), y


def safe_train_test_split(hf_ds, test_size, seed):
    try:
        return hf_ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    except Exception:
        return hf_ds.train_test_split(test_size=test_size, seed=seed)


def build_splits(dataset_dict):
    has_test = "test" in dataset_dict
    has_val = "validation" in dataset_dict

    hf_train = dataset_dict["train"]
    hf_val = dataset_dict["validation"] if has_val else None
    hf_test = dataset_dict["test"] if has_test else None

    if hf_val is None:
        split = safe_train_test_split(hf_train, test_size=args.val_split, seed=args.seed)
        hf_train, hf_val = split["train"], split["test"]
    if hf_test is None:
        split = safe_train_test_split(hf_train, test_size=args.val_split, seed=args.seed)
        hf_train, hf_test = split["train"], split["test"]
    return hf_train, hf_val, hf_test


hf_train, hf_val, hf_test = build_splits(ds)

# Train loader with augmentation
train_set = Galaxy10(hf_train, train_tf)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=(device == "cuda")
)

# Deterministic loaders for eval
val_set = Galaxy10(hf_val, eval_tf)
test_set = Galaxy10(hf_test, eval_tf)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device == "cuda")
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device == "cuda")
)

# A *non-augmented* train loader for building the kNN feature bank and visualization
train_set_eval = Galaxy10(hf_train, eval_tf)
train_loader_eval = torch.utils.data.DataLoader(
    train_set_eval, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=(device == "cuda")
)

# ============================================================
# 3. Patching utilities
# ============================================================
def patchify(imgs, p=PATCH_SIZE):
    B, C, H, W = imgs.shape
    patches = imgs.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(B, -1, C * p * p)

def unpatchify(patches, p=PATCH_SIZE):
    B, N, PP = patches.shape
    C = 3
    h = w = IMG_SIZE // p
    x = patches.view(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, h * p, w * p)

def random_masking(x, mask_ratio=MASK_RATIO):
    B, N, D = x.shape
    N_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :N_keep]
    x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=x.device)
    mask[:, :N_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_kept, mask, ids_restore, ids_keep


# import math

# def random_masking(
#     x: torch.Tensor,
#     mask_ratio: float,
#     *,
#     scores: torch.Tensor | None = None,  # [B, N]
#     h: int | None = None,
#     w: int | None = None,
#     block_frac: float = 0.4,
#     bright_frac: float = 0.4,
#     ratio_jitter: float = 0.0,           # IMPORTANT: keep 0.0 to ensure fixed N_keep across batch
#     device: torch.device | None = None,
# ):
#     """
#     Hybrid masking that guarantees a fixed N_keep for the entire batch.
#     Returns: x_kept [B, N_keep, D], mask [B, N] (1=masked), ids_restore [B, N], ids_keep [B, N_keep]
#     """
#     B, N, D = x.shape
#     device = device or x.device
#     if h is None or w is None:
#         g = int(math.isqrt(N))
#         assert g * g == N, "Provide h,w for non-square layouts"
#         h = w = g

#     # ---- target counts (fixed across batch) ----
#     N_mask_target = int(round(N * mask_ratio))
#     N_mask_target = max(0, min(N, N_mask_target))
#     N_keep = N - N_mask_target

#     mask = torch.zeros(B, N, device=device, dtype=torch.bool)

#     # ---- 1) blockwise masking ----
#     if block_frac > 0 and N_mask_target > 0:
#         block_budget = int(round(N_mask_target * block_frac))
#         for b in range(B):
#             remaining = block_budget
#             attempts = 0
#             while remaining > 0 and attempts < 8:
#                 attempts += 1
#                 area = max(1, min(remaining, int(0.3 * N)))
#                 rh = max(1, int(torch.randint(1, min(h, int(math.sqrt(area))) + 1, (1,)).item()))
#                 rw = max(1, min(w, area // rh if area // rh > 0 else 1))
#                 r0 = int(torch.randint(0, max(1, h - rh + 1), (1,)).item())
#                 c0 = int(torch.randint(0, max(1, w - rw + 1), (1,)).item())
#                 # build rectangle boolean mask [N]
#                 rows = torch.arange(h, device=device)
#                 cols = torch.arange(w, device=device)
#                 rect = ((rows[:, None] >= r0) & (rows[:, None] < r0 + rh) &
#                         (cols[None, :] >= c0) & (cols[None, :] < c0 + rw)).flatten()
#                 newly = (~mask[b]) & rect
#                 add = int(newly.sum().item())
#                 if add == 0:
#                     continue
#                 mask[b, newly] = True
#                 remaining -= add

#     # ---- 2) brightness / score-driven masking ----
#     if scores is not None and bright_frac > 0 and N_mask_target > 0:
#         assert scores.shape == (B, N)
#         s = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
#         # remaining budget after blocks (per image)
#         remaining = (N_mask_target - mask.sum(dim=1)).clamp_min(0)
#         take_bright = (remaining.float() * bright_frac).round().long()
#         for b in range(B):
#             k = int(min(take_bright[b].item(), (~mask[b]).sum().item()))
#             if k <= 0:
#                 continue
#             idx_avail = torch.nonzero(~mask[b], as_tuple=False).squeeze(1)
#             scores_avail = s[b, idx_avail]
#             if idx_avail.numel() > 0:
#                 topk = torch.topk(scores_avail, k=k, largest=True, sorted=False).indices
#                 mask[b, idx_avail[topk]] = True

#     # ---- 3) random fill or trim to meet EXACT N_mask_target ----
#     for b in range(B):
#         masked_now = int(mask[b].sum().item())
#         if masked_now < N_mask_target:
#             # add random masks from remaining
#             need = N_mask_target - masked_now
#             idx_avail = torch.nonzero(~mask[b], as_tuple=False).squeeze(1)
#             choice = idx_avail[torch.randperm(idx_avail.numel(), device=device)[:need]]
#             mask[b, choice] = True
#         elif masked_now > N_mask_target:
#             # unmask some randomly to reduce to target
#             extra = masked_now - N_mask_target
#             idx_masked = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
#             choice = idx_masked[torch.randperm(idx_masked.numel(), device=device)[:extra]]
#             mask[b, choice] = False
#         # now exactly N_mask_target masked → exactly N_keep kept

#     # ---- build ids_keep / ids_restore (consistent sizes) ----
#     keep = ~mask
#     ids_keep = torch.zeros(B, N_keep, device=device, dtype=torch.long)
#     ids_mask = torch.zeros(B, N_mask_target, device=device, dtype=torch.long)
#     for b in range(B):
#         idx_keep_b = torch.nonzero(keep[b], as_tuple=False).squeeze(1)
#         # just in case: take first N_keep (order arbitrary)
#         ids_keep[b] = idx_keep_b[:N_keep]
#         idx_mask_b = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
#         ids_mask[b] = idx_mask_b[:N_mask_target]
#     # permutation per image: [keep..., mask...]
#     ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)                      # [B, N]
#     # ids_restore maps from shuffled back to original positions
#     ids_restore = torch.argsort(ids_shuffle, dim=1)

#     # gather kept tokens into a fixed [B, N_keep, D]
#     x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

#     return x_kept, mask.float(), ids_restore, ids_keep


# ============================================================
# 3.1. Fixed 2D sin-cos positional embeddings
# ============================================================
def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """Return 1D sin-cos positional embeddings for a vector of positions.

    embed_dim must be even. positions: [N]
    Output: [N, embed_dim]
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half_dim = embed_dim // 2
    # Compute frequencies
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=positions.device)
    freq_seq = 1.0 / (10000 ** (freq_seq / half_dim))  # [half_dim]
    # Outer product: [N, half_dim]
    args = positions.float().unsqueeze(1) * freq_seq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int = None) -> torch.Tensor:
    """Return 2D sin-cos positional embeddings.

    Output: [1, grid_h*grid_w, embed_dim]
    """
    if grid_w is None:
        grid_w = grid_h
    assert embed_dim % 2 == 0, "embed_dim must be even"
    # Flattened 2D grid coordinates
    yy = torch.arange(grid_h, dtype=torch.float32)
    xx = torch.arange(grid_w, dtype=torch.float32)
    pos_y = yy.repeat_interleave(grid_w)  # [N]
    pos_x = xx.repeat(grid_h)             # [N]
    # Half/half split for y/x embeddings
    dim_half = embed_dim // 2
    emb_y = _get_1d_sincos_pos_embed(dim_half, pos_y)
    emb_x = _get_1d_sincos_pos_embed(dim_half, pos_x)
    emb = torch.cat([emb_y, emb_x], dim=1)  # [N, embed_dim]
    return emb.unsqueeze(0)  # [1, N, embed_dim]

# ============================================================
# 4. MAE with ViT encoder
# ============================================================
class MAEViT(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE,
                 patch_size=PATCH_SIZE,
                 emb_dim=384,
                 dec_dim=512,
                 enc_depth=12,
                 dec_depth=4,
                 enc_heads=6,
                 dec_heads=8,
                 mask_ratio=0.65):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.in_dim = 3 * patch_size * patch_size

        # Linear Patch Embedding for patchified tokens (used in encode)
        self.patch_embed = nn.Linear(self.in_dim, emb_dim)

        # Conv-based Patch Embedding (as in MAE): kernel=stride=patch_size
        self.patch_embed_conv = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Fixed 2D sin-cos positional embeddings (encoder)
        grid_size = img_size // patch_size
        self.register_buffer(
            "pos_embed_enc",
            get_2d_sincos_pos_embed(emb_dim, grid_size, grid_size),
            persistent=False,
        )

        # ViT encoder
        self.encoder = nn.ModuleList([
            Block(dim=emb_dim, num_heads=enc_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(emb_dim)

        # Mask token + decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.proj_enc_to_dec = nn.Linear(emb_dim, dec_dim)
        # Fixed 2D sin-cos positional embeddings (decoder)
        self.register_buffer(
            "pos_embed_dec",
            get_2d_sincos_pos_embed(dec_dim, grid_size, grid_size),
            persistent=False,
        )

        self.decoder = nn.ModuleList([
            Block(dim=dec_dim, num_heads=dec_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, self.in_dim)

    def encode(self, imgs):
        # Build raw target patches for potential consumers (e.g., visualization)
        patches = patchify(imgs)
        # Conv patch embed produces [B, emb_dim, H/ps, W/ps] -> flatten to [B, N, emb_dim]
        x = self.patch_embed_conv(imgs)
        B, C, H_, W_ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H_ * W_, C)
        x_kept, mask, ids_restore, ids_keep = random_masking(x, self.mask_ratio)
        pos_enc = torch.gather(
            self.pos_embed_enc.expand(x.shape[0], -1, -1),
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        x_kept = x_kept + pos_enc
        for blk in self.encoder:
            x_kept = blk(x_kept)
        x_kept = self.enc_norm(x_kept)
        return x_kept, patches, mask, ids_restore
    # def encode(self, imgs):
    #     patches = patchify(imgs)                       # [B, N, 3*p*p]
    #     x = self.patch_embed(patches)                  # [B, N, C]
    #     # brightness (L1 over RGB patch), higher => more salient
    #     with torch.no_grad():
    #         bright = patches.abs().mean(dim=-1)        # [B, N]
    #     h = w = self.img_size // self.patch_size

    #     x_kept, mask, ids_restore, ids_keep = random_masking(
    #         x, self.mask_ratio,
    #         scores=bright, h=h, w=w,
    #         block_frac=0.4, bright_frac=0.4, ratio_jitter=0.10
    #     )
    #     # positional embed only for kept tokens
    #     pos_enc = torch.gather(
    #         self.pos_embed_enc.expand(x.shape[0], -1, -1),
    #         1,
    #         ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    #     )
    #     x_kept = x_kept + pos_enc
    #     for blk in self.encoder:
    #         x_kept = blk(x_kept)
    #     x_kept = self.enc_norm(x_kept)
    #     return x_kept, patches, mask, ids_restore

    def decode(self, enc_tokens, ids_restore):
        B = enc_tokens.size(0)
        N = self.num_patches
        x_dec_kept = self.proj_enc_to_dec(enc_tokens)
        mask_tokens = self.mask_token.expand(B, N - x_dec_kept.shape[1], -1)
        x_combined = torch.cat([x_dec_kept, mask_tokens], dim=1)
        x_dec = torch.gather(
            x_combined, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_combined.shape[-1])
        )
        x_dec = x_dec + self.pos_embed_dec
        for blk in self.decoder:
            x_dec = blk(x_dec)
        x_dec = self.dec_norm(x_dec)
        pred = self.pred(x_dec)
        return pred

    def forward(self, imgs):
        # Build target patches from input images
        patches = patchify(imgs)
        # Normalize each patch (per-sample, per-patch) before reconstruction loss
        # Shape: patches [B, N, C*p*p]
        with torch.no_grad():
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, unbiased=False, keepdim=True)
            std = (var + 1e-6).sqrt()
            target = (patches - mean) / std

        enc_tokens, _, mask, ids_restore = self.encode(imgs)
        pred = self.decode(enc_tokens, ids_restore)
        # Normalize predictions with same normalization statistics
        pred = (pred - mean) / std

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return pred, loss

    @torch.no_grad()
    def forward_with_intermediates(self, imgs):
        enc_tokens, patches, mask, ids_restore = self.encode(imgs)
        pred = self.decode(enc_tokens, ids_restore)
        return pred, patches, mask

    @torch.no_grad()
    def extract_features(self, imgs, pool: str = "mean"):
        # Use conv patch embed to build token sequence
        x = self.patch_embed_conv(imgs)
        B, C, H_, W_ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H_ * W_, C)
        x = x + self.pos_embed_enc
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        if pool == "mean":
            x = x.mean(dim=1)
        return x


# ============================================================
# 5) TensorBoard helpers (denorm + image panels)
# ============================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def denorm(x):
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0, 1)

@torch.no_grad()
def log_recon_samples(writer, model, imgs, global_step, n_show=8):
    model.eval()
    imgs = imgs[:n_show].to(device)
    with torch.random.fork_rng(devices=[torch.device(device)] if device != "cpu" else []):
        torch.manual_seed(12345)
        pred, patches, mask = model.forward_with_intermediates(imgs)

    masked_patches = patches * (1 - mask.unsqueeze(-1))
    masked_img = unpatchify(masked_patches)
    recon_full = unpatchify(pred)
    blended_patches = patches * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
    recon_blended = unpatchify(blended_patches)

    grid_input  = vutils.make_grid(denorm(imgs), nrow=min(4, n_show))
    grid_masked = vutils.make_grid(denorm(masked_img), nrow=min(4, n_show))
    grid_blend  = vutils.make_grid(denorm(recon_blended), nrow=min(4, n_show))
    grid_full   = vutils.make_grid(denorm(recon_full), nrow=min(4, n_show))

    writer.add_image("00_input", grid_input, global_step)
    writer.add_image("01_masked_input", grid_masked, global_step)
    writer.add_image("02_recon_blended(masked_filled)", grid_blend, global_step)
    writer.add_image("03_recon_full", grid_full, global_step)


@torch.no_grad()
def log_recon_samples_wandb(model, imgs, global_step, n_show=8):
    if not (args.use_wandb and (wandb is not None)):
        return
    model.eval()
    imgs = imgs[:n_show].to(device)
    with torch.random.fork_rng(devices=[torch.device(device)] if device != "cpu" else []):
        torch.manual_seed(12345)
        pred, patches, mask = model.forward_with_intermediates(imgs)

    masked_patches = patches * (1 - mask.unsqueeze(-1))
    masked_img = unpatchify(masked_patches)
    recon_full = unpatchify(pred)
    blended_patches = patches * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
    recon_blended = unpatchify(blended_patches)

    grid_input  = vutils.make_grid(denorm(imgs), nrow=min(4, n_show))
    grid_masked = vutils.make_grid(denorm(masked_img), nrow=min(4, n_show))
    grid_blend  = vutils.make_grid(denorm(recon_blended), nrow=min(4, n_show))
    grid_full   = vutils.make_grid(denorm(recon_full), nrow=min(4, n_show))

    wandb.log({
        "images/00_input": wandb.Image(grid_input.cpu()),
        "images/01_masked_input": wandb.Image(grid_masked.cpu()),
        "images/02_recon_blended(masked_filled)": wandb.Image(grid_blend.cpu()),
        "images/03_recon_full": wandb.Image(grid_full.cpu()),
    }, step=global_step)

    table = wandb.Table(columns=["original", "masked", "predicted"])
    imgs_den = denorm(imgs).cpu()
    masked_den = denorm(masked_img).cpu()
    recon_den = denorm(recon_full).cpu()
    for i in range(imgs.shape[0]):
        table.add_data(wandb.Image(imgs_den[i]), wandb.Image(masked_den[i]), wandb.Image(recon_den[i]))
    wandb.log({"reconstruction_table": table}, step=global_step)


# ============================================================
# 6) kNN evaluation utilities
# ============================================================
@torch.no_grad()
def extract_features_for_loader(model: MAEViT, loader: torch.utils.data.DataLoader):
    model.eval()
    feats = []
    labels = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        z = model.extract_features(xb)
        z = F.normalize(z, dim=1)
        feats.append(z)
        labels.append(yb.to(device))
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def infer_num_classes_from_loader(loader: torch.utils.data.DataLoader) -> int:
    max_label = 0
    for _, yb in loader:
        max_label = max(max_label, int(yb.max().item()))
    return max_label + 1


@torch.no_grad()
def evaluate_linear_probe_on_loader(probe: nn.Module,
                                    model: MAEViT,
                                    loader: torch.utils.data.DataLoader,
                                    device: str) -> float:
    probe.eval()
    feats, labels = extract_features_for_loader(model, loader)
    logits = probe(feats.to(device))
    acc = (logits.argmax(1).cpu() == labels.cpu()).float().mean().item()
    return acc


def train_offline_probe_const_lr(train_feats: torch.Tensor,
                                 train_labels: torch.Tensor,
                                 val_feats: torch.Tensor,
                                 val_labels: torch.Tensor,
                                 emb_dim: int,
                                 num_classes: int,
                                 epochs: int,
                                 lr: float,
                                 weight_decay: float,
                                 batch_size: int,
                                 device: str) -> nn.Module:
    probe = nn.Linear(emb_dim, num_classes).to(device)
    # Apply weight decay to weights only
    wd_params, nowd_params = [], []
    for n, p in probe.named_parameters():
        (wd_params if p.ndim > 1 else nowd_params).append(p)
    opt = torch.optim.SGD(
        [{"params": wd_params, "weight_decay": weight_decay},
         {"params": nowd_params, "weight_decay": 0.0}],
        lr=lr, momentum=0.9
    )
    ce = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = probe(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    # Optionally evaluate on val inside caller
    return probe

@torch.no_grad()
def knn_top1(train_feats, train_labels, query_feats, query_labels, k: int, T: float, num_classes: int):
    sims = query_feats @ train_feats.t()  # cosine similarity
    top_sim, top_idx = sims.topk(k, dim=1)
    top_labels = train_labels[top_idx]  # [Nq, k]
    weights = torch.exp(top_sim / T)
    votes = torch.zeros(query_feats.size(0), num_classes, device=query_feats.device)
    votes.scatter_add_(1, top_labels, weights)
    preds = votes.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()
    return acc


# ============================================================
# 7) Checkpoint helpers
# ============================================================
def extract_encoder_bundle_from_state(model: MAEViT) -> Dict[str, torch.Tensor]:
    """Collect all encoder-facing parameters needed for feature extraction."""
    full = model.state_dict()
    keep_keys = []
    for k in full.keys():
        if (
            k.startswith("patch_embed_conv.")
            or k == "pos_embed_enc"
            or k.startswith("encoder.")
            or k.startswith("enc_norm.")
        ):
            keep_keys.append(k)
    return {k: full[k] for k in keep_keys}

def encoder_meta(args_obj) -> Dict[str, Any]:
    return {
        "img_size": args_obj.img_size,
        "patch_size": args_obj.patch_size,
        "emb_dim": args_obj.emb_dim,
        "enc_depth": args_obj.enc_depth,
        "enc_heads": args_obj.enc_heads,
        "seed": args_obj.seed,
    }


# ============================================================
# 8) Training loop with epochs + TensorBoard + Checkpoints
# ============================================================
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def append_metrics_csv(path, row_dict, header_order):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = args.run_dir
os.makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(log_dir=run_dir)

# Optional Weights & Biases init (simple and robust)
wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None
if args.use_wandb:
    if wandb is None:
        print("Weights & Biases is not installed. Disabling --use_wandb.")
        args.use_wandb = False
    else:
        run_name = args.wandb_run_name if args.wandb_run_name else f"mae_{timestamp}"
        wandb_args = {
            "project": args.wandb_project,
            "name": run_name,
            "config": vars(args),
            "mode": args.wandb_mode,
            "dir": os.path.join(run_dir, "wandb"),
            "settings": wandb.Settings(start_method="thread"),
        }
        os.makedirs(wandb_args["dir"], exist_ok=True)
        if args.wandb_entity:
            wandb_args["entity"] = args.wandb_entity
        try:
            wandb_run = wandb.init(**wandb_args)
        except Exception as e:
            print(f"Failed to initialize W&B: {e}. Disabling --use_wandb.")
            args.use_wandb = False

model = MAEViT(
    img_size=args.img_size,
    patch_size=args.patch_size,
    emb_dim=args.emb_dim,
    dec_dim=args.dec_dim,
    enc_depth=args.enc_depth,
    dec_depth=args.dec_depth,
    enc_heads=args.enc_heads,
    dec_heads=args.dec_heads,
    mask_ratio=args.mask_ratio,
).to(device)

# Optimizer: AdamW with (0.9, 0.95)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

def build_scheduler(optimizer, num_epochs, warmup_epochs, steps_per_epoch):
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(max(0, warmup_epochs) * steps_per_epoch)

    def lr_lambda(current_step: int):
        if total_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine 1->0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

global_step = 0
start_epoch = 0
last_ckpt_path = os.path.join(run_dir, "last.ckpt")

# Save hparams
hparams_path = os.path.join(run_dir, "hparams.json")
save_json(hparams_path, vars(args))

# Optionally resume
auto_resume_path = last_ckpt_path if args.resume == "" else args.resume
if os.path.isfile(auto_resume_path):
    ckpt = torch.load(auto_resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception:
            pass
    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"Resumed from {auto_resume_path} at epoch {start_epoch}")
else:
    print("Starting fresh training run.")

metrics_csv = os.path.join(run_dir, "metrics.csv")
metrics_header = [
    "epoch",
    "train_loss",
    "val_knn_top1",
    "test_knn_top1",
    "val_online_top1",
    "val_offline_top1",
    "test_offline_top1",
]

def save_checkpoint(epoch, is_best=False):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(state, last_ckpt_path)
    if is_best:
        torch.save(state, os.path.join(run_dir, "best.ckpt"))

def save_encoder_bundle(epoch: int, tag: str):
    bundle = {
        "state_dict": extract_encoder_bundle_from_state(model),
        "meta": encoder_meta(args),
        "epoch": epoch,
    }
    path = os.path.join(run_dir, f"encoder_{tag}.pth")
    torch.save(bundle, path)
    # Also update rolling latest
    torch.save(bundle, os.path.join(run_dir, "encoder_latest.pth"))

best_val = -1.0
try:
    # Accurate grad accumulation
    if args.accum_steps and args.accum_steps > 0:
        accum_steps = args.accum_steps
    else:
        accum_steps = max(1, math.ceil(args.global_batch_size / args.batch_size))
    eff_batch = args.batch_size * accum_steps
    steps_per_epoch_nominal = len(train_loader)
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch_nominal / accum_steps))
    print(f"[Info] grad_accum={accum_steps} | effective_batch={eff_batch} | "
          f"optimizer_steps/epoch={optimizer_steps_per_epoch}")

    scheduler = build_scheduler(
        opt,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        steps_per_epoch=optimizer_steps_per_epoch
    )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for step, (xb, _) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            _, loss = model(xb)

            (loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            # Also log reconstruction MSE explicitly (identical to loss here)
            writer.add_scalar("train/mse_step", loss.item(), global_step)
            if args.use_wandb and (wandb is not None):
                try:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/mse_step": loss.item(),
                        "global_step": global_step
                    }, step=global_step)
                except Exception:
                    pass
            global_step += 1

        epoch_loss /= len(train_loader)
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
        writer.add_scalar("train/mse_epoch", epoch_loss, epoch)
        if args.use_wandb and (wandb is not None):
            try:
                wandb.log({
                    "train/loss_epoch": epoch_loss,
                    "train/mse_epoch": epoch_loss,
                    "epoch": epoch
                }, step=global_step)
            except Exception:
                pass

        # Visualization panels (use eval loader for stability)
        if epoch % args.vis_every == 0:
            xb_vis, _ = next(iter(train_loader_eval))
            log_recon_samples(writer, model, xb_vis, global_step, n_show=8)
            if args.use_wandb and (wandb is not None):
                try:
                    log_recon_samples_wandb(model, xb_vis, global_step, n_show=8)
                except Exception:
                    pass

        # Eval kNN (use eval transforms for a *stable* train feature bank)
        val_acc = float("nan")
        test_acc = float("nan")
        val_online_acc = float("nan")
        val_offline_acc = float("nan")
        test_offline_acc = float("nan")
        if epoch % args.eval_every == 0:
            train_feats, train_labels = extract_features_for_loader(model, train_loader_eval)
            num_classes = int(train_labels.max().item()) + 1
            val_feats, val_labels = extract_features_for_loader(model, val_loader)
            test_feats, test_labels = extract_features_for_loader(model, test_loader)

            val_acc = knn_top1(train_feats, train_labels, val_feats, val_labels,
                               k=args.knn_k, T=args.knn_t, num_classes=num_classes)
            test_acc = knn_top1(train_feats, train_labels, test_feats, test_labels,
                                k=args.knn_k, T=args.knn_t, num_classes=num_classes)

            writer.add_scalar("val/knn_top1", val_acc, epoch)
            writer.add_scalar("test/knn_top1", test_acc, epoch)
            if args.use_wandb and (wandb is not None):
                try:
                    wandb.log({"val/knn_top1": val_acc, "test/knn_top1": test_acc, "epoch": epoch}, step=global_step)
                except Exception:
                    pass

            if val_acc > best_val:
                best_val = val_acc
                save_checkpoint(epoch, is_best=True)
                save_encoder_bundle(epoch, tag=f"epoch_{epoch:03d}_best")

        # Offline probe (train on cached features) at specified cadence
        if args.offline_probe and (epoch % args.offline_probe_every == 0):
            # Ensure we have features; recompute if not computed this epoch
            tr_feats, tr_labels = extract_features_for_loader(model, train_loader_eval)
            va_feats, va_labels = extract_features_for_loader(model, val_loader)
            te_feats, te_labels = extract_features_for_loader(model, test_loader)
            emb_dim = tr_feats.shape[1]
            num_classes_off = int(max(tr_labels.max().item(), va_labels.max().item(), te_labels.max().item())) + 1

            probe = train_offline_probe_const_lr(
                train_feats=tr_feats.cpu(),
                train_labels=tr_labels.cpu(),
                val_feats=va_feats.cpu(),
                val_labels=va_labels.cpu(),
                emb_dim=emb_dim,
                num_classes=num_classes_off,
                epochs=args.offline_probe_epochs,
                lr=args.offline_probe_lr,
                weight_decay=args.offline_probe_weight_decay,
                batch_size=args.offline_probe_batch_size,
                device=device,
            )

            with torch.no_grad():
                probe.eval()
                v_logits = probe(va_feats.to(device))
                t_logits = probe(te_feats.to(device))
                val_offline_acc = (v_logits.argmax(1).cpu() == va_labels.cpu()).float().mean().item()
                test_offline_acc = (t_logits.argmax(1).cpu() == te_labels.cpu()).float().mean().item()

            writer.add_scalar("val/offline_top1", val_offline_acc, epoch)
            writer.add_scalar("test/offline_top1", test_offline_acc, epoch)
            if args.use_wandb and (wandb is not None):
                try:
                    wandb.log({
                        "val/offline_top1": val_offline_acc,
                        "test/offline_top1": test_offline_acc,
                        "epoch": epoch
                    }, step=global_step)
                except Exception:
                    pass

        append_metrics_csv(metrics_csv, {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_knn_top1": val_acc,
            "test_knn_top1": test_acc,
            "val_online_top1": val_online_acc,
            "val_offline_top1": val_offline_acc,
            "test_offline_top1": test_offline_acc,
        }, metrics_header)

        # Save checkpoints
        save_checkpoint(epoch, is_best=False)
        if epoch % args.save_every == 0:
            save_encoder_bundle(epoch, tag=f"epoch_{epoch:03d}")

        print(f"[Epoch {epoch:03d}] mse={epoch_loss:.4f} val@1={val_acc:.4f} test@1={test_acc:.4f}")

except KeyboardInterrupt:
    print("Training interrupted by user. Saving interrupt checkpoint...")
    torch.save({
        "epoch": epoch if 'epoch' in locals() else 0,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
    }, os.path.join(run_dir, "interrupt.ckpt"))
    # Rolling latest encoder bundle
    try:
        save_encoder_bundle(epoch if 'epoch' in locals() else 0, tag="interrupt")
    except Exception:
        pass
    if args.use_wandb and (wandb is not None):
        try:
            wandb.finish()
        except Exception:
            pass
    raise

# Final saves
encoder_bundle_path = os.path.join(run_dir, "mae_encoder_bundle.pth")
full_model_path = os.path.join(run_dir, "mae_full.pth")
torch.save({"state_dict": extract_encoder_bundle_from_state(model),
            "meta": encoder_meta(args),
            "epoch": epoch if 'epoch' in locals() else args.epochs}, encoder_bundle_path)
torch.save(model.state_dict(), full_model_path)
print(f"✅ Encoder bundle saved to {encoder_bundle_path}")
print(f"✅ Full MAE model saved to {full_model_path}")

# TB hparams at end with final metrics
final_metrics = {}
try:
    final_metrics = {"hparam/val_knn_top1": val_acc, "hparam/test_knn_top1": test_acc, "hparam/train_loss": epoch_loss}
    writer.add_hparams(vars(args), final_metrics)
except Exception:
    pass

writer.close()
if args.use_wandb and (wandb is not None):
    try:
        wandb.finish()
    except Exception:
        pass
print(f"TensorBoard logs saved to: {run_dir}")
