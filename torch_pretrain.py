# ============================================================
# 1. Imports and setup
# ============================================================


import argparse
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils as vutils
from datasets import load_dataset
from timm.models.vision_transformer import Block
from torch.utils.tensorboard import SummaryWriter


try:
    import wandb
except Exception:
    wandb = None


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


def parse_args():
    parser = argparse.ArgumentParser("MAE pretraining on Galaxy10 with kNN eval and checkpoints")
    # data
    parser.add_argument("--dataset", type=str, default="matthieulel/galaxy10_decals")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.1, help="val split fraction when dataset has no validation")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--dec_dim", type=int, default=512)
    parser.add_argument("--enc_depth", type=int, default=6)
    parser.add_argument("--dec_depth", type=int, default=4)
    parser.add_argument("--enc_heads", type=int, default=3)
    parser.add_argument("--dec_heads", type=int, default=8)
    parser.add_argument("--mask_ratio", type=float, default=0.75)

    # optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)

    # eval
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--knn_t", type=float, default=0.07, help="temperature for kNN soft voting")
    parser.add_argument("--eval_every", type=int, default=1)

    # io
    parser.add_argument("--run_dir", type=str, default="runs/mae_galaxy10")
    parser.add_argument("--save_every", type=int, default=5, help="save encoder every N epochs")
    parser.add_argument("--vis_every", type=int, default=10, help="log reconstruction panels every N epochs")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume; if empty, auto-resume from run_dir/last.ckpt if present")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="mae_galaxy10", help="W&B project")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity (team or username)")
    parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional)")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B mode")

    args = parser.parse_args() if hasattr(__builtins__, "__IPYTHON__") is False else parser.parse_args("")
    return args

# ============================================================
# 2. Dataset (Galaxy10) and transforms
# ============================================================
args = parse_args()

IMG_SIZE = args.img_size
PATCH_SIZE = args.patch_size
MASK_RATIO = args.mask_ratio

g = torch.Generator()
g.manual_seed(args.seed)

base_tf = transforms.Compose([
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
    """Try stratified split on 'label'; fallback to random split if unsupported."""
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
        # if no test, further split from train (small fraction)
        split = safe_train_test_split(hf_train, test_size=args.val_split, seed=args.seed)
        hf_train, hf_test = split["train"], split["test"]
    return hf_train, hf_val, hf_test


hf_train, hf_val, hf_test = build_splits(ds)

train_set = Galaxy10(hf_train, base_tf)
val_set = Galaxy10(hf_val, base_tf)
test_set = Galaxy10(hf_test, base_tf)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device == "cuda"))
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

# ============================================================
# 3. Patching utilities
# ============================================================
def patchify(imgs, p=PATCH_SIZE):
    B, C, H, W = imgs.shape
    patches = imgs.unfold(2, p, p).unfold(3, p, p)
    patches = patches.permute(0,2,3,1,4,5).contiguous()
    return patches.view(B, -1, C*p*p)

def unpatchify(patches, p=PATCH_SIZE):
    B, N, PP = patches.shape
    C = 3
    h = w = IMG_SIZE // p
    x = patches.view(B, h, w, C, p, p).permute(0,3,1,4,2,5).contiguous()
    return x.view(B, C, h*p, w*p)

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

# ============================================================
# 4. MAE with ViT encoder
# ============================================================
class MAEViT(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE,
                 patch_size=PATCH_SIZE,
                 emb_dim=192,
                 dec_dim=512,
                 enc_depth=6,
                 dec_depth=4,
                 enc_heads=3,
                 dec_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.in_dim = 3 * patch_size * patch_size

        # Patch embed
        self.patch_embed = nn.Linear(self.in_dim, emb_dim)

        # Positional embeddings
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)

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
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.num_patches, dec_dim))
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

        self.decoder = nn.ModuleList([
            Block(dim=dec_dim, num_heads=dec_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, self.in_dim)

    def encode(self, imgs):
        patches = patchify(imgs)
        x = self.patch_embed(patches)
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
        enc_tokens, patches, mask, ids_restore = self.encode(imgs)
        pred = self.decode(enc_tokens, ids_restore)
        loss = ((pred - patches) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return pred, loss

    @torch.no_grad()
    def forward_with_intermediates(self, imgs):
        """For logging: returns pred, patches, mask."""
        enc_tokens, patches, mask, ids_restore = self.encode(imgs)
        pred = self.decode(enc_tokens, ids_restore)
        return pred, patches, mask

    @torch.no_grad()
    def extract_features(self, imgs, pool: str = "mean"):
        """Encode images without masking for representation learning and probing.

        pool: 'mean' to average tokens.
        Returns: [B, emb_dim] features.
        """
        patches = patchify(imgs)
        x = self.patch_embed(patches)
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
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

def denorm(x):
    # x in normalized space; return [0,1] clipped
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return x.clamp(0,1)

@torch.no_grad()
def log_recon_samples(writer, model, imgs, global_step, n_show=8):
    model.eval()
    imgs = imgs[:n_show].to(device)  # [B,3,H,W]
    pred, patches, mask = model.forward_with_intermediates(imgs)  # pred/patches: [B,N,768], mask: [B,N]

    # masked input: keep visible, zero masked
    masked_patches = patches * (1 - mask.unsqueeze(-1))
    masked_img = unpatchify(masked_patches)

    # full reconstruction
    recon_full = unpatchify(pred)

    # blended recon: use GT for visible, pred for masked
    blended_patches = patches * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
    recon_blended = unpatchify(blended_patches)

    # make grids (denormalize for viewing)
    grid_input  = vutils.make_grid(denorm(imgs), nrow=min(4, n_show))
    grid_masked = vutils.make_grid(denorm(masked_img), nrow=min(4, n_show))
    grid_blend  = vutils.make_grid(denorm(recon_blended), nrow=min(4, n_show))
    grid_full   = vutils.make_grid(denorm(recon_full), nrow=min(4, n_show))

    writer.add_image("00_input",  grid_input,  global_step)
    writer.add_image("01_masked_input", grid_masked, global_step)
    writer.add_image("02_recon_blended(masked_filled)", grid_blend, global_step)
    writer.add_image("03_recon_full", grid_full, global_step)


@torch.no_grad()
def log_recon_samples_wandb(model, imgs, global_step, n_show=8):
    if not (args.use_wandb and (wandb is not None)):
        return
    model.eval()
    imgs = imgs[:n_show].to(device)
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

    # sample-wise comparison table: original vs masked vs predicted
    table = wandb.Table(columns=["original", "masked", "predicted"])
    imgs_den = denorm(imgs).cpu()
    masked_den = denorm(masked_img).cpu()
    recon_den = denorm(recon_full).cpu()
    for i in range(imgs.shape[0]):
        table.add_data(
            wandb.Image(imgs_den[i]),
            wandb.Image(masked_den[i]),
            wandb.Image(recon_den[i]),
        )
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


@torch.no_grad()
def knn_top1(train_feats, train_labels, query_feats, query_labels, k: int, T: float, num_classes: int):
    sims = query_feats @ train_feats.t()  # cosine similarity because features are normalized
    top_sim, top_idx = sims.topk(k, dim=1)
    top_labels = train_labels[top_idx]  # [Nq, k]
    weights = torch.exp(top_sim / T)
    # one-hot vote with weights
    votes = torch.zeros(query_feats.size(0), num_classes, device=query_feats.device)
    votes.scatter_add_(1, top_labels, weights)
    preds = votes.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()
    return acc


# ============================================================
# 7) Training loop with epochs + TensorBoard + Checkpoints
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

# Optional Weights & Biases init
wandb_run = None
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
        }
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

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    model.load_state_dict(ckpt["model"])  # full model
    opt.load_state_dict(ckpt["optimizer"]) if "optimizer" in ckpt else None
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


best_val = -1.0
try:
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            _, loss = model(xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            if args.use_wandb and (wandb is not None):
                try:
                    wandb.log({"train/loss_step": loss.item(), "global_step": global_step}, step=global_step)
                except Exception:
                    pass
            global_step += 1

        epoch_loss /= len(train_loader)
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
        if args.use_wandb and (wandb is not None):
            try:
                wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch}, step=global_step)
            except Exception:
                pass

        # Visualization panels
        if epoch % args.vis_every == 0:
            xb_vis, _ = next(iter(train_loader))
            log_recon_samples(writer, model, xb_vis, global_step, n_show=8)
            if args.use_wandb and (wandb is not None):
                try:
                    log_recon_samples_wandb(model, xb_vis, global_step, n_show=8)
                except Exception:
                    pass

        # Eval kNN
        val_acc = float("nan")
        test_acc = float("nan")
        if epoch % args.eval_every == 0:
            train_feats, train_labels = extract_features_for_loader(model, train_loader)
            num_classes = int(train_labels.max().item()) + 1
            val_feats, val_labels = extract_features_for_loader(model, val_loader)
            test_feats, test_labels = extract_features_for_loader(model, test_loader)

            val_acc = knn_top1(train_feats, train_labels, val_feats, val_labels, k=args.knn_k, T=args.knn_t, num_classes=num_classes)
            test_acc = knn_top1(train_feats, train_labels, test_feats, test_labels, k=args.knn_k, T=args.knn_t, num_classes=num_classes)

            writer.add_scalar("val/knn_top1", val_acc, epoch)
            writer.add_scalar("test/knn_top1", test_acc, epoch)
            if args.use_wandb and (wandb is not None):
                try:
                    wandb.log({
                        "val/knn_top1": val_acc,
                        "test/knn_top1": test_acc,
                        "epoch": epoch,
                    }, step=global_step)
                except Exception:
                    pass

            # track best
            if val_acc > best_val:
                best_val = val_acc
                save_checkpoint(epoch, is_best=True)

        # Save metrics row
        append_metrics_csv(metrics_csv, {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_knn_top1": val_acc,
            "test_knn_top1": test_acc,
        }, metrics_header)

        # Save checkpoints
        save_checkpoint(epoch, is_best=False)
        if epoch % args.save_every == 0:
            enc_path = os.path.join(run_dir, f"encoder_epoch_{epoch:03d}.pth")
            torch.save(model.encoder.state_dict(), enc_path)
            # convenience latest encoder
            torch.save(model.encoder.state_dict(), os.path.join(run_dir, "encoder_latest.pth"))

        print(f"[Epoch {epoch:03d}] loss={epoch_loss:.4f} val@1={val_acc:.4f} test@1={test_acc:.4f}")

except KeyboardInterrupt:
    print("Training interrupted by user. Saving interrupt checkpoint...")
    torch.save({
        "epoch": epoch if 'epoch' in locals() else 0,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "args": vars(args),
    }, os.path.join(run_dir, "interrupt.ckpt"))
    torch.save(model.encoder.state_dict(), os.path.join(run_dir, "encoder_latest.pth"))
    if args.use_wandb and (wandb is not None):
        try:
            wandb.finish()
        except Exception:
            pass
    raise

# Final saves
encoder_path = os.path.join(run_dir, "mae_encoder.pth")
full_model_path = os.path.join(run_dir, "mae_full.pth")
torch.save(model.encoder.state_dict(), encoder_path)
torch.save(model.state_dict(), full_model_path)
print(f"✅ Encoder weights saved to {encoder_path}")
print(f"✅ Full MAE model saved to {full_model_path}")

# TB hparams at end with final metrics
final_metrics = {}
try:
    # last known values from CSV row above
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