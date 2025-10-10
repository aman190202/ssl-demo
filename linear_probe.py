#!/usr/bin/env python3
# linear_probe.py
#
# Train a robust linear probe on features from your MAE encoder bundle.
# Fixes included:
#  - Uses the *same* eval transform as pretrain (Resize+CenterCrop+IN normalization)
#  - Extracts features exactly like pretrain (conv patch embed + pos + encoder + mean pool)
#  - Optional L2-normalize, then per-dim z-score using *train* stats (standardize)
#  - LBFGS (default) or AdamW training paths
#  - Robust num_classes inference (ClassLabel or from label lists)

import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset, ClassLabel
from timm.models.vision_transformer import Block


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------
# Dataset helpers (Galaxy10 or any HF image dataset with 'image','label')
# --------------------------
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, tf):
        self.base = hf_split
        self.tf = tf
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img = self.base[idx]["image"]
        y = int(self.base[idx]["label"])
        return self.tf(img), y

def safe_train_test_split(hf_ds, test_size: float, seed: int):
    try:
        return hf_ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    except Exception:
        return hf_ds.train_test_split(test_size=test_size, seed=seed)

def build_splits(dataset_dict, val_frac: float, seed: int):
    has_test = "test" in dataset_dict
    has_val  = "validation" in dataset_dict
    hf_train = dataset_dict["train"]
    hf_val   = dataset_dict["validation"] if has_val else None
    hf_test  = dataset_dict["test"] if has_test else None

    if hf_val is None:
        split = safe_train_test_split(hf_train, test_size=val_frac, seed=seed)
        hf_train, hf_val = split["train"], split["test"]
    if hf_test is None:
        split = safe_train_test_split(hf_train, test_size=val_frac, seed=seed)
        hf_train, hf_test = split["train"], split["test"]
    return hf_train, hf_val, hf_test

def infer_num_classes(hf_train, hf_val, hf_test) -> int:
    feat = hf_train.features.get("label", None)
    if isinstance(feat, ClassLabel) and feat.num_classes is not None:
        return int(feat.num_classes)
    # Fallback: compute from concatenated label lists
    labs = []
    for ds in (hf_train, hf_val, hf_test):
        labs.extend([int(x) for x in ds["label"]])
    # Assumes labels are 0..C-1; safe for Galaxy10. If your labels are not contiguous,
    # replace with: return len(set(labs)) and remap labels before training.
    return max(labs) + 1


# --------------------------
# Minimal encoder for feature extraction (matches your pretrain)
# Keeps keys: patch_embed_conv., pos_embed_enc, encoder., enc_norm.
# --------------------------
class EncoderForFeatures(nn.Module):
    def __init__(self, img_size: int, patch_size: int, emb_dim: int, enc_depth: int, enc_heads: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed_conv = nn.Conv2d(
            in_channels=3,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Will be loaded from checkpoint (buffer): shape [1, N, emb_dim]
        self.register_buffer("pos_embed_enc", torch.zeros(1, self.num_patches, emb_dim), persistent=False)

        self.encoder = nn.ModuleList([
            Block(dim=emb_dim, num_heads=enc_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(emb_dim)

    @torch.no_grad()
    def extract_features(self, imgs, pool: str = "mean"):
        x = self.patch_embed_conv(imgs)                 # [B, C, H/ps, W/ps]
        B, C, H_, W_ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H_ * W_, C)  # [B, N, C]
        x = x + self.pos_embed_enc
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        if pool == "mean":
            x = x.mean(dim=1)
        return x


# --------------------------
# Feature extraction + probe training
# --------------------------
@torch.no_grad()
def extract_bank(model: EncoderForFeatures, loader, dev: str, l2norm: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        z = model.extract_features(xb)           # [B, D]
        if l2norm:
            z = F.normalize(z, dim=1)
        feats.append(z.cpu())
        labels.append(yb)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

def standardize_feats(tr: torch.Tensor, va: torch.Tensor, te: torch.Tensor):
    mean = tr.mean(dim=0, keepdim=True)
    std  = tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (tr - mean)/std, (va - mean)/std, (te - mean)/std, mean, std

def evaluate(clf: nn.Module, feats: torch.Tensor, labels: torch.Tensor, dev: str) -> float:
    clf.eval()
    with torch.no_grad():
        logits = clf(feats.to(dev, non_blocking=True))
        pred = logits.argmax(1).cpu()
    return (pred == labels).float().mean().item()

def train_probe_lbfgs(
    trX: torch.Tensor, trY: torch.Tensor, in_dim: int, num_classes: int, dev: str,
    max_iter: int = 100
) -> nn.Module:
    clf = nn.Linear(in_dim, num_classes, bias=True).to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter, history_size=50, line_search_fn="strong_wolfe")

    trX = trX.to(dev, non_blocking=True)
    trY = trY.to(dev, non_blocking=True)

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = clf(trX)
        loss = ce(logits, trY)
        loss.backward()
        return loss

    opt.step(closure)
    return clf

def train_probe_adamw(
    trX: torch.Tensor, trY: torch.Tensor, in_dim: int, num_classes: int, dev: str,
    epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 2048
) -> nn.Module:
    clf = nn.Linear(in_dim, num_classes, bias=True).to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    ds = torch.utils.data.TensorDataset(trX, trY)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        clf.train()
        for xb, yb in dl:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            logits = clf(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return clf


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser("Robust linear probe on MAE encoder features")
    # I/O
    ap.add_argument("--encoder_path", type=str, default="runs/mae_galaxy10/encoder_latest.pth",
                    help="Path to encoder bundle saved by pretrain (encoder_*.pth or encoder_latest.pth)")
    ap.add_argument("--save_dir", type=str, default="runs/linear_probe",
                    help="Where to save the trained linear probe and stats")
    # Data
    ap.add_argument("--dataset", type=str, default="matthieulel/galaxy10_decals")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    # Probe
    ap.add_argument("--probe_mode", type=str, default="lbfgs", choices=["lbfgs", "adamw"])
    ap.add_argument("--lbfgs_max_iter", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)            # for AdamW path
    ap.add_argument("--lr", type=float, default=1e-3)             # for AdamW path
    ap.add_argument("--weight_decay", type=float, default=1e-4)   # for AdamW path
    ap.add_argument("--adamw_batch_size", type=int, default=2048)
    # Features
    ap.add_argument("--l2norm", action="store_true", help="L2-normalize features before standardization")
    ap.add_argument("--no_standardize", action="store_true", help="Disable per-dim z-scoring using train stats")
    args = ap.parse_args()

    dev = device_str()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------------- Data & transforms (match pretrain eval) ----------------
    eval_tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    ds = load_dataset(args.dataset)
    hf_train, hf_val, hf_test = build_splits(ds, args.val_split, args.seed)

    train_ds = HFDataset(hf_train, eval_tf)
    val_ds   = HFDataset(hf_val,   eval_tf)
    test_ds  = HFDataset(hf_test,  eval_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev == "cuda")
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev == "cuda")
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev == "cuda")
    )

    # ---------------- Classes ----------------
    num_classes = infer_num_classes(hf_train, hf_val, hf_test)
    print(f"[Info] num_classes={num_classes}")

    # ---------------- Load encoder bundle ----------------
    if not os.path.isfile(args.encoder_path):
        raise FileNotFoundError(f"Encoder bundle not found: {args.encoder_path}")
    bundle = torch.load(args.encoder_path, map_location="cpu")
    if "state_dict" in bundle and "meta" in bundle:
        sd = bundle["state_dict"]
        meta = bundle["meta"]
        print(f"[Info] Loaded encoder bundle with meta: {meta}")
        emb_dim    = int(meta.get("emb_dim", 384))
        enc_depth  = int(meta.get("enc_depth", 12))
        enc_heads  = int(meta.get("enc_heads", 6))
        img_size   = int(meta.get("img_size", args.img_size))
        patch_size = int(meta.get("patch_size", args.patch_size))
    else:
        sd = bundle
        enc_norm_w = sd.get("enc_norm.weight", None)
        if enc_norm_w is None:
            raise RuntimeError("State dict missing enc_norm.*; cannot infer emb_dim. Provide a bundle with 'meta'.")
        emb_dim = int(enc_norm_w.shape[0])
        img_size = args.img_size
        patch_size = args.patch_size
        enc_depth = sum([1 for k in sd.keys() if k.startswith("encoder.") and k.endswith(".mlp.fc1.weight")])
        enc_heads = 6
        print("[Warn] Using inferred architecture. Consider supplying a bundle with 'meta'.")

    model = EncoderForFeatures(
        img_size=img_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        enc_depth=enc_depth,
        enc_heads=enc_heads,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {missing[:4]}{'...' if len(missing)>4 else ''}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {unexpected[:4]}{'...' if len(unexpected)>4 else ''}")
    model.to(dev).eval()
    print(f"[Info] Encoder ready on {dev} | emb_dim={emb_dim} depth={enc_depth} heads={enc_heads} "
          f"| img/patch={img_size}/{patch_size}")

    # ---------------- Extract features ----------------
    tr_feats, tr_labels = extract_bank(model, train_loader, dev, l2norm=args.l2norm)
    va_feats, va_labels = extract_bank(model, val_loader,   dev, l2norm=args.l2norm)
    te_feats, te_labels = extract_bank(model, test_loader,  dev, l2norm=args.l2norm)

    print(f"[Features] train={tuple(tr_feats.shape)} val={tuple(va_feats.shape)} test={tuple(te_feats.shape)} "
          f"| l2norm={'yes' if args.l2norm else 'no'}")

    # ---------------- Standardize (recommended) ----------------
    if args.no_standardize:
        tr_std, va_std, te_std = tr_feats, va_feats, te_feats
        mean = torch.zeros(1, tr_feats.shape[1])
        std  = torch.ones(1, tr_feats.shape[1])
        print("[Info] Standardization disabled.")
    else:
        tr_std, va_std, te_std, mean, std = standardize_feats(tr_feats, va_feats, te_feats)
        print("[Info] Per-dim z-scored features using *train* stats.")

    in_dim = tr_std.shape[1]

    # ---------------- Train probe ----------------
    if args.probe_mode == "lbfgs":
        clf = train_probe_lbfgs(tr_std, tr_labels, in_dim, num_classes, dev, max_iter=args.lbfgs_max_iter)
    else:
        clf = train_probe_adamw(tr_std, tr_labels, in_dim, num_classes, dev,
                                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                batch_size=args.adamw_batch_size)

    # ---------------- Evaluate ----------------
    train_acc = evaluate(clf, tr_std, tr_labels, dev)
    val_acc   = evaluate(clf, va_std, va_labels, dev)
    test_acc  = evaluate(clf, te_std, te_labels, dev)
    print(f"[Probe] train@1 = {train_acc:.4f} | val@1 = {val_acc:.4f} | test@1 = {test_acc:.4f}")

    # ---------------- Save probe + stats ----------------
    out_path = os.path.join(args.save_dir, "linear_probe.pt")
    torch.save({
        "state_dict": clf.state_dict(),
        "in_dim": in_dim,
        "num_classes": num_classes,
        "mean": mean,                # for standardization at inference
        "std": std.clamp_min(1e-6),
        "l2norm": args.l2norm,
        "standardized": (not args.no_standardize),
        "encoder_path": args.encoder_path,
        "dataset": args.dataset,
        "meta": {
            "img_size": img_size, "patch_size": patch_size,
            "emb_dim": emb_dim, "enc_depth": enc_depth, "enc_heads": enc_heads
        }
    }, out_path)
    print(f"[Save] Probe saved to: {out_path}")


if __name__ == "__main__":
    main()