# Confidence: 95%
# Rationale: Loader now accepts full ckpts, encoder bundles, and raw state_dicts; architecture is inferred from checkpoint meta when present.
#            This revision adds robust LR scheduling for the linear probe (onecycle/cosine/step/const) and weight-decay handling.

import argparse
import os
import sys
import math
from typing import Tuple, Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from timm.models.vision_transformer import Block


# -------- Fixed 2D sin-cos positional embeddings --------
def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half_dim = embed_dim // 2
    freq_seq = torch.arange(half_dim, dtype=torch.float32, device=positions.device)
    freq_seq = 1.0 / (10000 ** (freq_seq / half_dim))
    args = positions.float().unsqueeze(1) * freq_seq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int = None) -> torch.Tensor:
    if grid_w is None:
        grid_w = grid_h
    assert embed_dim % 2 == 0, "embed_dim must be even"
    yy = torch.arange(grid_h, dtype=torch.float32)
    xx = torch.arange(grid_w, dtype=torch.float32)
    pos_y = yy.repeat_interleave(grid_w)
    pos_x = xx.repeat(grid_h)
    dim_half = embed_dim // 2
    emb_y = _get_1d_sincos_pos_embed(dim_half, pos_y)
    emb_x = _get_1d_sincos_pos_embed(dim_half, pos_x)
    emb = torch.cat([emb_y, emb_x], dim=1)
    return emb.unsqueeze(0)


def parse_args():
    p = argparse.ArgumentParser("Linear probe on Galaxy10 with a frozen MAE encoder")

    # data
    p.add_argument("--dataset", type=str, default="matthieulel/galaxy10_decals")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # encoder arch (may be overridden by checkpoint meta)
    p.add_argument("--emb_dim", type=int, default=384)
    p.add_argument("--enc_depth", type=int, default=12)
    p.add_argument("--enc_heads", type=int, default=6)
    p.add_argument("--respect_ckpt_meta", action="store_true",
                   help="If set, override emb_dim/depth/heads from checkpoint meta/args when available")

    # load weights
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to full model checkpoint (.ckpt/.pth) or encoder bundle (.pth)")

    # probe optimization
    p.add_argument("--probe_epochs", type=int, default=200)
    p.add_argument("--probe_lr", type=float, default=1e-2,
                   help="Base LR (used directly for const/step/cosine; initial_lr for cosine warmup)")
    p.add_argument("--probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--probe_batch_size", type=int, default=1024)

    # LR scheduling
    p.add_argument("--schedule", type=str, default="onecycle",
                   choices=["onecycle", "cosine", "step", "const"])
    p.add_argument("--max_lr", type=float, default=None,
                   help="Peak LR for onecycle. If None, defaults to 0.05 for bs=1024 scaled linearly.")
    p.add_argument("--eta_min", type=float, default=1e-4,
                   help="Final min LR for cosine schedule.")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Warmup epochs for cosine schedule.")
    p.add_argument("--nesterov", action="store_true", help="Enable Nesterov momentum for SGD")

    # misc
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return p.parse_args()


def select_device(pref: str) -> str:
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return pref


# -------- Minimal encoder for feature extraction --------
class MAEViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, emb_dim: int, enc_depth: int, enc_heads: int, use_conv_patch_embed: bool = False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.in_dim = 3 * patch_size * patch_size

        self.use_conv = use_conv_patch_embed
        if self.use_conv:
            # Conv-based embed: kernel=stride=patch_size, then flatten to tokens
            self.patch_embed_conv = nn.Conv2d(
                in_channels=3,
                out_channels=emb_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
            # Use fixed 2D sin-cos positional embeddings (computed at runtime); store as non-persistent buffer
            grid_size = img_size // patch_size
            self.register_buffer(
                "pos_embed_enc",
                get_2d_sincos_pos_embed(emb_dim, grid_size, grid_size),
                persistent=False,
            )
        else:
            # Linear patch embed + learned pos embed (legacy)
            self.patch_embed = nn.Linear(self.in_dim, emb_dim)
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
            nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)

        self.encoder = nn.ModuleList([
            Block(dim=emb_dim, num_heads=enc_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(emb_dim)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(B, -1, C * p * p)

    @torch.no_grad()
    def extract_features(self, imgs: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        if self.use_conv:
            x = self.patch_embed_conv(imgs)
            B, C, H_, W_ = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, H_ * W_, C)
            x = x + self.pos_embed_enc
        else:
            patches = self.patchify(imgs)
            x = self.patch_embed(patches)
            x = x + self.pos_embed_enc
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        if pool == "mean":
            x = x.mean(dim=1)
        return x


# -------- Data utilities --------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def safe_train_test_split(hf_ds, test_size: float, seed: int):
    try:
        return hf_ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column="label")
    except Exception:
        return hf_ds.train_test_split(test_size=test_size, seed=seed)

def build_splits(ds_dict, val_frac: float, seed: int):
    has_test = "test" in ds_dict
    has_val = "validation" in ds_dict
    hf_train = ds_dict["train"]
    hf_val = ds_dict["validation"] if has_val else None
    hf_test = ds_dict["test"] if has_test else None
    if hf_val is None:
        split = safe_train_test_split(hf_train, test_size=val_frac, seed=seed)
        hf_train, hf_val = split["train"], split["test"]
    if hf_test is None:
        split = safe_train_test_split(hf_train, test_size=val_frac, seed=seed)
        hf_train, hf_test = split["train"], split["test"]
    return hf_train, hf_val, hf_test


# -------- Checkpoint introspection & loader --------
def _read_ckpt_file(path: str, device: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)

def _infer_arch_from_payload(payload: Any, fallback: Dict[str, int]) -> Dict[str, int]:
    # Try new encoder bundle
    if isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict):
        m = payload["meta"]
        keys = ["emb_dim", "enc_depth", "enc_heads", "img_size", "patch_size"]
        if all(k in m for k in keys):
            return {
                "emb_dim": int(m["emb_dim"]),
                "enc_depth": int(m["enc_depth"]),
                "enc_heads": int(m["enc_heads"]),
                "img_size": int(m["img_size"]),
                "patch_size": int(m["patch_size"]),
            }
    # Try full training ckpt with args
    if isinstance(payload, dict) and "args" in payload and isinstance(payload["args"], dict):
        a = payload["args"]
        for k in ("emb_dim", "enc_depth", "enc_heads", "img_size", "patch_size"):
            if k not in a:
                break
        else:
            return {
                "emb_dim": int(a["emb_dim"]),
                "enc_depth": int(a["enc_depth"]),
                "enc_heads": int(a["enc_heads"]),
                "img_size": int(a["img_size"]),
                "patch_size": int(a["patch_size"]),
            }
    return fallback

def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    # New encoder bundle format
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]

    # Full training ckpt
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]

    # Raw sd
    if isinstance(payload, dict):
        return payload

    raise RuntimeError("Unrecognized checkpoint structure.")

def _has_encoder_bundle_keys(sd: Dict[str, torch.Tensor]) -> bool:
    has_patch = any(k.startswith("patch_embed.") or k.startswith("patch_embed_conv.") for k in sd.keys())
    has_pos   = ("pos_embed_enc" in sd)  # optional for conv+fixed sincos, but present in bundle as buffer name
    has_norm  = any(k.startswith("enc_norm.") for k in sd.keys())
    return has_patch and has_norm

def load_encoder_weights(model: MAEViT, ckpt_path: str, device: str, strict_verify: bool = True) -> None:
    payload = _read_ckpt_file(ckpt_path, device)
    sd = _extract_state_dict(payload)

    # If this looks like a full model state_dict, filter to encoder bundle keys
    filtered = {}
    for k, v in sd.items():
        if (
            k.startswith("patch_embed.")
            or k.startswith("patch_embed_conv.")
            or k == "pos_embed_enc"
            or k.startswith("encoder.")
            or k.startswith("enc_norm.")
        ):
            filtered[k] = v

    if not _has_encoder_bundle_keys(filtered):
        # If only encoder.* exist, this is insufficient and would freeze random patch_embed/pos_embed
        if strict_verify:
            missing = ["patch_embed.*", "pos_embed_enc", "enc_norm.*"]
            raise RuntimeError(
                "Checkpoint does not contain encoder bundle keys. "
                f"Found keys: {list(sd.keys())[:8]}...; missing one of {missing}. "
                "Re-export your pretrain with an encoder bundle or pass a full training checkpoint."
            )
        else:
            # Last resort: load what we can (not recommended)
            filtered = sd

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Sanity: require patch embed (linear or conv) and enc_norm; pos_embed_enc is optional for conv+fixed sincos
    critical_missing = [k for k in missing if k.startswith("patch_embed.") or k.startswith("patch_embed_conv.") or k.startswith("enc_norm.")]
    if strict_verify and len(critical_missing) > 0:
        raise RuntimeError(f"Critical encoder parameters not loaded: {critical_missing}")


# -------- Feature extraction over a DataLoader --------
@torch.no_grad()
def extract_split_features(model: MAEViT, loader: DataLoader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        z = model.extract_features(xb)          # [B, emb_dim]
        z = F.normalize(z, dim=1)
        feats.append(z.cpu())
        labels.append(yb.cpu())
    return torch.cat(feats, 0), torch.cat(labels, 0)


# -------- Linear probe training on cached features (with schedulers) --------
def train_probe(train_feats: torch.Tensor,
                train_labels: torch.Tensor,
                num_classes: int,
                emb_dim: int,
                epochs: int,
                lr: float,
                wd: float,
                device: str,
                val_feats: Optional[torch.Tensor] = None,
                val_labels: Optional[torch.Tensor] = None,
                schedule: str = "onecycle",
                max_lr: Optional[float] = None,
                eta_min: float = 1e-4,
                warmup_epochs: int = 5,
                batch_size: int = 1024,
                nesterov: bool = False) -> nn.Module:
    probe = nn.Linear(emb_dim, num_classes).to(device)

    # Weight decay on weights only (bias has wd=0)
    wd_params, nowd_params = [], []
    for n, p in probe.named_parameters():
        (wd_params if p.ndim > 1 else nowd_params).append(p)

    # For onecycle we initialize optimizer with a placeholder LR; scheduler will shape it.
    init_lr = (max_lr if (schedule == "onecycle" and max_lr is not None) else lr)
    opt = torch.optim.SGD(
        [{"params": wd_params, "weight_decay": wd},
         {"params": nowd_params, "weight_decay": 0.0}],
        lr=init_lr, momentum=0.9, nesterov=nesterov
    )
    ce = nn.CrossEntropyLoss()

    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- Scheduler setup ---
    if schedule == "onecycle":
        # Default max_lr ~0.05 for bs=1024; scale linearly with batch_size
        if max_lr is None:
            max_lr = 0.05 * (batch_size / 1024.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.1, div_factor=25.0, final_div_factor=10.0
        )
        step_per_batch = True
    elif schedule == "cosine":
        warmup_steps = warmup_epochs * max(1, len(train_loader))
        total_steps  = epochs * max(1, len(train_loader))
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / float(max(1, warmup_steps))
            prog = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # Scale from 1.0 -> eta_min/lr
            return 0.5 * (1.0 + math.cos(math.pi * prog)) * (1.0 - eta_min / lr) + (eta_min / lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        step_per_batch = True
    elif schedule == "step":
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[int(0.6*epochs), int(0.9*epochs)], gamma=0.1
        )
        step_per_batch = False
    else:  # "const"
        sched = None
        step_per_batch = False

    for ep in range(1, epochs + 1):
        probe.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = probe(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None and step_per_batch:
                sched.step()
            ep_loss += loss.item()
        ep_loss /= max(1, len(train_loader))

        # Validation
        if val_feats is not None and val_labels is not None:
            with torch.no_grad():
                probe.eval()
                logits = probe(val_feats.to(device))
                val_acc = (logits.argmax(1).cpu() == val_labels).float().mean().item()
            print(f"[Probe] epoch {ep:03d} | lr {opt.param_groups[0]['lr']:.5f} | train_loss {ep_loss:.4f} | val@1 {val_acc:.4f}")
        else:
            print(f"[Probe] epoch {ep:03d} | lr {opt.param_groups[0]['lr']:.5f} | train_loss {ep_loss:.4f}")

        if sched is not None and not step_per_batch:
            sched.step()

    return probe


@torch.no_grad()
def evaluate_probe(probe: nn.Module, feats: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    probe.eval()
    logits = probe(feats.to(device))
    acc = (logits.argmax(1).cpu() == labels).float().mean().item()
    return acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    # Load raw payload once to infer arch if requested
    payload = torch.load(args.ckpt_path, map_location=device)
    if args.respect_ckpt_meta:
        inferred = _infer_arch_from_payload(payload, {
            "emb_dim": args.emb_dim,
            "enc_depth": args.enc_depth,
            "enc_heads": args.enc_heads,
            "img_size": args.img_size,
            "patch_size": args.patch_size,
        })
        args.emb_dim    = inferred["emb_dim"]
        args.enc_depth  = inferred["enc_depth"]
        args.enc_heads  = inferred["enc_heads"]
        args.img_size   = inferred["img_size"]
        args.patch_size = inferred["patch_size"]

    # 1) Data
    tf = build_transforms(args.img_size)
    ds = load_dataset(args.dataset)

    class Galaxy(torch.utils.data.Dataset):
        def __init__(self, split, tfm):
            self.base = split
            self.tfm = tfm
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            img = self.base[i]["image"]
            y = int(self.base[i]["label"])
            return self.tfm(img), y

    train_split, val_split, test_split = build_splits(ds, args.val_split, args.seed)
    train_set = Galaxy(train_split, tf)
    val_set   = Galaxy(val_split, tf)
    test_set  = Galaxy(test_split, tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))

    # 2) Model (encoder only)
    # Detect conv-based patch embed from checkpoint keys
    sd = _extract_state_dict(payload)
    use_conv = any(k.startswith("patch_embed_conv.") for k in sd.keys())

    model = MAEViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        use_conv_patch_embed=use_conv,
    ).to(device)

    # 3) Load weights (frozen)
    try:
        load_encoder_weights(model, args.ckpt_path, device, strict_verify=True)
    except Exception as e:
        print(f"[Error] Failed to load encoder weights: {e}", file=sys.stderr)
        sys.exit(2)

    for p in model.parameters():
        p.requires_grad = False

    # 4) Extract features
    print("Extracting features...")
    tr_feats, tr_labels = extract_split_features(model, train_loader, device)
    va_feats, va_labels = extract_split_features(model, val_loader, device)
    te_feats, te_labels = extract_split_features(model, test_loader, device)

    num_classes = int(max(tr_labels.max().item(), va_labels.max().item(), te_labels.max().item())) + 1
    emb_dim = tr_feats.shape[1]

    # 5) Train probe with chosen schedule
    probe = train_probe(tr_feats, tr_labels, num_classes, emb_dim,
                        epochs=args.probe_epochs, lr=args.probe_lr,
                        wd=args.probe_weight_decay, device=device,
                        val_feats=va_feats, val_labels=va_labels,
                        schedule=args.schedule, max_lr=args.max_lr,
                        eta_min=args.eta_min, warmup_epochs=args.warmup_epochs,
                        batch_size=args.probe_batch_size, nesterov=args.nesterov)

    # 6) Evaluate on test
    test_acc = evaluate_probe(probe, te_feats, te_labels, device)
    print(f"\nTest top-1 accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)