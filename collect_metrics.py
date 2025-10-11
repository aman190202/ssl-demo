#!/usr/bin/env python3
"""
Collect & plot MAE experiment metrics across many run folders,
STRICTLY from an existing metrics.csv. Any folder without metrics.csv
is skipped (no TB/W&B reconstruction).

Usage:
  python collect_metrics_csv_only.py --roots runs logs --outdir analysis_plots --overlay-val
"""

import argparse
import os
import sys
from typing import List, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


def find_run_dirs_with_csv(roots: List[str]) -> List[str]:
    run_dirs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, _ in os.walk(root):
            mpath = os.path.join(dirpath, "metrics.csv")
            if os.path.isfile(mpath):
                run_dirs.append(dirpath)
    return sorted(set(os.path.normpath(d) for d in run_dirs))


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c == "source":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target columns:
      epoch, train_loss, val_knn_top1, test_knn_top1, val_offline_top1, test_offline_top1, val_online_top1
    Accepts either the canonical names or the TB-style names from your training script.
    """
    rename_map = {
        "train/loss_epoch": "train_loss",
        "train/mse_epoch": "train_loss",  # if both exist, we’ll keep train/loss_epoch
        "val/knn_top1": "val_knn_top1",
        "test/knn_top1": "test_knn_top1",
        "val/offline_top1": "val_offline_top1",
        "test/offline_top1": "test_offline_top1",
        "val/online_top1": "val_online_top1",
    }

    # Prefer train/loss_epoch over train/mse_epoch if both exist
    if "train/loss_epoch" in df.columns and "train/mse_epoch" in df.columns:
        df = df.drop(columns=["train/mse_epoch"], errors="ignore")

    df = df.rename(columns=rename_map)

    needed = [
        "epoch",
        "train_loss",
        "val_knn_top1",
        "test_knn_top1",
        "val_offline_top1",
        "test_offline_top1",
        "val_online_top1",
    ]
    for n in needed:
        if n not in df.columns:
            df[n] = pd.NA

    # Keep only what we need (plus anything extra is ignored)
    df = df[needed]
    df = coerce_numeric(df)
    # Clean epoch
    df = df[pd.notna(df["epoch"])].copy()
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_run_metrics(df: pd.DataFrame, run_name: str, outdir: str):
    ensure_outdir(outdir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=120)

    ax = axes[0, 0]
    if df["train_loss"].notna().any():
        ax.plot(df["epoch"], df["train_loss"])
    ax.set_title("Train Reconstruction Loss (per-epoch)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")

    ax = axes[0, 1]
    any_val = False
    if df["val_knn_top1"].notna().any():
        ax.plot(df["epoch"], df["val_knn_top1"], label="val kNN@1")
        any_val = True
    if df["val_online_top1"].notna().any():
        ax.plot(df["epoch"], df["val_online_top1"], linestyle="--", label="val online probe@1")
        any_val = True
    ax.set_title("Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1")
    if any_val:
        ax.legend(loc="best")

    ax = axes[1, 0]
    if df["test_knn_top1"].notna().any():
        ax.plot(df["epoch"], df["test_knn_top1"])
    ax.set_title("Test kNN@1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1")

    ax = axes[1, 1]
    any_off = False
    if df["val_offline_top1"].notna().any():
        ax.plot(df["epoch"], df["val_offline_top1"], label="val offline probe@1")
        any_off = True
    if df["test_offline_top1"].notna().any():
        ax.plot(df["epoch"], df["test_offline_top1"], label="test offline probe@1")
        any_off = True
    ax.set_title("Offline probe (if logged)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1")
    if any_off:
        ax.legend(loc="best")

    fig.suptitle(run_name, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(outdir, f"{run_name}_metrics.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_run(df: pd.DataFrame, run_name: str) -> Dict:
    best_val = float(pd.to_numeric(df["val_knn_top1"], errors="coerce").max(skipna=True))
    best_test = float(pd.to_numeric(df["test_knn_top1"], errors="coerce").max(skipna=True))
    best_val_off = float(pd.to_numeric(df["val_offline_top1"], errors="coerce").max(skipna=True))
    best_test_off = float(pd.to_numeric(df["test_offline_top1"], errors="coerce").max(skipna=True))
    final_loss = (
        float(pd.to_numeric(df["train_loss"], errors="coerce").dropna().tail(1).values[0])
        if df["train_loss"].notna().any()
        else float("nan")
    )
    return {
        "run": run_name,
        "epochs_logged": int(df["epoch"].max()),
        "best_val_knn_top1": best_val,
        "best_test_knn_top1": best_test,
        "best_val_offline_top1": best_val_off,
        "best_test_offline_top1": best_test_off,
        "final_train_loss": final_loss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["runs"], help="Root folders containing run subdirectories")
    ap.add_argument("--outdir", type=str, default="analysis_plots", help="Where to save figures and summary")
    ap.add_argument("--overlay-val", action="store_true", help="Also produce an overlay figure of val kNN across runs")
    args = ap.parse_args()

    run_dirs = find_run_dirs_with_csv(args.roots)
    if not run_dirs:
        print("No runs with metrics.csv found under the provided roots.")
        sys.exit(1)

    ensure_outdir(args.outdir)

    summaries = []
    overlay = []

    for run_dir in run_dirs:
        run_name = os.path.basename(os.path.normpath(run_dir))
        mpath = os.path.join(run_dir, "metrics.csv")

        try:
            raw = pd.read_csv(mpath)
        except Exception as e:
            print(f"[SKIP] {run_name}: failed to read metrics.csv ({e})")
            continue

        df = normalize_schema(raw)
        if df.empty:
            print(f"[SKIP] {run_name}: metrics.csv has no usable rows.")
            continue

        try:
            plot_run_metrics(df, run_name, args.outdir)
        except Exception as e:
            print(f"[WARN] Plotting failed for {run_name}: {e}")

        try:
            summaries.append(summarize_run(df, run_name))
        except Exception as e:
            print(f"[WARN] Summarize failed for {run_name}: {e}")

        if args.overlay_val and df["val_knn_top1"].notna().any():
            overlay.append((run_name, df[["epoch", "val_knn_top1"]].copy()))

    # Summary CSV
    if summaries:
        summary_df = pd.DataFrame(summaries).sort_values("best_val_knn_top1", ascending=False)
        out_summary = os.path.join(args.outdir, "summary.csv")
        summary_df.to_csv(out_summary, index=False)
        print(f"[OK] Wrote summary to {out_summary}")

    # Optional overlay
    if overlay:
        fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
        for run_name, dfi in overlay:
            ax.plot(dfi["epoch"], dfi["val_knn_top1"], label=run_name)
        ax.set_title("Validation kNN@1 across runs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Top-1")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "overlay_val_knn.png"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()