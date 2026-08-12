#!/usr/bin/env python3
"""
plot_variance_distributions.py

Per-group variance distribution plots matching the ViT S-VELCRO style:
  1. Per-block variance histograms   (skyblue filled, one PNG per block per group)
  2. Per-head variance histograms    (skyblue filled, one PNG per head per group)
  3. Cross-group block comparison    (step overlay, one PNG per block)
  4. Cross-group head comparison     (step overlay, one PNG per head)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sam_velcro_pkg.constants import GLOBAL_ATTN_INDEXES, NUM_BLOCKS, NUM_HEADS


BINS = 200


def _p99_range(arr: np.ndarray) -> float:
    p99 = float(np.percentile(arr, 99))
    if p99 <= 0:
        p99 = float(np.max(arr)) + 1e-8
    return p99


# ── 1. Per-block variance histograms (single group, skyblue filled) ──────────

def plot_block_variance(profile_dir: Path, group: str, out_dir: Path) -> None:
    group_dir = profile_dir / group
    block_dir = out_dir / "block_variance" / group
    block_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        arrs = []
        for h in range(NUM_HEADS):
            vp = group_dir / f"block{b}_head{h}_variance.npy"
            if vp.exists():
                arrs.append(np.load(vp).ravel())
        if not arrs:
            continue

        arr = np.concatenate(arrs)
        p99 = _p99_range(arr)
        block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=BINS, range=(0, p99), color='skyblue', edgecolor='black')
        plt.title(f"Block Variance — block {b} [{block_type}] ({group})")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(block_dir / f"{group}_block{b}_variance.png", dpi=150)
        plt.close()

    print(f"  [block_variance] {group}: saved to {block_dir}")


# ── 2. Per-head variance histograms (single group, skyblue filled) ───────────

def plot_head_variance(profile_dir: Path, group: str, out_dir: Path) -> None:
    group_dir = profile_dir / group
    head_dir = out_dir / "head_variance" / group
    head_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            vp = group_dir / f"block{b}_head{h}_variance.npy"
            if not vp.exists():
                continue

            arr = np.load(vp).ravel()
            if arr.size == 0:
                continue

            p99 = _p99_range(arr)
            block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

            plt.figure(figsize=(6, 4))
            plt.hist(arr, bins=BINS, range=(0, p99), color='skyblue', edgecolor='black')
            plt.title(f"Head Variance — block {b} head {h} [{block_type}] ({group})")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(head_dir / f"{group}_block{b}_head{h}_variance.png", dpi=150)
            plt.close()

    print(f"  [head_variance] {group}: saved to {head_dir}")


# ── 3. Cross-group block comparison (step histograms overlaid) ───────────────

def plot_block_comparison(profile_dir: Path, groups: list, out_dir: Path) -> None:
    comp_dir = out_dir / "comparison" / "blocks"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        data = {}
        for g in groups:
            arrs = []
            for h in range(NUM_HEADS):
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    arrs.append(np.load(vp).ravel())
            if arrs:
                data[g] = np.concatenate(arrs)

        if not data:
            continue

        all_vals = np.concatenate(list(data.values()))
        p99 = _p99_range(all_vals)
        block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

        plt.figure(figsize=(7, 5))
        for g, arr in data.items():
            plt.hist(arr, bins=BINS, range=(0, p99), histtype='step', label=g, linewidth=1.5)
        plt.title(f"Block Variance Comparison — block {b} [{block_type}]")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comp_dir / f"compare_block{b}.png", dpi=150)
        plt.close()

    print(f"  [comparison/blocks] Saved to {comp_dir}")


# ── 4. Cross-group head comparison (step histograms overlaid) ────────────────

def plot_head_comparison(profile_dir: Path, groups: list, out_dir: Path) -> None:
    comp_dir = out_dir / "comparison" / "heads"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            data = {}
            for g in groups:
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    arr = np.load(vp).ravel()
                    if arr.size > 0:
                        data[g] = arr

            if not data:
                continue

            all_vals = np.concatenate(list(data.values()))
            p99 = _p99_range(all_vals)
            block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

            plt.figure(figsize=(7, 5))
            for g, arr in data.items():
                plt.hist(arr, bins=BINS, range=(0, p99), histtype='step', label=g, linewidth=1.5)
            plt.title(f"Head Variance Comparison — block {b} head {h} [{block_type}]")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(comp_dir / f"compare_block{b}_head{h}.png", dpi=150)
            plt.close()

    print(f"  [comparison/heads] Saved to {comp_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-block and per-head variance distributions (ViT S-VELCRO style)"
    )
    parser.add_argument("--profile_dir", required=True,
                        help="Path to profile directory (e.g. results_sam_velcro/profile)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated groups (default: auto-detect from profile_dir)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <profile_dir>/../plots/variance)")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        raise FileNotFoundError(f"Profile directory not found: {profile_dir}")

    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]
    else:
        groups = sorted([
            d.name for d in profile_dir.iterdir()
            if d.is_dir() and (d / "block0_head0_variance.npy").exists()
        ])

    if not groups:
        print("No profiled groups found.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else profile_dir.parent / "plots" / "variance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Groups: {groups}")
    print(f"Output: {out_dir}")
    print()

    # 1 + 2: Per-group individual histograms
    for g in groups:
        print(f"[1/4] Per-block variance — {g}")
        plot_block_variance(profile_dir, g, out_dir)

        print(f"[2/4] Per-head variance — {g}")
        plot_head_variance(profile_dir, g, out_dir)

    # 3 + 4: Cross-group comparisons (need ≥2 groups)
    if len(groups) >= 2:
        print(f"\n[3/4] Cross-group block comparison ({len(groups)} groups)")
        plot_block_comparison(profile_dir, groups, out_dir)

        print(f"[4/4] Cross-group head comparison ({len(groups)} groups)")
        plot_head_comparison(profile_dir, groups, out_dir)
    else:
        print("\n[3/4] Skipped (need ≥2 groups for comparison)")
        print("[4/4] Skipped (need ≥2 groups for comparison)")

    total_block = len(groups) * NUM_BLOCKS
    total_head = len(groups) * NUM_BLOCKS * NUM_HEADS
    print(f"\nDone! Generated up to {total_block} block + {total_head} head plots")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
