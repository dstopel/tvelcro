"""Visualization functions for SAM VELCRO experiments."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import GLOBAL_ATTN_INDEXES, NUM_BLOCKS, NUM_HEADS


def plot_freeze_heatmap(
    frozen_set: set,
    out_path: Path,
    title: str = "Frozen Heads Map",
) -> None:
    grid = np.zeros((NUM_BLOCKS, NUM_HEADS), dtype=np.float32)
    for b, h in frozen_set:
        grid[b, h] = 1.0

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(grid, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)

    for b in GLOBAL_ATTN_INDEXES:
        ax.axhline(y=b, color="gold", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Block Index")
    ax.set_title(title)
    ax.set_xticks(range(NUM_HEADS))
    ax.set_yticks(range(0, NUM_BLOCKS, 2))
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Active", "Frozen"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_scores_per_block(
    entries: List[dict],
    frozen_set: set,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in range(NUM_BLOCKS):
        block_entries = sorted(
            [e for e in entries if e["block"] == b],
            key=lambda e: e["rank_score"],
        )
        if not block_entries:
            continue

        heads = [e["head"] for e in block_entries]
        scores = [e["rank_score"] for e in block_entries]
        colors = ["#d62728" if (b, e["head"]) in frozen_set else "#1f77b4" for e in block_entries]
        block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(heads)), scores, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([str(h) for h in heads], fontsize=8)
        ax.set_xlabel("Head (sorted by rank)")
        ax.set_ylabel("Rank Score (Σ variance)")
        ax.set_title(f"Block {b} [{block_type}]")
        fig.tight_layout()
        fig.savefig(out_dir / f"block{b}_ranks.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_cosine_sim_curve(
    diagnostics: Dict[str, dict],
    out_path: Path,
) -> None:
    blocks = sorted(set(int(k.split("_")[0].replace("block", "")) for k in diagnostics))
    mean_sims = []
    for b in blocks:
        block_sims = [
            v["mean_cosine_sim"]
            for k, v in diagnostics.items()
            if k.startswith(f"block{b}_")
        ]
        mean_sims.append(np.mean(block_sims) if block_sims else 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if b in GLOBAL_ATTN_INDEXES else "#1f77b4" for b in blocks]
    ax.bar(blocks, mean_sims, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Mean Cosine Similarity (computed vs mean)")
    ax.set_title("Activation Diagnostics: How well does the mean approximate actual outputs?")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_cosine_map(
    cosine_map: np.ndarray,
    out_path: Path,
    title: str = "Feature Cosine Similarity (64×64)",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cosine_map, cmap="RdYlGn", vmin=0.8, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Spatial X")
    ax.set_ylabel("Spatial Y")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mask_overlay_comparison(
    image: np.ndarray,
    masks_orig: List[dict],
    masks_patch: List[dict],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(image)
    _overlay_masks(ax1, masks_orig, alpha=0.4)
    ax1.set_title(f"Original ({len(masks_orig)} masks)")
    ax1.axis("off")

    ax2.imshow(image)
    _overlay_masks(ax2, masks_patch, alpha=0.4)
    ax2.set_title(f"VELCRO Patched ({len(masks_patch)} masks)")
    ax2.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_masks(ax, masks: List[dict], alpha: float = 0.4) -> None:
    rng = np.random.RandomState(42)
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    for mask_data in sorted_masks:
        seg = mask_data["segmentation"]
        color = rng.rand(3)
        overlay = np.zeros((*seg.shape, 4))
        overlay[seg] = [*color, alpha]
        ax.imshow(overlay)


def plot_iou_distribution(
    all_ious: List[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_ious, bins=50, histtype="step", linewidth=1.5, color="#1f77b4")
    ax.axvline(np.mean(all_ious), color="red", linestyle="--", label=f"Mean: {np.mean(all_ious):.3f}")
    ax.axvline(np.median(all_ious), color="orange", linestyle="--", label=f"Median: {np.median(all_ious):.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("Matched Mask IoU Distribution (Original vs Patched)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_group_variance_comparison(
    profile_dir: Path,
    group_names: List[str],
    out_dir: Path,
    bins: int = 200,
) -> None:
    """Cross-group overlaid step histograms per block and per head (ViT-style)."""
    # --- Per-block comparison ---
    block_dir = out_dir / "comparison" / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        data = {}
        for g in group_names:
            var_path = profile_dir / g / f"block{b}_head0_variance.npy"
            if not var_path.exists():
                continue
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
        p99 = float(np.percentile(all_vals, 99))
        if p99 <= 0:
            p99 = float(np.max(all_vals)) + 1e-8
        block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"

        fig, ax = plt.subplots(figsize=(7, 5))
        for g, arr in data.items():
            ax.hist(arr, bins=bins, range=(0, p99), histtype="step", label=g, linewidth=1.5)
        ax.set_title(f"Block Variance Comparison — block {b} [{block_type}]")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(block_dir / f"compare_block{b}.png", dpi=150)
        plt.close(fig)

    # --- Per-head comparison ---
    head_dir = out_dir / "comparison" / "heads"
    head_dir.mkdir(parents=True, exist_ok=True)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            data = {}
            for g in group_names:
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    arr = np.load(vp).ravel()
                    if arr.size > 0:
                        data[g] = arr

            if not data:
                continue

            all_vals = np.concatenate(list(data.values()))
            p99 = float(np.percentile(all_vals, 99))
            if p99 <= 0:
                p99 = float(np.max(all_vals)) + 1e-8

            fig, ax = plt.subplots(figsize=(7, 5))
            for g, arr in data.items():
                ax.hist(arr, bins=bins, range=(0, p99), histtype="step", label=g, linewidth=1.5)
            ax.set_title(f"Head Variance Comparison — block {b} head {h}")
            ax.set_xlabel("Variance")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(head_dir / f"compare_block{b}_head{h}.png", dpi=150)
            plt.close(fig)

    print(f"  [comparison] Saved cross-group plots to {out_dir / 'comparison'}")


def plot_iou_vs_flop_reduction(
    sweep_results: List[dict],
    out_dir: Path,
) -> None:
    """IoU vs FLOP reduction curves per domain group (accuracy-vs-FLOPs style).

    sweep_results: list of dicts, each with keys:
        group, frac, flop_reduction, mean_iou, miss_rate
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(set(r["group"] for r in sweep_results))
    group_colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 0.7, len(groups)))))

    # --- Per-group plots ---
    for group in groups:
        entries = sorted(
            [r for r in sweep_results if r["group"] == group],
            key=lambda r: r["flop_reduction"],
        )
        if not entries:
            continue

        x = [e["flop_reduction"] for e in entries]
        y_iou = [e["mean_iou"] * 100 for e in entries]
        y_miss = [e["miss_rate"] * 100 for e in entries]
        fracs = [f"{e['frac']:.3f}" for e in entries]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y_iou, "o-", color="tab:blue", label="Mean IoU", markersize=7)
        ax.plot(x, y_miss, "o--", color="tab:red", label="Miss Rate", markersize=7)

        for xi, yi, fr in zip(x, y_iou, fracs):
            ax.annotate(fr, (xi, yi), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", color="tab:blue")

        ax.set_xlabel("Attention FLOP Reduction (%)", fontsize=12)
        ax.set_ylabel("Metric (%)", fontsize=12)
        ax.set_title(f"S-VELCRO SAM ViT-H — {group.capitalize()}: IoU & Miss Rate vs FLOP Reduction",
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{group}_iou_vs_flops.png", dpi=150)
        plt.close(fig)

    # --- Combined all-groups IoU plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in groups:
        entries = sorted(
            [r for r in sweep_results if r["group"] == group],
            key=lambda r: r["flop_reduction"],
        )
        if not entries:
            continue
        x = [e["flop_reduction"] for e in entries]
        y = [e["mean_iou"] * 100 for e in entries]
        ax.plot(x, y, "o-", color=group_colors[group], label=group.capitalize(), markersize=7)

    ax.set_xlabel("Attention FLOP Reduction (%)", fontsize=12)
    ax.set_ylabel("Mean IoU (%)", fontsize=12)
    ax.set_title("S-VELCRO SAM ViT-H — All Groups: Mean IoU vs FLOP Reduction", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "all_groups_iou.png", dpi=150)
    plt.close(fig)

    print(f"  [sweep] Saved IoU-vs-FLOPs plots to {out_dir}")


def plot_before_after_variance(
    profile_dir: Path,
    group_names: List[str],
    frozen_entries: List[dict],
    out_dir: Path,
    bins: int = 200,
) -> None:
    """Before/after block variance histograms with frozen-head annotation (ViT-style)."""
    frozen_set = {(e["block"], e["head"]) for e in frozen_entries}
    ba_dir = out_dir / "before_after"
    ba_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 0.3, max(len(group_names), 1)))

    for b in range(NUM_BLOCKS):
        block_type = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"
        n_frozen = sum(1 for h in range(NUM_HEADS) if (b, h) in frozen_set)

        arrs_per_group = {}
        for g in group_names:
            head_arrs = []
            for h in range(NUM_HEADS):
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    head_arrs.append(np.load(vp).ravel())
            if head_arrs:
                arrs_per_group[g] = np.concatenate(head_arrs)

        if not arrs_per_group:
            continue

        all_vals = np.concatenate(list(arrs_per_group.values()))
        p99 = float(np.percentile(all_vals, 99))
        if p99 <= 0:
            p99 = float(np.max(all_vals)) + 1e-8

        fig, (ax_all, ax_frozen, ax_active) = plt.subplots(3, 1, figsize=(8, 10))
        fig.suptitle(f"Block {b} [{block_type}] — {n_frozen}/{NUM_HEADS} heads frozen",
                     fontsize=13, fontweight="bold")

        for gi, g in enumerate(group_names):
            if g not in arrs_per_group:
                continue
            ax_all.hist(arrs_per_group[g], bins=bins, range=(0, p99),
                        histtype="step", label=g, color=colors[gi], linewidth=1.5)
        ax_all.set_ylabel("All Heads\nFrequency")
        ax_all.grid(True, alpha=0.3)
        ax_all.legend(fontsize=8)

        for gi, g in enumerate(group_names):
            frozen_arrs = []
            for h in range(NUM_HEADS):
                if (b, h) not in frozen_set:
                    continue
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    frozen_arrs.append(np.load(vp).ravel())
            if frozen_arrs:
                ax_frozen.hist(np.concatenate(frozen_arrs), bins=bins, range=(0, p99),
                               histtype="step", label=g, color=colors[gi], linewidth=1.5)
        ax_frozen.set_ylabel("Frozen Heads\nFrequency")
        ax_frozen.grid(True, alpha=0.3)
        ax_frozen.legend(fontsize=8)

        for gi, g in enumerate(group_names):
            active_arrs = []
            for h in range(NUM_HEADS):
                if (b, h) in frozen_set:
                    continue
                vp = profile_dir / g / f"block{b}_head{h}_variance.npy"
                if vp.exists():
                    active_arrs.append(np.load(vp).ravel())
            if active_arrs:
                ax_active.hist(np.concatenate(active_arrs), bins=bins, range=(0, p99),
                                histtype="step", label=g, color=colors[gi], linewidth=1.5)
        ax_active.set_ylabel("Active Heads\nFrequency")
        ax_active.set_xlabel("Variance")
        ax_active.grid(True, alpha=0.3)
        ax_active.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(ba_dir / f"before_after_block{b}.png", dpi=150)
        plt.close(fig)

    print(f"  [before/after] Saved plots to {ba_dir}")


def plot_domain_specialization_matrix(
    results: Dict[Tuple[str, str], float],
    groups: List[str],
    out_path: Path,
) -> None:
    """Heatmap: rows=profiled-on, cols=evaluated-on, values=mean IoU."""
    n = len(groups)
    grid = np.full((n, n), np.nan)

    for i, prof in enumerate(groups):
        for j, evl in enumerate(groups):
            key = (prof, evl)
            if key in results:
                grid[i, j] = results[key]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels([g.capitalize() for g in groups], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([g.capitalize() for g in groups])
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Profiled On")
    ax.set_title("Domain Specialization Matrix — Mean IoU")

    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                color = "white" if grid[i, j] < 0.75 else "black"
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean IoU")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_windowed_vs_global_rank(
    profile_dir: Path,
    group_name: str,
    out_path: Path,
) -> None:
    """Compare rank score distributions between windowed and global blocks."""
    windowed_ranks = []
    global_ranks = []

    group_dir = profile_dir / group_name
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            vp = group_dir / f"block{b}_head{h}_variance.npy"
            if not vp.exists():
                continue
            rank = float(np.sum(np.load(vp)))
            if b in GLOBAL_ATTN_INDEXES:
                global_ranks.append(rank)
            else:
                windowed_ranks.append(rank)

    if not windowed_ranks or not global_ranks:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_ranks = windowed_ranks + global_ranks
    p99 = float(np.percentile(all_ranks, 99))

    ax.hist(windowed_ranks, bins=50, range=(0, p99), histtype="step",
            label=f"Windowed (n={len(windowed_ranks)})", linewidth=1.5, color="tab:blue")
    ax.hist(global_ranks, bins=50, range=(0, p99), histtype="step",
            label=f"Global (n={len(global_ranks)})", linewidth=1.5, color="tab:red")

    ax.axvline(np.median(windowed_ranks), color="tab:blue", linestyle="--", alpha=0.6,
               label=f"Windowed median: {np.median(windowed_ranks):.1f}")
    ax.axvline(np.median(global_ranks), color="tab:red", linestyle="--", alpha=0.6,
               label=f"Global median: {np.median(global_ranks):.1f}")

    ax.set_xlabel("Rank Score (Σ variance)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Windowed vs Global Head Rank Scores — {group_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_heatmap(
    profile_dir: Path,
    group_name: str,
    out_path: Path,
) -> None:
    group_dir = profile_dir / group_name
    grid = np.full((NUM_BLOCKS, NUM_HEADS), np.nan)

    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            var_path = group_dir / f"block{b}_head{h}_variance.npy"
            if var_path.exists():
                var_arr = np.load(var_path)
                grid[b, h] = np.log10(np.sum(var_arr) + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    for b in GLOBAL_ATTN_INDEXES:
        ax.axhline(y=b, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Block Index")
    ax.set_title(f"log₁₀(Rank Score) — {group_name}")
    ax.set_xticks(range(NUM_HEADS))
    ax.set_yticks(range(0, NUM_BLOCKS, 2))
    fig.colorbar(im, ax=ax, shrink=0.5, label="log₁₀(Σ variance)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
