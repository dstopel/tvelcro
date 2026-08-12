#!/usr/bin/env python3
"""
analyze_profile.py — Extract compact statistics from SAM VELCRO profile data.

Outputs a text report you can copy-paste for analysis:
  - Per-group summary (total variance, mean rank score)
  - Per-block breakdown (windowed vs global, variance stats)
  - Per-head rank table (sorted, with freeze candidates highlighted)
  - Cross-group rank correlation (do groups agree on which heads are low-variance?)
  - Top/bottom heads across all groups
"""

import json
from pathlib import Path
import argparse

import numpy as np

NUM_BLOCKS = 32
NUM_HEADS = 16
GLOBAL_ATTN_INDEXES = (7, 15, 23, 31)


def load_rank_scores(profile_dir: Path, group: str) -> dict:
    """Load rank_score = sum(variance) for each (block, head)."""
    group_dir = profile_dir / group
    scores = {}
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            vp = group_dir / f"block{b}_head{h}_variance.npy"
            if vp.exists():
                arr = np.load(vp)
                scores[(b, h)] = float(np.sum(arr))
    return scores


def variance_stats(profile_dir: Path, group: str) -> dict:
    """Per-head detailed stats: mean, std, p50, p99 of the variance array."""
    group_dir = profile_dir / group
    stats = {}
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            vp = group_dir / f"block{b}_head{h}_variance.npy"
            if vp.exists():
                arr = np.load(vp).ravel()
                stats[(b, h)] = {
                    "rank_score": float(np.sum(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p50": float(np.median(arr)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(np.max(arr)),
                    "shape": list(np.load(vp).shape),
                }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze SAM VELCRO profile data")
    parser.add_argument("--profile_dir", required=True,
                        help="Path to profile directory (e.g. results_sam_velcro/profile)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated groups (default: auto-detect)")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: print to stdout)")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)

    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]
    else:
        groups = sorted([
            d.name for d in profile_dir.iterdir()
            if d.is_dir() and (d / "block0_head0_variance.npy").exists()
        ])

    if not groups:
        print("ERROR: No profiled groups found.")
        return

    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 80)
    out("SAM ViT-H S-VELCRO — Profile Analysis Report")
    out("=" * 80)
    out(f"Profile dir: {profile_dir}")
    out(f"Groups: {groups}")
    out(f"Architecture: 32 blocks × 16 heads × 80 head_dim")
    out(f"Global blocks: {list(GLOBAL_ATTN_INDEXES)}")
    out()

    # ── Per-group summary ────────────────────────────────────────────────────
    all_scores = {}
    for g in groups:
        scores = load_rank_scores(profile_dir, g)
        all_scores[g] = scores
        vals = list(scores.values())
        windowed = [v for (b, h), v in scores.items() if b not in GLOBAL_ATTN_INDEXES]
        global_s = [v for (b, h), v in scores.items() if b in GLOBAL_ATTN_INDEXES]

        out(f"{'─' * 80}")
        out(f"GROUP: {g}  ({len(scores)} heads profiled)")
        out(f"{'─' * 80}")
        out(f"  Total rank scores:    {np.sum(vals):.1f}")
        out(f"  Mean rank score:      {np.mean(vals):.2f}")
        out(f"  Std rank score:       {np.std(vals):.2f}")
        out(f"  Min / Max:            {np.min(vals):.2f} / {np.max(vals):.2f}")
        out(f"  Windowed mean:        {np.mean(windowed):.2f}  (n={len(windowed)})")
        out(f"  Global mean:          {np.mean(global_s):.2f}  (n={len(global_s)})")
        out(f"  Global/Windowed ratio: {np.mean(global_s)/np.mean(windowed):.2f}x")
        out()

    # ── Per-block rank table ─────────────────────────────────────────────────
    out()
    out("=" * 80)
    out("PER-BLOCK MEAN RANK SCORES")
    out("=" * 80)
    header = f"{'Block':<7} {'Type':<9} " + " ".join(f"{g:<12}" for g in groups)
    out(header)
    out("-" * len(header))

    for b in range(NUM_BLOCKS):
        btype = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"
        row = f"{b:<7} {btype:<9} "
        for g in groups:
            block_scores = [all_scores[g].get((b, h), 0) for h in range(NUM_HEADS)]
            row += f"{np.mean(block_scores):<12.2f} "
        out(row)
    out()

    # ── Top-10 lowest and highest heads per group ────────────────────────────
    for g in groups:
        scores = all_scores[g]
        sorted_heads = sorted(scores.items(), key=lambda x: x[1])

        out(f"{'─' * 80}")
        out(f"GROUP: {g} — LOWEST 20 heads (best freeze candidates)")
        out(f"{'─' * 80}")
        out(f"  {'Rank':<6} {'Block':<7} {'Head':<6} {'Type':<9} {'Score':<12}")
        for i, ((b, h), score) in enumerate(sorted_heads[:20]):
            btype = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"
            out(f"  {i+1:<6} {b:<7} {h:<6} {btype:<9} {score:<12.2f}")
        out()

        out(f"  HIGHEST 10 heads (most active, never freeze):")
        out(f"  {'Rank':<6} {'Block':<7} {'Head':<6} {'Type':<9} {'Score':<12}")
        for i, ((b, h), score) in enumerate(reversed(sorted_heads[-10:])):
            btype = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "windowed"
            out(f"  {len(scores)-9+i:<6} {b:<7} {h:<6} {btype:<9} {score:<12.2f}")
        out()

    # ── Cross-group rank correlation ─────────────────────────────────────────
    if len(groups) >= 2:
        out("=" * 80)
        out("CROSS-GROUP RANK CORRELATION (Spearman)")
        out("=" * 80)
        out(f"  Do different domains agree on which heads are low-variance?")
        out()

        from scipy.stats import spearmanr

        all_keys = sorted(set.intersection(*[set(all_scores[g].keys()) for g in groups]))
        header = f"{'':>14}" + "".join(f"{g:>12}" for g in groups)
        out(header)
        for g1 in groups:
            row = f"{g1:>14}"
            v1 = [all_scores[g1][k] for k in all_keys]
            for g2 in groups:
                v2 = [all_scores[g2][k] for k in all_keys]
                rho, _ = spearmanr(v1, v2)
                row += f"{rho:>12.3f}"
            out(row)
        out()
        out("  (1.0 = identical ranking, 0.0 = no agreement)")
        out()

    # ── Freeze simulation at standard fractions ──────────────────────────────
    out("=" * 80)
    out("FREEZE SIMULATION — Heads frozen per fraction")
    out("=" * 80)

    for g in groups:
        scores = all_scores[g]
        out(f"\n  Group: {g}")
        out(f"  {'Frac':<8} {'Frozen':<8} {'Wind.':<8} {'Glob.':<8} {'Overlap w/ general':<20}")

        for frac in [0.125, 0.25, 0.375, 0.50]:
            t_k = max(0, min(int(frac * NUM_HEADS), NUM_HEADS - 1))
            frozen = set()
            for b in range(NUM_BLOCKS):
                block_heads = sorted(
                    [(h, scores.get((b, h), 0)) for h in range(NUM_HEADS)],
                    key=lambda x: x[1]
                )
                for h, _ in block_heads[:t_k]:
                    frozen.add((b, h))

            n_wind = sum(1 for (b, h) in frozen if b not in GLOBAL_ATTN_INDEXES)
            n_glob = sum(1 for (b, h) in frozen if b in GLOBAL_ATTN_INDEXES)

            overlap = ""
            if "general" in all_scores and g != "general":
                gen_scores = all_scores["general"]
                gen_frozen = set()
                for b in range(NUM_BLOCKS):
                    block_heads = sorted(
                        [(h, gen_scores.get((b, h), 0)) for h in range(NUM_HEADS)],
                        key=lambda x: x[1]
                    )
                    for h, _ in block_heads[:t_k]:
                        gen_frozen.add((b, h))
                common = len(frozen & gen_frozen)
                overlap = f"{common}/{len(frozen)} ({100*common/len(frozen):.0f}%)"

            out(f"  {frac:<8.3f} {len(frozen):<8} {n_wind:<8} {n_glob:<8} {overlap:<20}")
    out()

    # ── Per-block head ranking (compact) ─────────────────────────────────────
    out("=" * 80)
    out("COMPACT HEAD RANKINGS (lowest→highest per block, first group)")
    out("=" * 80)
    g0 = groups[0]
    out(f"  Group: {g0}")
    out(f"  {'Block':<7} {'Type':<9} Heads sorted by rank score (lowest first)")
    out(f"  {'':<16} {'─' * 60}")

    for b in range(NUM_BLOCKS):
        btype = "GLOBAL" if b in GLOBAL_ATTN_INDEXES else "wind."
        block_heads = sorted(
            [(h, all_scores[g0].get((b, h), 0)) for h in range(NUM_HEADS)],
            key=lambda x: x[1]
        )
        head_str = " ".join(f"h{h}:{s:.0f}" for h, s in block_heads)
        out(f"  {b:<7} {btype:<9} {head_str}")
    out()

    # ── Output ───────────────────────────────────────────────────────────────
    report = "\n".join(lines)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
