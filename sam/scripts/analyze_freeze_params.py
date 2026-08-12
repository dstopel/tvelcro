#!/usr/bin/env python3
"""
analyze_freeze_params.py — Determine optimal freezing parameters per domain.

Analyzes profiled variance data to recommend freeze parameters for each domain
that balance compute savings against expected quality degradation.

For each domain this script:
  1. Computes per-head variance reduction ratios (domain vs general)
  2. Analyzes the distribution of reductions to find natural breakpoints
  3. Recommends parameters for both freeze methods:
     - rank method: freeze_fraction that targets a quality-preserving subset
     - ratio method: ratio_threshold at the distribution knee
  4. Estimates the number of heads frozen and attention FLOP savings

Usage:
    python scripts/analyze_freeze_params.py --profile_dir ./results_cross_domain/profile

    # With wave2 profiles in a separate directory:
    python scripts/analyze_freeze_params.py \
        --profile_dir ./results_cross_domain/profile \
        --extra_profile_dir ./results_wave2/profile

    # Custom IoU target (default 0.975 = ~2.5pp loss budget):
    python scripts/analyze_freeze_params.py \
        --profile_dir ./results_cross_domain/profile \
        --iou_target 0.98
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam_velcro_pkg.constants import GLOBAL_ATTN_INDEXES, NUM_BLOCKS, NUM_HEADS
from sam_velcro_pkg.flops import compute_flop_reduction


DOMAIN_LABELS = {
    "general": "General (COCO)",
    "kvasir": "Kvasir (Endoscopy)",
    "isic": "ISIC (Dermoscopy)",
    "busi": "BUSI (Ultrasound)",
    "pcb": "PCB (Circuit Board)",
    "deepglobe": "DeepGlobe (Satellite)",
}

TOTAL_HEADS = NUM_BLOCKS * NUM_HEADS


def load_head_variances(profile_dir: Path, group: str):
    """Load rank scores (sum of variance) for all heads in a group."""
    group_dir = profile_dir / group
    if not group_dir.exists():
        return None

    scores = {}
    for b in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            var_path = group_dir / f"block{b}_head{h}_variance.npy"
            if var_path.exists():
                var_arr = np.load(var_path)
                scores[(b, h)] = float(np.sum(var_arr))
    return scores if scores else None


def find_profile_dir(group: str, profile_dirs: list):
    """Find which profile directory contains a given group."""
    for pdir in profile_dirs:
        gdir = pdir / group
        if gdir.exists() and len(list(gdir.glob("*.npy"))) >= 512:
            return pdir
    return None


def compute_reductions(domain_scores, general_scores):
    """Compute variance reduction ratio for each head: 1 - (domain / general)."""
    reductions = []
    for (b, h), dom_score in domain_scores.items():
        if (b, h) not in general_scores:
            continue
        gen_score = general_scores[(b, h)]
        if gen_score < 1e-10:
            continue
        reduction = 1.0 - (dom_score / gen_score)
        block_type = "global" if b in GLOBAL_ATTN_INDEXES else "windowed"
        reductions.append({
            "block": b,
            "head": h,
            "block_type": block_type,
            "domain_score": dom_score,
            "general_score": gen_score,
            "reduction": reduction,
        })
    reductions.sort(key=lambda x: x["reduction"], reverse=True)
    return reductions


def find_knee_threshold(reductions):
    """Find the knee point in the sorted reduction curve.

    Uses the maximum distance from the line connecting the first and last points
    of the sorted reduction values (L-method).
    """
    values = np.array([r["reduction"] for r in reductions])
    n = len(values)
    if n < 3:
        return 0.3

    x = np.arange(n, dtype=float)
    x_norm = x / (n - 1)
    y_norm = (values - values[-1]) / (values[0] - values[-1] + 1e-10)

    distances = np.abs(y_norm - (1.0 - x_norm))
    knee_idx = int(np.argmax(distances))

    return float(values[knee_idx])


def find_threshold_for_target_count(reductions, target_count):
    """Find the ratio threshold that freezes approximately target_count heads."""
    if target_count <= 0:
        return 0.99
    if target_count >= len(reductions):
        return -1.0

    sorted_r = sorted(reductions, key=lambda x: x["reduction"], reverse=True)
    if target_count < len(sorted_r):
        return sorted_r[target_count - 1]["reduction"]
    return sorted_r[-1]["reduction"]



def find_count_for_attn_target(reductions, target_attn_pct):
    """Find the minimum number of heads (sorted by reduction) to reach a target attention FLOP %."""
    from sam_velcro_pkg.flops import TOTAL_ATTN_FLOPS, PER_HEAD_SAVINGS_WINDOWED, PER_HEAD_SAVINGS_GLOBAL

    target_saved = target_attn_pct / 100.0 * TOTAL_ATTN_FLOPS
    sorted_r = sorted(reductions, key=lambda x: x["reduction"], reverse=True)

    cumulative = 0.0
    for i, r in enumerate(sorted_r):
        if r["block_type"] == "global":
            cumulative += PER_HEAD_SAVINGS_GLOBAL
        else:
            cumulative += PER_HEAD_SAVINGS_WINDOWED
        if cumulative >= target_saved:
            return i + 1

    return len(sorted_r)


def build_frozen_entry_list(reductions, count):
    """Build a list of frozen head entries for FLOP calculation."""
    sorted_r = sorted(reductions, key=lambda x: x["reduction"], reverse=True)
    entries = []
    for r in sorted_r[:count]:
        entries.append({"block": r["block"], "block_type": r["block_type"]})
    return entries


def analyze_domain(domain, domain_scores, general_scores, iou_target):
    """Full analysis for one domain. Returns a recommendation dict."""
    reductions = compute_reductions(domain_scores, general_scores)
    if not reductions:
        return None

    values = np.array([r["reduction"] for r in reductions])
    positive_reductions = values[values > 0]

    knee = find_knee_threshold(reductions)

    attn_targets = {"conservative": 15.0, "recommended": 25.0, "aggressive": 37.5}

    operating_points = {}
    for label, target_attn_pct in attn_targets.items():
        count = find_count_for_attn_target(reductions, target_attn_pct)
        entries = build_frozen_entry_list(reductions, count)
        flops = compute_flop_reduction(entries)
        threshold = find_threshold_for_target_count(reductions, count)
        frac = count / TOTAL_HEADS

        n_windowed = sum(1 for e in entries if e["block_type"] == "windowed")
        n_global = count - n_windowed

        operating_points[label] = {
            "heads_frozen": count,
            "freeze_fraction": round(frac, 4),
            "ratio_threshold": round(threshold, 4),
            "attention_flop_reduction_pct": round(flops["attention_flop_reduction_pct"], 2),
            "saved_gflops": round(flops["saved_gflops"], 2),
            "windowed_frozen": n_windowed,
            "global_frozen": n_global,
            "min_reduction_in_set": round(float(threshold), 4),
        }

    stats = {
        "total_heads": TOTAL_HEADS,
        "heads_with_positive_reduction": int(np.sum(values > 0)),
        "heads_above_30pct": int(np.sum(values > 0.30)),
        "heads_above_50pct": int(np.sum(values > 0.50)),
        "heads_above_70pct": int(np.sum(values > 0.70)),
        "mean_reduction": round(float(np.mean(positive_reductions)) if len(positive_reductions) > 0 else 0, 4),
        "median_reduction": round(float(np.median(positive_reductions)) if len(positive_reductions) > 0 else 0, 4),
        "max_reduction": round(float(values[0]) if len(values) > 0 else 0, 4),
        "knee_threshold": round(knee, 4),
    }

    rank_scores = np.array([r["domain_score"] for r in reductions])
    sorted_rank = np.sort(rank_scores)

    rank_params = {}
    for target_frac in [0.125, 0.25, 0.375]:
        n = int(target_frac * NUM_HEADS)
        block_cutoffs = []
        for b in range(NUM_BLOCKS):
            block_heads = [(r["domain_score"], r["head"]) for r in reductions if r["block"] == b]
            block_heads.sort()
            if len(block_heads) >= n:
                cutoff = block_heads[n - 1][0]
            else:
                cutoff = block_heads[-1][0] if block_heads else 0
            block_cutoffs.append(cutoff)
        rank_params[str(target_frac)] = {
            "freeze_per_block": n,
            "total_frozen": n * NUM_BLOCKS,
            "mean_block_cutoff": round(float(np.mean(block_cutoffs)), 2),
            "max_block_cutoff": round(float(np.max(block_cutoffs)), 2),
        }

    return {
        "domain": domain,
        "label": DOMAIN_LABELS.get(domain, domain),
        "stats": stats,
        "operating_points": operating_points,
        "rank_method_fractions": rank_params,
        "top_10_heads": [
            {
                "block": r["block"],
                "head": r["head"],
                "type": r["block_type"],
                "reduction_pct": round(r["reduction"] * 100, 1),
                "domain_var": round(r["domain_score"], 1),
                "general_var": round(r["general_score"], 1),
            }
            for r in reductions[:10]
        ],
    }


def print_report(results, iou_target):
    """Print a human-readable report."""
    print()
    print("=" * 100)
    print("SAM ViT-H S-VELCRO — FREEZE PARAMETER ANALYSIS")
    print(f"IoU preservation target: {iou_target:.1%}")
    print("=" * 100)

    for res in results:
        domain = res["domain"]
        label = res["label"]
        stats = res["stats"]
        ops = res["operating_points"]

        print(f"\n{'─' * 100}")
        print(f"  {label}")
        print(f"{'─' * 100}")

        print(f"\n  Reduction distribution:")
        print(f"    Heads with domain var < general var:  {stats['heads_with_positive_reduction']} / {stats['total_heads']}")
        print(f"    Heads with >30% reduction:           {stats['heads_above_30pct']}")
        print(f"    Heads with >50% reduction:           {stats['heads_above_50pct']}")
        print(f"    Heads with >70% reduction:           {stats['heads_above_70pct']}")
        print(f"    Mean reduction (positive heads):     {stats['mean_reduction']:.1%}")
        print(f"    Max reduction:                       {stats['max_reduction']:.1%}")
        print(f"    Distribution knee:                   {stats['knee_threshold']:.1%}")

        print(f"\n  Recommended operating points:")
        print(f"    {'Level':<14} {'Heads':>6} {'Frac':>6} {'Ratio Thr':>10} {'Attn FLOPs%':>12} {'GFLOPs saved':>13}")
        print(f"    {'─' * 65}")
        for level in ["conservative", "recommended", "aggressive"]:
            op = ops[level]
            print(f"    {level:<14} {op['heads_frozen']:>6} {op['freeze_fraction']:>6.3f} "
                  f"{op['ratio_threshold']:>10.4f} {op['attention_flop_reduction_pct']:>11.2f}% "
                  f"{op['saved_gflops']:>12.2f}")

        print(f"\n  Top 5 most locality-exhibiting heads:")
        print(f"    {'Block':<7} {'Head':<6} {'Type':<10} {'Reduction':>10} {'Domain var':>12} {'General var':>12}")
        print(f"    {'─' * 60}")
        for h in res["top_10_heads"][:5]:
            print(f"    {h['block']:<7} {h['head']:<6} {h['type']:<10} "
                  f"{h['reduction_pct']:>9.1f}% {h['domain_var']:>12.1f} {h['general_var']:>12.1f}")

    print(f"\n{'=' * 100}")
    print("\n  RECOMMENDED COMMANDS")
    print(f"{'=' * 100}")

    for res in results:
        domain = res["domain"]
        op = res["operating_points"]["recommended"]
        print(f"\n  # {res['label']} — {op['heads_frozen']} heads, "
              f"{op['attention_flop_reduction_pct']:.1f}% attn FLOP savings")
        print(f"  # Ratio method (adaptive, domain-aware):")
        print(f"  python sam_velcro.py --mode freeze \\")
        print(f"      --freeze_method ratio \\")
        print(f"      --ratio_threshold {op['ratio_threshold']:.4f} \\")
        print(f"      --profile_dir <PROFILE_DIR> \\")
        print(f"      --profile_group {domain} \\")
        print(f"      --general_group general \\")
        print(f"      --out_dir results/freeze_{domain}_recommended")
        print()
        print(f"  # Rank method (fixed fraction, simpler):")
        print(f"  python sam_velcro.py --mode freeze \\")
        print(f"      --freeze_method rank \\")
        print(f"      --freeze_fraction {op['freeze_fraction']:.4f} \\")
        print(f"      --profile_dir <PROFILE_DIR> \\")
        print(f"      --profile_group {domain} \\")
        print(f"      --out_dir results/freeze_{domain}_recommended")

    print()
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze profile data to determine optimal freeze parameters per domain."
    )
    parser.add_argument("--profile_dir", type=str, required=True,
                        help="Primary profile directory (must contain general/ subdirectory)")
    parser.add_argument("--extra_profile_dir", type=str, default=None,
                        help="Additional profile directory (e.g., for wave2 domains)")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domain names (auto-detected if omitted)")
    parser.add_argument("--general_group", type=str, default="general",
                        help="Name of the general/baseline profile group")
    parser.add_argument("--iou_target", type=float, default=0.975,
                        help="Target IoU preservation level (default: 0.975 = 2.5pp loss budget)")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Output JSON path (default: <profile_dir>/freeze_recommendations.json)")
    args = parser.parse_args()

    profile_dirs = [Path(args.profile_dir)]
    if args.extra_profile_dir:
        profile_dirs.append(Path(args.extra_profile_dir))

    for pdir in profile_dirs:
        if not pdir.exists():
            print(f"ERROR: Profile directory does not exist: {pdir}")
            sys.exit(1)

    general_pdir = find_profile_dir(args.general_group, profile_dirs)
    if general_pdir is None:
        print(f"ERROR: General profile '{args.general_group}' not found in any profile directory")
        sys.exit(1)

    general_scores = load_head_variances(general_pdir, args.general_group)
    print(f"Loaded general profile: {len(general_scores)} heads from {general_pdir / args.general_group}")

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    else:
        domains = []
        for pdir in profile_dirs:
            for subdir in sorted(pdir.iterdir()):
                if subdir.is_dir() and subdir.name != args.general_group:
                    npy_count = len(list(subdir.glob("*_variance.npy")))
                    if npy_count >= 256:
                        domains.append(subdir.name)
        domains = sorted(set(domains))

    if not domains:
        print("ERROR: No domain profiles found. Specify --domains or check profile directories.")
        sys.exit(1)

    print(f"Domains to analyze: {domains}")

    results = []
    for domain in domains:
        pdir = find_profile_dir(domain, profile_dirs)
        if pdir is None:
            print(f"  WARNING: Profile for '{domain}' not found, skipping")
            continue

        domain_scores = load_head_variances(pdir, domain)
        if domain_scores is None:
            print(f"  WARNING: No variance data for '{domain}', skipping")
            continue

        print(f"  Analyzing {domain} ({len(domain_scores)} heads)...")
        result = analyze_domain(domain, domain_scores, general_scores, args.iou_target)
        if result:
            results.append(result)

    if not results:
        print("ERROR: No domains could be analyzed.")
        sys.exit(1)

    print_report(results, args.iou_target)

    out_json = Path(args.out_json) if args.out_json else Path(args.profile_dir) / "freeze_recommendations.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "iou_target": args.iou_target,
            "general_group": args.general_group,
            "domains": results,
        }, f, indent=2)
    print(f"\nRecommendations saved to: {out_json}")


if __name__ == "__main__":
    main()
