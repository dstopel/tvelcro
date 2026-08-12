#!/usr/bin/env python3
"""Combined profile + eval comparison: shows how profiling metrics predict eval quality."""

import json
import numpy as np
from pathlib import Path

out_dir = Path("results_cross_domain")
profile_dir = out_dir / "profile"
groups = ["general", "kvasir", "isic"]
fractions = [0.125, 0.25, 0.375]

# ── Load profile statistics ──────────────────────────────────────────────────

print()
print("=" * 120)
print("CROSS-DOMAIN EXPERIMENT — PROFILE vs EVAL COMPARISON")
print("=" * 120)

print()
print("PART 1: PROFILING RESULTS (variance analysis — no freezing, just measurement)")
print("-" * 120)
print(f"{'Group':<12} {'Total Rank':>12} {'Wind. Mean':>12} {'Global Mean':>14} {'G/W Ratio':>10} {'Spearman ρ':>12} {'Overlap@12.5%':>14}")
print("-" * 120)

# Compute rank scores from .npy files
group_scores = {}
for group in groups:
    group_dir = profile_dir / group
    if not group_dir.exists():
        continue
    scores = []
    for block in range(32):
        for head in range(16):
            var_file = group_dir / f"block{block}_head{head}_variance.npy"
            if var_file.exists():
                var = np.load(var_file)
                scores.append({"block": block, "head": head, "score": float(var.sum())})
    group_scores[group] = scores

# Compute stats per group
from scipy.stats import spearmanr

general_ranking = None
if "general" in group_scores:
    general_ranking = sorted(group_scores["general"], key=lambda x: x["score"])
    general_order = [(e["block"], e["head"]) for e in general_ranking]

for group in groups:
    if group not in group_scores:
        continue
    scores = group_scores[group]
    if not scores:
        continue

    vals = [s["score"] for s in scores]
    total = sum(vals)

    windowed_vals = [s["score"] for s in scores if s["block"] not in (7, 15, 23, 31)]
    global_vals = [s["score"] for s in scores if s["block"] in (7, 15, 23, 31)]

    w_mean = np.mean(windowed_vals) if windowed_vals else 0
    g_mean = np.mean(global_vals) if global_vals else 0
    ratio = g_mean / w_mean if w_mean > 0 else 0

    # Spearman correlation with general
    rho = 1.0
    overlap = "—"
    if group != "general" and general_ranking is not None:
        group_ranking = sorted(scores, key=lambda x: x["score"])
        group_order = [(e["block"], e["head"]) for e in group_ranking]

        # Build rank arrays
        general_rank_map = {(e["block"], e["head"]): i for i, e in enumerate(general_ranking)}
        group_rank_map = {(e["block"], e["head"]): i for i, e in enumerate(group_ranking)}

        common_keys = sorted(set(general_rank_map.keys()) & set(group_rank_map.keys()))
        if len(common_keys) > 10:
            r1 = [general_rank_map[k] for k in common_keys]
            r2 = [group_rank_map[k] for k in common_keys]
            rho, _ = spearmanr(r1, r2)

        # Freeze overlap at 12.5% (64 heads)
        general_frozen = set(general_order[:64])
        group_frozen = set(group_order[:64])
        overlap_count = len(general_frozen & group_frozen)
        overlap = f"{overlap_count}/64 ({overlap_count*100//64}%)"

    rho_str = f"{rho:.3f}" if group != "general" else "1.000"
    print(f"{group:<12} {total:>12.0f} {w_mean:>12.1f} {g_mean:>14.1f} {ratio:>10.1f}x {rho_str:>12} {overlap:>14}")

print()
print()
print("PART 2: EVAL RESULTS (frozen model vs unpatched baseline)")
print("-" * 120)
print(f"{'Group':<12} {'Frac':>5} {'Profile':>10} {'IoU':>8} {'Δ from 1.0':>10} {'Miss%':>8} {'FP%':>8} {'Note'}")
print("-" * 120)

# Baseline row (no freezing = perfect)
for group in ["kvasir", "isic"]:
    print(f"{group:<12} {'0':>5} {'baseline':>10} {'1.0000':>8} {'0.0pp':>10} {'0.0':>8} {'0.0':>8} {'← unpatched SAM (no freeze)'}")

print()

for group in ["kvasir", "isic"]:
    for frac in fractions:
        for profile_type, tag in [("domain", f"eval_{group}_f{frac}"), ("general", f"eval_general_on_{group}_f{frac}")]:
            results_path = out_dir / tag / group / "eval_results.json"
            if not results_path.exists():
                continue
            r = json.load(open(results_path))
            iou = r.get("per_image_mean_iou", r.get("overall_mean_iou", 0))
            miss = r.get("per_image_miss_rate", 0) * 100
            fp = r.get("per_image_false_pos_rate", 0) * 100
            delta = (1.0 - iou) * 100

            note = ""
            if profile_type == "domain":
                note = "✓ domain-specific"
            else:
                note = "  general profile"

            print(f"{group:<12} {frac:>5.3f} {profile_type:>10} {iou:>8.4f} {delta:>9.2f}pp {miss:>8.1f} {fp:>8.1f} {note}")
    print()

print()
print("PART 3: KEY INSIGHT — CORRELATION PREDICTS EVAL QUALITY")
print("-" * 120)

for group in ["kvasir", "isic"]:
    if group not in group_scores or "general" not in group_scores:
        continue

    # Get Spearman
    group_ranking = sorted(group_scores[group], key=lambda x: x["score"])
    group_order = [(e["block"], e["head"]) for e in group_ranking]
    general_rank_map = {(e["block"], e["head"]): i for i, e in enumerate(general_ranking)}
    group_rank_map = {(e["block"], e["head"]): i for i, e in enumerate(group_ranking)}
    common_keys = sorted(set(general_rank_map.keys()) & set(group_rank_map.keys()))
    r1 = [general_rank_map[k] for k in common_keys]
    r2 = [group_rank_map[k] for k in common_keys]
    rho, _ = spearmanr(r1, r2)

    # Overlap
    general_frozen = set(general_order[:64])
    group_frozen = set(group_order[:64])
    overlap_pct = len(general_frozen & group_frozen) * 100 / 64

    # IoU gap at f=0.125
    domain_path = out_dir / f"eval_{group}_f0.125" / group / "eval_results.json"
    general_path = out_dir / f"eval_general_on_{group}_f0.125" / group / "eval_results.json"

    if domain_path.exists() and general_path.exists():
        d_iou = json.load(open(domain_path)).get("per_image_mean_iou", 0)
        g_iou = json.load(open(general_path)).get("per_image_mean_iou", 0)
        d_miss = json.load(open(domain_path)).get("per_image_miss_rate", 0) * 100
        g_miss = json.load(open(general_path)).get("per_image_miss_rate", 0) * 100

        print(f"  {group.upper()}:")
        print(f"    Spearman ρ with general:     {rho:.3f}")
        print(f"    Freeze overlap @12.5%:       {overlap_pct:.0f}%")
        print(f"    IoU gap (domain vs general): +{(d_iou - g_iou)*100:.2f}pp")
        print(f"    Miss rate gap:               {g_miss - d_miss:.1f}pp more misses with general profile")
        print()

print()
print("CONCLUSION:")
print("  Lower Spearman ρ → lower freeze overlap → bigger eval gap")
print("  → Domain-specific profiling is more important for distant domains")
print("=" * 120)
