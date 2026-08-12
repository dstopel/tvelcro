#!/usr/bin/env python3
"""Summarize all SAM VELCRO results across both experiment waves (domain-specific only)."""

import json
from pathlib import Path

RESULTS_DIRS = [
    Path("results_cross_domain"),  # Wave 1: kvasir, isic
    Path("results_wave2"),         # Wave 2: busi, pcb, deepglobe
]

FRACTIONS = [0.125, 0.25, 0.375]

print()
print("=" * 100)
print("SAM ViT-H S-VELCRO — ALL DOMAINS SUMMARY (domain-specific profiling only)")
print("=" * 100)
print()
print(f"{'Domain':<15} {'Frac':>5} {'Mean IoU':>10} {'Δ from 1.0':>10} {'Miss%':>8} {'FP%':>8} {'Skipped':>8} {'Frozen':>8} {'Attn%':>8}")
print("-" * 100)

all_rows = []

for results_dir in RESULTS_DIRS:
    if not results_dir.exists():
        continue

    # Auto-detect groups: look for eval_<group>_f0.125 directories
    groups = set()
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith("eval_") and "_f0.125" in d.name:
            name = d.name.replace("eval_", "").replace("_f0.125", "")
            if "general_on" not in name:
                groups.add(name)

    for group in sorted(groups):
        for frac in FRACTIONS:
            eval_tag = f"eval_{group}_f{frac}"
            results_path = results_dir / eval_tag / group / "eval_results.json"
            freeze_tag = f"freeze_{group}_f{frac}"
            manifest_path = results_dir / freeze_tag / "freeze_manifest.json"

            if not results_path.exists():
                continue

            r = json.load(open(results_path))
            iou = r.get("per_image_mean_iou", r.get("overall_mean_iou", 0))
            miss = r.get("per_image_miss_rate", 0) * 100
            fp = r.get("per_image_false_pos_rate", 0) * 100
            skipped = r.get("skipped_images", 0)
            delta = (1.0 - iou) * 100

            frozen = "?"
            attn_pct = "?"
            if manifest_path.exists():
                m = json.load(open(manifest_path))
                frozen = m.get("summary", {}).get("frozen", "?")
                attn_pct = m.get("flops", {}).get("attention_flop_reduction_pct", 0)

            print(f"{group:<15} {frac:>5.3f} {iou:>10.4f} {delta:>9.2f}pp {miss:>8.1f} {fp:>8.1f} {skipped:>8} {frozen:>8} {attn_pct:>8}")
            all_rows.append({
                "domain": group, "fraction": frac,
                "mean_iou": round(iou, 4), "delta_from_baseline_pp": round(delta, 2),
                "miss_rate_pct": round(miss, 1), "fp_rate_pct": round(fp, 1),
                "skipped": skipped, "frozen_heads": frozen,
                "attn_flop_reduction_pct": attn_pct,
            })
        print()

print("=" * 100)
print()

# ── Best operating point per domain ──────────────────────────────────────────

print("BEST OPERATING POINT PER DOMAIN (highest IoU at f=0.125):")
print("-" * 100)
print(f"{'Domain':<15} {'IoU':>8} {'Δ':>8} {'Miss%':>8} {'FP%':>8} {'Attn FLOPs saved':>18}")
print("-" * 100)

seen = set()
for row in all_rows:
    if row["fraction"] == 0.125 and row["domain"] not in seen:
        seen.add(row["domain"])
        print(f"{row['domain']:<15} {row['mean_iou']:>8.4f} {row['delta_from_baseline_pp']:>7.2f}pp {row['miss_rate_pct']:>8.1f} {row['fp_rate_pct']:>8.1f} {row['attn_flop_reduction_pct']:>17}%")

print()
print("=" * 100)
print()

# ── Degradation rate ─────────────────────────────────────────────────────────

print("DEGRADATION RATE (IoU loss per 10% attn FLOP reduction):")
print("-" * 100)

domains_seen = {}
for row in all_rows:
    d = row["domain"]
    if d not in domains_seen:
        domains_seen[d] = []
    domains_seen[d].append(row)

for domain, rows in domains_seen.items():
    if len(rows) >= 2:
        first = rows[0]
        last = rows[-1]
        flop_range = last["attn_flop_reduction_pct"] - first["attn_flop_reduction_pct"]
        iou_drop = last["delta_from_baseline_pp"] - first["delta_from_baseline_pp"]
        if flop_range > 0:
            rate = iou_drop / (flop_range / 10)
            print(f"  {domain:<15} {rate:.2f}pp IoU loss per 10% attn FLOPs removed")

print()
print("=" * 100)

# ── Save JSON ────────────────────────────────────────────────────────────────

out_path = Path("all_domains_summary.json")
json.dump(all_rows, open(out_path, "w"), indent=2)
print(f"\nJSON saved: {out_path}")
