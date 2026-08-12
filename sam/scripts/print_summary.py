#!/usr/bin/env python3
"""Print cross-domain experiment summary from eval results."""

import json
from pathlib import Path

out_dir = Path("results_cross_domain")
groups = ["kvasir", "isic"]
fractions = [0.125, 0.25, 0.375]

print()
print("=" * 110)
print("CROSS-DOMAIN LOCALITY EXPERIMENT — SUMMARY")
print("=" * 110)
print()
print(f"{'Domain':<15} {'Frac':>5} {'Profile':>12} {'Mean IoU':>10} {'Miss%':>8} {'FP%':>8} {'Skipped':>8} {'Frozen':>8} {'Attn%':>8}")
print("-" * 110)

rows = []
for group in groups:
    for frac in fractions:
        for profile_type, tag in [("domain", f"eval_{group}_f{frac}"), ("general", f"eval_general_on_{group}_f{frac}")]:
            # eval results are under <tag>/<group>/eval_results.json
            results_path = out_dir / tag / group / "eval_results.json"
            manifest_pattern = f"freeze_{group}_f{frac}" if profile_type == "domain" else f"freeze_general_on_{group}_f{frac}"
            manifest_path = out_dir / manifest_pattern / "freeze_manifest.json"

            iou = miss = fp = skipped = frozen = attn_pct = "N/A"
            if results_path.exists():
                r = json.load(open(results_path))
                iou = f"{r.get('per_image_mean_iou', r.get('overall_mean_iou', 0)):.4f}"
                miss = f"{r.get('per_image_miss_rate', 0) * 100:.1f}"
                fp = f"{r.get('per_image_false_pos_rate', 0) * 100:.1f}"
                skipped = str(r.get("skipped_images", 0))
            if manifest_path.exists():
                m = json.load(open(manifest_path))
                frozen = str(m.get("summary", {}).get("frozen", "?"))
                attn_pct = f"{m.get('flops', {}).get('attention_flop_reduction_pct', 0):.1f}"

            print(f"{group:<15} {frac:>5.3f} {profile_type:>12} {iou:>10} {miss:>8} {fp:>8} {skipped:>8} {frozen:>8} {attn_pct:>8}")
            rows.append({
                "domain": group, "fraction": frac, "profile": profile_type,
                "mean_iou": iou, "miss_rate": miss, "fp_rate": fp,
                "skipped": skipped, "frozen_heads": frozen, "attn_flop_pct": attn_pct,
            })
    print()

# Save JSON
summary_path = out_dir / "cross_domain_summary.json"
json.dump(rows, open(summary_path, "w"), indent=2)
print(f"Summary JSON saved: {summary_path}")
print()
print("=" * 110)
print("KEY: Compare 'domain' vs 'general' rows for each (domain, fraction) pair.")
print("If domain-specific IoU > general IoU → specialized profiling helps.")
print("=" * 110)
