#!/usr/bin/env bash
# summarize_coverage.sh — Print summary from existing coverage comparison results
#
# Usage:
#   bash scripts/summarize_coverage.sh
#   OUT_DIR=./results_coverage_comparison bash scripts/summarize_coverage.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-./results_coverage_comparison}"

if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUT_DIR"
    exit 1
fi

echo ""
echo "================================================================"
echo "  Coverage Comparison Summary (ref SAM vs frozen SAM)"
echo "  Source: $OUT_DIR"
echo "================================================================"
printf "%-11s %7s %8s %7s %8s %10s %10s\n" \
    "Domain" "Ref%" "Frozen%" "Drop" "MaxDrop" "AnyLoss" "FullLoss"
printf "%-11s %7s %8s %7s %8s %10s %10s\n" \
    "-----------" "-------" "--------" "-------" "--------" "----------" "----------"

for domain in kvasir isic busi pcb deepglobe; do
    JSON="$OUT_DIR/$domain/coverage_comparison.json"
    [ -f "$JSON" ] || continue

    python3 -c "
import json
with open('$JSON') as f:
    d = json.load(f)
s = d['summary']
total = d.get('num_in_drop_analysis', d.get('num_with_gt', 0))
ref = s['ref_mean_coverage'] * 100
frz = s['frozen_mean_coverage'] * 100
drop = s['mean_drop'] * 100
maxd = s['max_drop'] * 100
any_l = s['images_any_drop']
full_l = s['images_full_loss']
print(f\"{'$domain':<11} {ref:>6.1f}% {frz:>7.1f}% {drop:>5.1f}pp {maxd:>6.1f}pp {any_l:>4}/{total:<4}  {full_l:>4}/{total:<4}\")
"
done

echo "================================================================"
echo ""

# Worst images per domain
echo "Top 3 worst images per domain:"
echo ""

for domain in kvasir isic busi pcb deepglobe; do
    JSON="$OUT_DIR/$domain/coverage_comparison.json"
    [ -f "$JSON" ] || continue

    python3 -c "
import json
with open('$JSON') as f:
    d = json.load(f)
worst = d.get('worst_images', [])[:3]
if not worst:
    return
print(f\"  {d['domain'].upper()}:\")
for w in worst:
    print(f\"    {w['filename']}: ref={w['ref_coverage']*100:.1f}% -> frozen={w['frozen_coverage']*100:.1f}% (drop={w['drop']*100:.1f}pp)\")
print()
"
done

echo "================================================================"
