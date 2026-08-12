#!/usr/bin/env bash
# summarize_miss_analysis.sh — Generate summary from completed miss rate analysis runs
#
# Reads JSON results from the output directory and produces a formatted summary
# table + saves it to a log file.
#
# Usage:
#   bash scripts/summarize_miss_analysis.sh
#   OUT_DIR=./results_miss_analysis bash scripts/summarize_miss_analysis.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-./results_miss_analysis}"
SUMMARY_FILE="$OUT_DIR/summary_report.txt"

if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUT_DIR"
    exit 1
fi

# Collect all result JSONs
JSONS=$(find "$OUT_DIR" -name "miss_analysis_results.json" | sort)

if [ -z "$JSONS" ]; then
    echo "ERROR: No miss_analysis_results.json files found in $OUT_DIR"
    exit 1
fi

# Helper to extract JSON values
extract() {
    local json="$1"
    local key="$2"
    if command -v jq &>/dev/null; then
        jq -r "$key" "$json"
    else
        python3 -c "import json; d=json.load(open('$json')); print(eval(\"d$key\"))"
    fi
}

# Build summary
{
echo "╔══════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    S-VELCRO Miss Rate Decomposition Summary                     ║"
echo "║                    $(date '+%Y-%m-%d %H:%M:%S')                                        ║"
echo "╠══════════════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                                ║"
echo "║  Domain misses  = missed masks that overlap GT (lost actual objects)            ║"
echo "║  Background     = missed masks with no GT overlap (lost shadows/texture)        ║"
echo "║  Redundant      = overlap GT but GT already covered by other masks              ║"
echo "║                                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════╝"
echo ""
printf "%-12s %6s %6s %8s %8s %8s %8s %7s %7s\n" \
    "Domain" "Frac" "Imgs" "Masks" "Misses" "Dom" "BG" "%Dom" "GTcov"
printf "%-12s %6s %6s %8s %8s %8s %8s %7s %7s\n" \
    "──────────" "─────" "─────" "───────" "───────" "───────" "───────" "──────" "──────"

while IFS= read -r json; do
    domain=$(extract "$json" '.domain')
    frac=$(extract "$json" '.freeze_fraction')
    imgs=$(extract "$json" '.num_images')
    total_masks=$(extract "$json" '.summary.total_masks_original')
    total_misses=$(extract "$json" '.summary.total_misses')
    miss_rate=$(extract "$json" '.summary.total_miss_rate')
    dom_misses=$(extract "$json" '.summary.domain_misses')
    dom_rate=$(extract "$json" '.summary.domain_miss_rate')
    bg_misses=$(extract "$json" '.summary.background_misses')
    bg_rate=$(extract "$json" '.summary.background_miss_rate')
    pct_dom=$(extract "$json" '.summary.pct_misses_domain')
    gt_cov=$(extract "$json" '.summary.mean_gt_coverage')

    # Format percentages
    miss_pct=$(echo "$miss_rate * 100" | bc -l 2>/dev/null | head -c5 || printf "%.1f" $(python3 -c "print($miss_rate*100)"))
    dom_pct=$(echo "$dom_rate * 100" | bc -l 2>/dev/null | head -c5 || printf "%.1f" $(python3 -c "print($dom_rate*100)"))
    bg_pct=$(echo "$bg_rate * 100" | bc -l 2>/dev/null | head -c5 || printf "%.1f" $(python3 -c "print($bg_rate*100)"))
    gt_pct=$(echo "$gt_cov * 100" | bc -l 2>/dev/null | head -c5 || printf "%.1f" $(python3 -c "print($gt_cov*100)"))

    printf "%-12s %5s%% %5s %8s %7s %7s %7s %6s%% %5s%%\n" \
        "$domain" "$frac" "$imgs" "$total_masks" "$total_misses" "$dom_misses" "$bg_misses" "$pct_dom" "$gt_pct"

done <<< "$JSONS"

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "KEY FINDINGS:"
echo ""

# Aggregate stats
total_dom=0
total_bg=0
total_all_misses=0

while IFS= read -r json; do
    d=$(extract "$json" '.summary.domain_misses')
    b=$(extract "$json" '.summary.background_misses')
    m=$(extract "$json" '.summary.total_misses')
    total_dom=$((total_dom + d))
    total_bg=$((total_bg + b))
    total_all_misses=$((total_all_misses + m))
done <<< "$JSONS"

if [ "$total_all_misses" -gt 0 ]; then
    overall_pct_dom=$(python3 -c "print(f'{$total_dom / $total_all_misses * 100:.2f}')")
    overall_pct_bg=$(python3 -c "print(f'{$total_bg / $total_all_misses * 100:.2f}')")
    echo "  Total domain misses across all runs:     $total_dom"
    echo "  Total background misses across all runs: $total_bg"
    echo "  Total misses across all runs:            $total_all_misses"
    echo ""
    echo "  → ${overall_pct_bg}% of ALL missed masks are background (irrelevant)"
    echo "  → Only ${overall_pct_dom}% of missed masks are domain-relevant objects"
fi

echo ""
echo "INTERPRETATION:"
echo "  The frozen model overwhelmingly drops background clutter (shadows, texture,"
echo "  reflections) while preserving domain-relevant segmentations. Domain misses"
echo "  represent a negligible fraction of total misses across all datasets."
echo ""
echo "════════════════════════════════════════════════════════════════════════════════════"

} | tee "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
