#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Coverage Comparison Experiment
# Runs domain_coverage_comparison.py across all 5 datasets at a given freeze
# fraction (default 37.5%) and prints a combined summary table.
# ==============================================================================

# --- Configuration (env-overridable) ------------------------------------------
CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results_coverage_comparison}"
MAX_IMAGES="${MAX_IMAGES:-0}"  # 0 = all
DEVICE="${DEVICE:-cuda}"
FRACTION="${FRACTION:-0.375}"  # default to most aggressive

DOMAINS=(kvasir isic busi deepglobe pcb)
FORCE="${FORCE:-true}"

# --- Flags --------------------------------------------------------------------
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "  --dry-run   Print commands without executing"
            echo ""
            echo "Environment variables:"
            echo "  CKPT         SAM checkpoint (default: ./data/sam_vit_h_4b8939.pth)"
            echo "  DATA_ROOT    Data root for GT masks (default: ./data)"
            echo "  OUT_DIR      Output directory (default: ./results_coverage_comparison)"
            echo "  MAX_IMAGES   Max images per domain, 0=all (default: 0)"
            echo "  DEVICE       Torch device (default: cuda)"
            echo "  FRACTION     Freeze fraction (default: 0.375)"
            echo "  FORCE        Always re-run (default: true)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Helpers ------------------------------------------------------------------
get_image_dir() {
    local domain="$1"
    case "$domain" in
        kvasir|isic)
            if [[ -d "./image_dirs_cross_domain/${domain}_eval" ]]; then
                echo "./image_dirs_cross_domain"
            elif [[ -d "./image_dirs_wave2/${domain}_eval" ]]; then
                echo "./image_dirs_wave2"
            fi ;;
        busi|pcb|deepglobe)
            if [[ -d "./image_dirs_wave2/${domain}_eval" ]]; then
                echo "./image_dirs_wave2"
            elif [[ -d "./image_dirs_cross_domain/${domain}_eval" ]]; then
                echo "./image_dirs_cross_domain"
            fi ;;
    esac
}

find_manifest() {
    local domain="$1" f="$2"
    local candidates=(
        "./results_cross_domain/${domain}/freeze_${domain}_f${f}/freeze_manifest.json"
        "./results_cross_domain/freeze_${domain}_f${f}/freeze_manifest.json"
        "./results_wave2/${domain}/freeze_${domain}_f${f}/freeze_manifest.json"
        "./results_wave2/freeze_${domain}_f${f}/freeze_manifest.json"
        "./results_sam_velcro/${domain}/freeze_${domain}_f${f}/freeze_manifest.json"
    )
    for c in "${candidates[@]}"; do
        if [[ -f "$c" ]]; then echo "$c"; return 0; fi
    done
    return 1
}

# --- Verify checkpoint --------------------------------------------------------
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: SAM checkpoint not found: $CKPT" >&2; exit 1
fi

# --- Main loop ----------------------------------------------------------------
echo "================================================================"
echo "  Coverage Comparison — ${FRACTION} freeze fraction"
echo "  Domains: ${DOMAINS[*]}"
echo "================================================================"

mkdir -p "$OUT_DIR"
PASS=0; FAIL=0

for domain in kvasir isic busi pcb deepglobe; do
    echo ""
    echo ">>> $domain"

    IMG_DIR=$(get_image_dir "$domain")
    if [[ -z "${IMG_DIR:-}" ]]; then
        echo "  SKIP: image dir not found"; FAIL=$((FAIL+1)); continue
    fi

    MANIFEST=$(find_manifest "$domain" "$FRACTION" 2>/dev/null) || true
    if [[ -z "${MANIFEST:-}" ]]; then
        echo "  SKIP: manifest not found for f=$FRACTION"; FAIL=$((FAIL+1)); continue
    fi

    CMD="python3 scripts/domain_coverage_comparison.py"
    CMD+=" --domain $domain"
    CMD+=" --checkpoint $CKPT"
    CMD+=" --freeze_manifest $MANIFEST"
    CMD+=" --image_dir $IMG_DIR"
    CMD+=" --data_root $DATA_ROOT"
    CMD+=" --out_dir $OUT_DIR/$domain"
    CMD+=" --device $DEVICE"
    if [[ "$MAX_IMAGES" -gt 0 ]]; then CMD+=" --max_images $MAX_IMAGES"; fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY-RUN] $CMD"; continue
    fi

    echo "  Running..."
    mkdir -p "$OUT_DIR/$domain"
    set +e
    eval "$CMD" 2>&1
    RC=$?
    set -e

    if [[ $RC -eq 0 ]]; then
        echo "  PASS"; PASS=$((PASS+1))
    else
        echo "  FAIL (exit $RC)"; FAIL=$((FAIL+1))
    fi
done

# --- Summary table ------------------------------------------------------------
[[ "$DRY_RUN" -eq 1 ]] && exit 0

echo ""
echo "================================================================"
echo "  Coverage Comparison Summary (all domains @ ${FRACTION} freeze)"
echo "================================================================"
printf "%-11s %-7s %-8s %-6s %-8s %-8s %-8s\n" \
    "Domain" "Ref%" "Frozen%" "Drop" "MaxDrop" "AnyLoss" "FullLoss"
printf "%-11s %-7s %-8s %-6s %-8s %-8s %-8s\n" \
    "-----------" "-------" "--------" "------" "--------" "--------" "--------"

for domain in kvasir isic busi pcb deepglobe; do
    JSON="$OUT_DIR/$domain/coverage_comparison.json"
    [[ -f "$JSON" ]] || continue

    # Use jq if available, otherwise python fallback
    if command -v jq &>/dev/null; then
        ref=$(jq -r '.summary.ref_mean_coverage // empty' "$JSON" 2>/dev/null)
        frz=$(jq -r '.summary.frozen_mean_coverage // empty' "$JSON" 2>/dev/null)
        drop=$(jq -r '.summary.mean_drop // empty' "$JSON" 2>/dev/null)
        maxd=$(jq -r '.summary.max_drop // empty' "$JSON" 2>/dev/null)
        any=$(jq -r '.summary.images_any_drop // empty' "$JSON" 2>/dev/null)
        full=$(jq -r '.summary.images_full_loss // empty' "$JSON" 2>/dev/null)
        total=$(jq -r '.num_in_drop_analysis // .num_with_gt // empty' "$JSON" 2>/dev/null)
    else
        read -r ref frz drop maxd any full total < <(python3 -c "
import json, sys
with open('$JSON') as f: d = json.load(f)
s = d.get('summary', {})
print(s.get('ref_mean_coverage',''), s.get('frozen_mean_coverage',''),
      s.get('mean_drop',''), s.get('max_drop',''),
      s.get('images_any_drop',''), s.get('images_full_loss',''),
      d.get('num_in_drop_analysis', d.get('num_with_gt','')))
" 2>/dev/null || echo "")
    fi

    [[ -z "${ref:-}" ]] && continue
    ref_pct=$(python3 -c "print(f'{float($ref)*100:.1f}')")
    frz_pct=$(python3 -c "print(f'{float($frz)*100:.1f}')")
    drop_pp=$(python3 -c "print(f'{float($drop)*100:.1f}')")
    maxd_pp=$(python3 -c "print(f'{float($maxd)*100:.1f}')")
    printf "%-11s %5s%%  %5s%%   %4spp %5spp  %s/%s   %s/%s\n" \
        "$domain" "$ref_pct" "$frz_pct" "$drop_pp" "$maxd_pp" "$any" "$total" "$full" "$total"
done

echo "================================================================"
echo ""
echo "Results: PASS=$PASS  FAIL=$FAIL"
