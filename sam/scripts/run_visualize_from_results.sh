#!/usr/bin/env bash
# run_visualize_from_results.sh — Generate grid figures using existing coverage results
#
# Picks best/median/worst images from coverage_comparison.json (no re-ranking needed).
# Still runs SAM inference on those 3 images to draw the mask overlays.
#
# Usage:
#   bash scripts/run_visualize_from_results.sh

set -euo pipefail

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./visualizations}"
DEVICE="${DEVICE:-cuda}"
COV_DIR="${COV_DIR:-./results_coverage_comparison}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

IMG_DIR_W1="./image_dirs_cross_domain"
IMG_DIR_W2="./image_dirs_wave2"

get_image_dir() {
    local domain="$1"
    case "$domain" in
        kvasir|isic)
            [ -d "$IMG_DIR_W1/${domain}_eval" ] && echo "$IMG_DIR_W1" && return
            [ -d "$IMG_DIR_W2/${domain}_eval" ] && echo "$IMG_DIR_W2" && return ;;
        busi|pcb)
            [ -d "$IMG_DIR_W2/${domain}_eval" ] && echo "$IMG_DIR_W2" && return
            [ -d "$IMG_DIR_W1/${domain}_eval" ] && echo "$IMG_DIR_W1" && return ;;
    esac
    echo ""
}

echo "================================================================"
echo "  Generating Grid Visualizations (from existing results)"
echo "  Coverage results: $COV_DIR"
echo "  Output: $OUT_DIR/"
echo "================================================================"
echo ""

for domain in kvasir isic busi pcb; do
    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        echo ">>> $domain — SKIP (no image dir)"
        continue
    fi

    cov_json="$COV_DIR/$domain/coverage_comparison.json"
    if [ ! -f "$cov_json" ]; then
        echo ">>> $domain — SKIP (no coverage_comparison.json at $cov_json)"
        continue
    fi

    # Attention grid
    echo ">>> $domain — attention grid"
    python3 scripts/visualize_grid.py \
        --checkpoint "$CKPT" \
        --domain "$domain" \
        --image_dir "$img_dir" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        --mode attention \
        --from_json "$cov_json"

    # Combined grid (only if MLP profiles exist)
    mlp_profile="./results_combined_freeze/$domain/mlp_profile"
    if [ -d "$mlp_profile" ]; then
        echo ">>> $domain — combined grid"
        python3 scripts/visualize_grid.py \
            --checkpoint "$CKPT" \
            --domain "$domain" \
            --image_dir "$img_dir" \
            --data_root "$DATA_ROOT" \
            --out_dir "$OUT_DIR" \
            --device "$DEVICE" \
            --mode combined \
            --from_json "$cov_json"
    fi

    echo ""
done

echo "================================================================"
echo "  Done. Figures:"
ls "$OUT_DIR"/*_grid.png 2>/dev/null | sed 's/^/    /'
echo "================================================================"
