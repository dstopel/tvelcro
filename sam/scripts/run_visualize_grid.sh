#!/usr/bin/env bash
# run_visualize_grid.sh — Generate 3×4 grid figures for all datasets
#
# Each figure shows:
#   Row 1 (BEST):   image where freezing preserves quality best
#   Row 2 (MEDIAN): typical image
#   Row 3 (WORST):  image where freezing hurts most
#   Columns: [Original] [level 1] [level 2] [level 3]
#
# Produces two figures per dataset:
#   {domain}_attention_grid.png — attention-only sweep
#   {domain}_combined_grid.png  — attn 25% + MLP sweep
#
# Usage:
#   bash scripts/run_visualize_grid.sh
#   RANK_IMAGES=30 bash scripts/run_visualize_grid.sh  # faster ranking

set -euo pipefail

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./visualizations}"
DEVICE="${DEVICE:-cuda}"
RANK_IMAGES="${RANK_IMAGES:-50}"

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
echo "  Generating 3×4 Grid Visualizations"
echo "  Ranking from: $RANK_IMAGES images per dataset"
echo "  Output: $OUT_DIR/"
echo "================================================================"
echo ""

for domain in kvasir isic busi pcb; do
    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        echo ">>> $domain — SKIP (no image dir)"
        continue
    fi

    # Figure 1: Attention-only sweep
    echo ">>> $domain — attention grid"
    python3 scripts/visualize_grid.py \
        --checkpoint "$CKPT" \
        --domain "$domain" \
        --image_dir "$img_dir" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        --mode attention \
        --rank_images "$RANK_IMAGES"

    # Figure 2: Combined sweep (only if MLP profiles exist)
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
            --rank_images "$RANK_IMAGES"
    else
        echo ">>> $domain — combined grid SKIP (no MLP profile)"
    fi

    echo ""
done

echo "================================================================"
echo "  Done. Figures:"
ls "$OUT_DIR"/*_grid.png 2>/dev/null | sed 's/^/    /'
echo "================================================================"
