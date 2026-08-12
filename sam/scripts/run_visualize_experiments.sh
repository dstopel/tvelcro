#!/usr/bin/env bash
# run_visualize_experiments.sh — Generate all comparison figures for all datasets
#
# Produces per dataset:
#   {domain}_attention_sweep.png:  [Original] [12.5%] [25%] [37.5%]
#   {domain}_combined_sweep.png:   [Original] [+MLP 12.5%] [+MLP 25%] [+MLP 37.5%]
#
# Usage:
#   bash scripts/run_visualize_experiments.sh
#   IMAGE_INDEX=10 bash scripts/run_visualize_experiments.sh

set -euo pipefail

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./visualizations}"
IMAGE_INDEX="${IMAGE_INDEX:-5}"
DEVICE="${DEVICE:-cuda}"

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
echo "  Generating Experiment Visualizations"
echo "  Image index: $IMAGE_INDEX"
echo "  Output: $OUT_DIR/"
echo "================================================================"
echo ""

for domain in kvasir isic busi pcb; do
    echo ">>> $domain"

    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        echo "  SKIP: no image dir"
        echo ""
        continue
    fi

    python3 scripts/visualize_experiments.py \
        --checkpoint "$CKPT" \
        --domain "$domain" \
        --image_dir "$img_dir" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        --image_index "$IMAGE_INDEX"

    echo ""
done

echo "================================================================"
echo "  Done. Figures saved to: $OUT_DIR/"
echo "  Files:"
ls "$OUT_DIR"/*.png 2>/dev/null | sed 's/^/    /'
echo "================================================================"
