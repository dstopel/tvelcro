#!/usr/bin/env bash
# run_visualize_all.sh — Generate one comparison image per dataset
#
# Usage:
#   bash scripts/run_visualize_all.sh

set -euo pipefail

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results_combined_freeze/visualizations}"
MLP_FRACTION="${MLP_FRACTION:-0.25}"
IMAGE_INDEX="${IMAGE_INDEX:-5}"
DEVICE="${DEVICE:-cuda}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

IMG_DIR_W1="./image_dirs_cross_domain"
IMG_DIR_W2="./image_dirs_wave2"

MANIFEST_DIRS=(
    "./results_cross_domain"
    "./results_wave2"
)

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

find_manifest() {
    local domain="$1"
    for base in "${MANIFEST_DIRS[@]}"; do
        for p in \
            "$base/${domain}/freeze_${domain}_f0.25/freeze_manifest.json" \
            "$base/freeze_${domain}_f0.25/freeze_manifest.json"; do
            [ -f "$p" ] && echo "$p" && return
        done
    done
    echo ""
}

echo "================================================================"
echo "  Generating Comparison Visualizations"
echo "  MLP fraction: ${MLP_FRACTION}"
echo "  Image index: ${IMAGE_INDEX}"
echo "================================================================"
echo ""

for domain in kvasir isic busi pcb; do
    echo ">>> $domain"

    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        echo "  SKIP: no image dir found"
        continue
    fi

    manifest=$(find_manifest "$domain")
    if [ -z "$manifest" ]; then
        echo "  SKIP: no attention manifest found"
        continue
    fi

    mlp_profile="./results_combined_freeze/$domain/mlp_profile"
    if [ ! -d "$mlp_profile" ]; then
        echo "  SKIP: no MLP profile found at $mlp_profile"
        echo "  Run the combined experiment first to generate profiles."
        continue
    fi

    python3 scripts/visualize_combined_freeze.py \
        --checkpoint "$CKPT" \
        --domain "$domain" \
        --attn_manifest "$manifest" \
        --mlp_profile_dir "$mlp_profile" \
        --mlp_fraction "$MLP_FRACTION" \
        --image_dir "$img_dir" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        --image_index "$IMAGE_INDEX"

    echo ""
done

echo "================================================================"
echo "  Done. Images saved to: $OUT_DIR/"
echo "================================================================"
