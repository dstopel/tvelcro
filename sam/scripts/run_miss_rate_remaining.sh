#!/usr/bin/env bash
# run_miss_rate_remaining.sh — Continue miss rate analysis from PCB onward
#
# Usage:
#   bash scripts/run_miss_rate_remaining.sh
#   DEBUG=true bash scripts/run_miss_rate_remaining.sh

set -euo pipefail

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results_miss_analysis}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-true}"

IMG_DIR_W1="${IMG_DIR_W1:-./image_dirs_cross_domain}"
IMG_DIR_W2="${IMG_DIR_W2:-./image_dirs_wave2}"

MANIFEST_DIRS=(
    "./results_cross_domain"
    "./results_wave2"
    "./results_sam_velcro"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Remaining domains (pcb onward)
DOMAINS=(pcb deepglobe)
FRACTIONS=(0.125 0.25 0.375)

get_image_dir() {
    local domain="$1"
    if [ -d "$IMG_DIR_W1/${domain}_eval" ]; then echo "$IMG_DIR_W1"; return; fi
    if [ -d "$IMG_DIR_W2/${domain}_eval" ]; then echo "$IMG_DIR_W2"; return; fi
    if [ -d "$IMG_DIR_W1/$domain" ]; then echo "$IMG_DIR_W1"; return; fi
    if [ -d "$IMG_DIR_W2/$domain" ]; then echo "$IMG_DIR_W2"; return; fi
    echo ""
}

find_manifest() {
    local domain="$1"
    local frac="$2"
    for base_dir in "${MANIFEST_DIRS[@]}"; do
        local p="$base_dir/$domain/freeze_${domain}_f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return
        p="$base_dir/freeze_${domain}_f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return
        p="$base_dir/$domain/f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return
    done
    echo ""
}

MAX_IMG_ARG=""
if [ "$MAX_IMAGES" -gt 0 ] 2>/dev/null; then
    MAX_IMG_ARG="--max_images $MAX_IMAGES"
fi

echo "================================================================"
echo "  Miss Rate Analysis — Remaining Domains"
echo "  Domains: ${DOMAINS[*]}"
echo "  Fractions: ${FRACTIONS[*]}"
echo "================================================================"
echo ""

PASS=0; SKIP=0; FAIL=0

for domain in kvasir isic busi pcb deepglobe; do
    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        echo "  [SKIP] No eval images for $domain"
        SKIP=$((SKIP + ${#FRACTIONS[@]}))
        continue
    fi

    for frac in "${FRACTIONS[@]}"; do
        run_out="$OUT_DIR/$domain/f${frac}"

        if [ "$FORCE" != "true" ] && [ -f "$run_out/miss_analysis_results.json" ]; then
            echo "  [SKIP] $domain @ f=$frac — already done"
            SKIP=$((SKIP + 1))
            continue
        fi

        manifest=$(find_manifest "$domain" "$frac")
        if [ -z "$manifest" ]; then
            echo "  [SKIP] $domain @ f=$frac — no manifest found"
            SKIP=$((SKIP + 1))
            continue
        fi

        echo "  [RUN] $domain @ f=$frac"
        echo "        manifest: $manifest"
        echo "        img_dir:  $img_dir"

        if python3 scripts/miss_rate_analysis.py \
            --domain "$domain" \
            --checkpoint "$CKPT" \
            --freeze_manifest "$manifest" \
            --image_dir "$img_dir" \
            --data_root "$DATA_ROOT" \
            --out_dir "$run_out" \
            --device "$DEVICE" \
            $MAX_IMG_ARG; then
            echo "  ✓ $domain @ f=$frac — done"
            PASS=$((PASS + 1))
        else
            echo "  ✗ $domain @ f=$frac — FAILED"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done
done

echo "================================================================"
echo "  Done: $PASS passed, $SKIP skipped, $FAIL failed"
echo "================================================================"
