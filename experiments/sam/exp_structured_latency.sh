#!/bin/bash
# exp_structured_latency.sh - Wall-clock and energy benchmark for SAM ViT-H:
# baseline vs hooked (pre-hook mean replacement) vs sliced (structured removal).
#
# Times the encoder-only forward AND the full AutomaticMaskGenerator per image
# over EVAL_N images; records NVML energy when available; writes JSON and
# prints a summary table.
#
# Usage (from repo root):
#   bash experiments/sam/exp_structured_latency.sh
#   DOMAIN=isic FRAC=0.5 EVAL_N=50 bash experiments/sam/exp_structured_latency.sh
#
# TODO: extend to MLP freezing (sam_velcro_pkg/mlp_hooks.py) once MLP surgery
# (structured neuron removal) exists; this script covers attention heads only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH=.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DOMAIN="${DOMAIN:-kvasir}"
FRAC="${FRAC:-0.25}"
EVAL_N="${EVAL_N:-20}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
CKPT="${CKPT:-sam/data/sam_vit_h_4b8939.pth}"
MANIFEST="${MANIFEST:-sam/results_${DOMAIN}_frac${FRAC}/freeze_manifest.json}"
IMG_DIR="${IMG_DIR:-sam/image_dirs_cross_domain/${DOMAIN}_eval}"
OUT="${OUT:-results/sam_structured_latency/${DOMAIN}_frac${FRAC}}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-16}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
    echo "ERROR: SAM checkpoint not found: $CKPT (set CKPT=)"
    exit 1
fi
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: freeze manifest not found: $MANIFEST"
    echo "  Run a freeze phase for domain=$DOMAIN frac=$FRAC first, or set MANIFEST=."
    exit 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$OUT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SAM structured latency: domain=$DOMAIN frac=$FRAC eval_n=$EVAL_N"

python3 sam/bench_latency.py \
    --checkpoint "$CKPT" \
    --freeze_manifest "$MANIFEST" \
    --image_dir "$IMG_DIR" \
    --eval_n "$EVAL_N" \
    --device "$DEVICE" \
    --points_per_side "$POINTS_PER_SIDE" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --seed "$SEED" \
    --out "$OUT/latency.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $OUT/latency.json"
