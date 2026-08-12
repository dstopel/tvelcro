#!/bin/bash
# exp_latency.sh - Wall-clock and energy benchmark for timm vit_base_patch16_224:
# baseline vs hooked (pre-hook mean replacement) vs sliced (structured removal).
#
# Times the full-model forward at batch 1 and batch 8 (images/s); records NVML
# energy when available; writes JSON and prints a summary table.
#
# Usage (from repo root):
#   bash experiments/vit/exp_latency.sh
#   MANIFEST=path/to/freeze_manifest.json BATCH_SIZES=1,8,32 bash experiments/vit/exp_latency.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH=.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="${MODEL:-vit_base_patch16_224}"
MANIFEST="${MANIFEST:-vit/results/freeze_manifest.json}"
BATCH_SIZES="${BATCH_SIZES:-1,8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results/vit_latency}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
PRETRAINED="${PRETRAINED:-0}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: freeze manifest not found: $MANIFEST"
    echo "  Run a ViT freeze phase first, or set MANIFEST=."
    exit 1
fi

PRETRAINED_FLAG=""
if [ "$PRETRAINED" = "1" ]; then
    PRETRAINED_FLAG="--pretrained"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$OUT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ViT latency: model=$MODEL batches=$BATCH_SIZES"

# shellcheck disable=SC2086
python3 vit/bench_latency.py \
    --model "$MODEL" \
    --freeze_manifest "$MANIFEST" \
    --batch_sizes "$BATCH_SIZES" \
    --device "$DEVICE" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --seed "$SEED" \
    --out "$OUT/latency.json" \
    $PRETRAINED_FLAG

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $OUT/latency.json"
