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
# MODELS: space-separated; each entry is model_name:manifest_path:batch_sizes
# Defaults point at the sweep outputs of experiments/vit/exp_sweep.sh.
MODELS="${MODELS:-vit_base_patch16_224:results/vit/sweep/vit_base/freeze_manifest.json:1,8 dinov2_giant:results/vit/sweep/dinov2_giant/freeze_manifest.json:1,4}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-results/vit_latency}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
PRETRAINED="${PRETRAINED:-0}"

PRETRAINED_FLAG=""
if [ "$PRETRAINED" = "1" ]; then
    PRETRAINED_FLAG="--pretrained"
fi

mkdir -p "$OUT"
FAILED=0
for spec in $MODELS; do
    MODEL="${spec%%:*}"
    rest="${spec#*:}"
    MANIFEST="${rest%%:*}"
    BATCH_SIZES="${rest#*:}"
    if [ ! -f "$MANIFEST" ]; then
        # fall back to any manifest for this model under results/
        MANIFEST="$(find results -path "*${MODEL%%_*}*" -name freeze_manifest.json 2>/dev/null | head -1)"
    fi
    if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
        echo "SKIP $MODEL: no freeze manifest found (run exp_sweep.sh first or set MODELS=)"
        FAILED=1
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ViT latency: model=$MODEL manifest=$MANIFEST batches=$BATCH_SIZES"
    # shellcheck disable=SC2086
    python3 vit/bench_latency.py \
        --model "$MODEL" \
        --freeze_manifest "$MANIFEST" \
        --batch_sizes "$BATCH_SIZES" \
        --device "$DEVICE" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --seed "$SEED" \
        --out "$OUT/latency_${MODEL}.json" \
        $PRETRAINED_FLAG
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $OUT/"
exit $FAILED
