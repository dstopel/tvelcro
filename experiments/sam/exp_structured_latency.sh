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
# Runs two configurations:
#   1) attention-only (FRAC of heads)            -> $OUT/latency.json
#   2) OPTIONAL combined attention + MLP, off by default (MLP_FRACTION=0)
#      (uses sam/surgery_mlp.py bias-folded structured neuron removal; needs
#       the MLP profile written by exp_main_sweep.sh --profile_mlp)
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
DATA_ROOT="${DATA_ROOT:-./data}"
CKPT="${CKPT:-${DATA_ROOT}/sam_vit_h_4b8939.pth}"
SWEEP_OUT="${SWEEP_OUT:-results/sam_main_sweep}"
MANIFEST="${MANIFEST:-${SWEEP_OUT}/freeze_${DOMAIN}_f${FRAC}/freeze_manifest.json}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/image_dirs/${DOMAIN}_eval}"
OUT="${OUT:-results/sam_structured_latency/${DOMAIN}_frac${FRAC}}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-16}"
MLP_PROFILE_DIR="${MLP_PROFILE_DIR:-${SWEEP_OUT}/profile}"
MLP_FRACTION="${MLP_FRACTION:-0}"  # combined config is opt-in; attention-only by default

# fall back to any manifest for this domain if the default location is empty
if [ ! -f "$MANIFEST" ]; then
    FOUND="$(find results -path "*${DOMAIN}*" -name freeze_manifest.json 2>/dev/null | head -1)"
    [ -n "$FOUND" ] && MANIFEST="$FOUND"
fi

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

# ---------------------------------------------------------------------------
# Combined configuration: attention FRAC + MLP_FRACTION of neurons
# ---------------------------------------------------------------------------
if [ "$(python3 -c "print(float('$MLP_FRACTION') > 0)")" = "True" ] && ls "$MLP_PROFILE_DIR/$DOMAIN"/block*_mlp_variance.npy >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Combined config: attention $FRAC + MLP $MLP_FRACTION"
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
        --mlp_profile_dir "$MLP_PROFILE_DIR" \
        --mlp_group "$DOMAIN" \
        --mlp_fraction "$MLP_FRACTION" \
        --out "$OUT/latency_combined.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $OUT/latency_combined.json"
else
    echo "Combined config disabled (attention-only run; set MLP_FRACTION=0.375 to enable)"
fi
