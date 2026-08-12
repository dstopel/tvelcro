#!/bin/bash
# =============================================================================
# exp_main_sweep.sh - Main SAM domain sweep (attention-only freeze)
#
# 5 domains x fractions 0.125/0.25/0.375: profile each domain, freeze the
# lowest-variance heads per block, evaluate patched vs original masks.
#
# Run from repo root:
#   bash experiments/sam/exp_main_sweep.sh
#   N=50 DOMAINS="kvasir pcb" bash experiments/sam/exp_main_sweep.sh   # quick test
#
# Expects $IMG_DIR/<domain>_profile/ and $IMG_DIR/<domain>_eval/ image dirs
# (see sam/scripts/exp_cross_domain.sh Phase 0 for dataset download/split)
# and the SAM checkpoint at $CKPT (sam/download_data.sh fetches it).
#
# ETA (A10 24GB, N=300): profile ~10-15 min/domain, eval ~20-30 min per
# (domain, fraction) cell. 5 domains x 3 fractions: ~6-9 hours total.
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config (env-overridable) ------------------------------------------------
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./results}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/image_dirs}"
CKPT="${CKPT:-${DATA_ROOT}/sam_vit_h_4b8939.pth}"
N="${N:-300}"
DOMAINS="${DOMAINS:-kvasir isic busi deepglobe pcb}"
FRACS="${FRACS:-0.125 0.25 0.375}"
SEED="${SEED:-0}"

export PYTHONPATH=.
OUT="${OUT_ROOT}/sam_main_sweep"
LOG_DIR="${OUT}/logs"

if [ ! -f sam/sam_velcro.py ]; then
    echo "ERROR: run from repo root (sam/sam_velcro.py not found)" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $CKPT (run sam/download_data.sh)" >&2
    exit 1
fi

mkdir -p "$OUT" "$LOG_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Phase 1: Profile each domain (also profiles MLP for later combined runs)
for D in $DOMAINS; do
    if [ ! -d "$IMG_DIR/${D}_profile" ]; then
        log "SKIP profile $D (no image dir at $IMG_DIR/${D}_profile)"
        continue
    fi
    if [ -f "$OUT/profile/$D/block0_head0_variance.npy" ]; then
        log "SKIP profile $D (exists)"
        continue
    fi
    log "Profiling $D (N=$N)..."
    python3 sam/sam_velcro.py --mode profile \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --domain_groups "$D" \
        --max_images "$N" \
        --profile_mlp \
        --out_dir "$OUT/profile" \
        2>&1 | tee "$LOG_DIR/profile_${D}.log"
done

# --- Phase 2: Freeze + Eval grid ---------------------------------------------
for D in $DOMAINS; do
    if [ ! -d "$IMG_DIR/${D}_eval" ]; then
        log "SKIP eval $D (no image dir at $IMG_DIR/${D}_eval)"
        continue
    fi
    for F in $FRACS; do
        FREEZE_TAG="freeze_${D}_f${F}"
        EVAL_TAG="eval_${D}_f${F}"
        if [ -f "$OUT/$EVAL_TAG/$D/eval_results.json" ]; then
            log "SKIP $EVAL_TAG (results exist)"
            continue
        fi
        log "--- $D @ f=$F ---"
        python3 sam/sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group "$D" \
            --freeze_fraction "$F" \
            --seed "$SEED" \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee "$LOG_DIR/${FREEZE_TAG}.log"

        python3 sam/sam_velcro.py --mode eval \
            --checkpoint "$CKPT" \
            --image_dir "$IMG_DIR" \
            --data_root "$DATA_ROOT" \
            --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
            --eval_groups "$D" \
            --max_images "$N" \
            --diagnostic_level 2 \
            --out_dir "$OUT/$EVAL_TAG" \
            2>&1 | tee "$LOG_DIR/${EVAL_TAG}.log"
    done
done

# --- Phase 3: Sweep plots ----------------------------------------------------
GROUPS_CSV="$(echo "$DOMAINS" | tr ' ' ',')"
FRACS_CSV="$(echo "$FRACS" | tr ' ' ',')"
log "Generating sweep plots..."
python3 sam/plot_sweep_results.py \
    --results_dir "$OUT" \
    --groups "$GROUPS_CSV" \
    --fractions "$FRACS_CSV" \
    2>&1 | tee "$LOG_DIR/plot_sweep.log" || log "WARN: plotting failed (non-fatal)"

log "Done. Results in $OUT/"
