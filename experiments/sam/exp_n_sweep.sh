#!/bin/bash
# =============================================================================
# exp_n_sweep.sh - Profile sample-size sweep (how many images does profiling need?)
#
# Profiles each domain with --max_images in {25,50,100,200,300}, freezes at
# fraction 0.25 from each profile, evaluates, then runs analyze_profile.py on
# the per-N profiles for Spearman rank-stability (do small-N profiles rank
# heads the same as large-N profiles?).
#
# Run from repo root:
#   bash experiments/sam/exp_n_sweep.sh
#   DOMAINS="kvasir" N_VALUES="25 100" bash experiments/sam/exp_n_sweep.sh  # quick test
#
# Expects $IMG_DIR/<domain>_profile/ and $IMG_DIR/<domain>_eval/ image dirs.
#
# ETA (A10 24GB): profiling 25+50+100+200+300=675 images ~25-35 min per
# domain; eval (N_EVAL=200) ~15-20 min per N value. 2 domains x 5 N values:
# ~4-5 hours total.
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config (env-overridable) ------------------------------------------------
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./results}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/image_dirs}"
CKPT="${CKPT:-${DATA_ROOT}/sam_vit_h_4b8939.pth}"
DOMAINS="${DOMAINS:-kvasir pcb}"
N_VALUES="${N_VALUES:-25 50 100 200 300}"
N_EVAL="${N_EVAL:-200}"
FRAC="${FRAC:-0.25}"
SEED="${SEED:-0}"

export PYTHONPATH=.
OUT="${OUT_ROOT}/sam_n_sweep"
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

# --- Phase 1+2: Per-N profile, freeze at $FRAC, eval -------------------------
for D in $DOMAINS; do
    if [ ! -d "$IMG_DIR/${D}_profile" ]; then
        log "SKIP $D (no image dir at $IMG_DIR/${D}_profile)"
        continue
    fi
    for NP in $N_VALUES; do
        PROF_DIR="$OUT/profile_n${NP}"
        if [ ! -f "$PROF_DIR/$D/block0_head0_variance.npy" ]; then
            log "Profiling $D with N=$NP..."
            python3 sam/sam_velcro.py --mode profile \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --domain_groups "$D" \
                --max_images "$NP" \
                --out_dir "$PROF_DIR" \
                2>&1 | tee "$LOG_DIR/profile_${D}_n${NP}.log"
        else
            log "SKIP profile $D N=$NP (exists)"
        fi

        FREEZE_TAG="freeze_${D}_n${NP}"
        EVAL_TAG="eval_${D}_n${NP}"
        if [ -f "$OUT/$EVAL_TAG/$D/eval_results.json" ]; then
            log "SKIP $EVAL_TAG (results exist)"
            continue
        fi
        log "--- $D profiled at N=$NP, freeze f=$FRAC, eval ---"
        python3 sam/sam_velcro.py --mode freeze \
            --profile_dir "$PROF_DIR" \
            --profile_group "$D" \
            --freeze_fraction "$FRAC" \
            --seed "$SEED" \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee "$LOG_DIR/${FREEZE_TAG}.log"

        python3 sam/sam_velcro.py --mode eval \
            --checkpoint "$CKPT" \
            --image_dir "$IMG_DIR" \
            --data_root "$DATA_ROOT" \
            --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
            --eval_groups "$D" \
            --max_images "$N_EVAL" \
            --diagnostic_level 2 \
            --out_dir "$OUT/$EVAL_TAG" \
            2>&1 | tee "$LOG_DIR/${EVAL_TAG}.log"
    done
done

# --- Phase 3: Spearman rank stability across N -------------------------------
# analyze_profile.py computes cross-group Spearman correlations, so expose the
# per-N profiles of one domain as sibling groups via symlinks.
STAB_DIR="$OUT/profile_stability"
mkdir -p "$STAB_DIR"

for D in $DOMAINS; do
    GROUPS=""
    for NP in $N_VALUES; do
        SRC="$OUT/profile_n${NP}/$D"
        if [ -d "$SRC" ]; then
            ln -sfn "$(cd "$SRC" && pwd)" "$STAB_DIR/${D}_n${NP}"
            GROUPS="${GROUPS:+$GROUPS,}${D}_n${NP}"
        fi
    done
    if [ -z "$GROUPS" ]; then
        continue
    fi
    log "Rank stability for $D: groups $GROUPS"
    python3 sam/analyze_profile.py \
        --profile_dir "$STAB_DIR" \
        --groups "$GROUPS" \
        --out "$OUT/rank_stability_${D}.txt" \
        2>&1 | tee "$LOG_DIR/rank_stability_${D}.log"
    log "  Report: $OUT/rank_stability_${D}.txt (see CROSS-GROUP RANK CORRELATION table)"
done

log "Done. Results in $OUT/"
