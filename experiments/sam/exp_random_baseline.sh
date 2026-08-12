#!/bin/bash
# =============================================================================
# exp_random_baseline.sh - Random and entropy ranking baselines
#
# Sanity check for the variance ranking: freeze the same number of heads per
# block chosen (a) uniformly at random (seeded, per block) and (b) by lowest
# entropy, at fractions 0.125/0.25/0.375. If variance ranking is meaningful,
# it should beat random and be comparable or better than entropy.
#
# Run from repo root:
#   bash experiments/sam/exp_random_baseline.sh
#   SEEDS="0" FRACS="0.25" bash experiments/sam/exp_random_baseline.sh  # quick test
#
# Reuses profiles from exp_main_sweep.sh when present (PROFILE_DIR), else
# profiles fresh. Entropy ranking needs profile_summary.json files written by
# the current code (older profiles lack per-head entropy).
#
# ETA (A10 24GB, N=300): eval ~20-30 min per run. Default grid is
# 2 domains x 3 fracs x (3 random seeds + 1 entropy) = 24 evals: ~8-12 hours.
# Reduce N / SEEDS / FRACS for a faster pass.
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Config (env-overridable) ------------------------------------------------
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./results}"
IMG_DIR="${IMG_DIR:-${DATA_ROOT}/image_dirs}"
CKPT="${CKPT:-${DATA_ROOT}/sam_vit_h_4b8939.pth}"
N="${N:-300}"
DOMAINS="${DOMAINS:-kvasir pcb}"
FRACS="${FRACS:-0.125 0.25 0.375}"
SEEDS="${SEEDS:-0 1 2}"
PROFILE_DIR="${PROFILE_DIR:-${OUT_ROOT}/sam_main_sweep/profile}"

export PYTHONPATH=.
OUT="${OUT_ROOT}/sam_random_baseline"
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

run_freeze_eval() {
    # run_freeze_eval <domain> <profile_dir> <frac> <method> <seed> <tag>
    local D="$1" PROF="$2" F="$3" METHOD="$4" SEED="$5" TAG="$6"
    if [ -f "$OUT/eval_${TAG}/$D/eval_results.json" ]; then
        log "SKIP $TAG (results exist)"
        return 0
    fi
    log "--- $D @ f=$F method=$METHOD seed=$SEED ---"
    python3 sam/sam_velcro.py --mode freeze \
        --profile_dir "$PROF" \
        --profile_group "$D" \
        --freeze_fraction "$F" \
        --freeze_method "$METHOD" \
        --seed "$SEED" \
        --out_dir "$OUT/freeze_${TAG}" \
        2>&1 | tee "$LOG_DIR/freeze_${TAG}.log"

    python3 sam/sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --data_root "$DATA_ROOT" \
        --freeze_manifest "$OUT/freeze_${TAG}/freeze_manifest.json" \
        --eval_groups "$D" \
        --max_images "$N" \
        --diagnostic_level 2 \
        --out_dir "$OUT/eval_${TAG}" \
        2>&1 | tee "$LOG_DIR/eval_${TAG}.log"
}

for D in $DOMAINS; do
    if [ ! -d "$IMG_DIR/${D}_eval" ]; then
        log "SKIP $D (no image dir at $IMG_DIR/${D}_eval)"
        continue
    fi

    # Resolve profile: reuse main sweep's if present, else profile fresh here
    PROF="$PROFILE_DIR"
    if [ ! -f "$PROF/$D/block0_head0_variance.npy" ]; then
        PROF="$OUT/profile"
        if [ ! -f "$PROF/$D/block0_head0_variance.npy" ]; then
            log "Profiling $D (no existing profile found, N=$N)..."
            python3 sam/sam_velcro.py --mode profile \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --domain_groups "$D" \
                --max_images "$N" \
                --out_dir "$PROF" \
                2>&1 | tee "$LOG_DIR/profile_${D}.log"
        fi
    fi

    for F in $FRACS; do
        # Random ranking: one run per seed
        for S in $SEEDS; do
            run_freeze_eval "$D" "$PROF" "$F" random "$S" "${D}_random_s${S}_f${F}"
        done
        # Entropy ranking: deterministic, single run
        run_freeze_eval "$D" "$PROF" "$F" entropy 0 "${D}_entropy_f${F}"
    done
done

# --- Summary -----------------------------------------------------------------
log "Summary:"
python3 - "$OUT" "$DOMAINS" "$FRACS" "$SEEDS" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
domains = sys.argv[2].split()
fracs = sys.argv[3].split()
seeds = sys.argv[4].split()

print(f"{'Domain':<10} {'Frac':>6} {'Method':<10} {'Seed':>4} {'Mean IoU':>9} {'Miss%':>7} {'FP%':>7}")
rows = []
for d in domains:
    for f in fracs:
        tags = [(f"{d}_random_s{s}_f{f}", "random", s) for s in seeds]
        tags.append((f"{d}_entropy_f{f}", "entropy", "-"))
        for tag, method, seed in tags:
            p = out / f"eval_{tag}" / d / "eval_results.json"
            if not p.exists():
                continue
            r = json.load(open(p))
            print(f"{d:<10} {f:>6} {method:<10} {seed:>4} "
                  f"{r.get('per_image_mean_iou', 0):>9.4f} "
                  f"{r.get('per_image_miss_rate', 0)*100:>6.1f} "
                  f"{r.get('per_image_false_pos_rate', 0)*100:>6.1f}")
            rows.append({"domain": d, "fraction": float(f), "method": method,
                         "seed": seed, **{k: r.get(k) for k in (
                             "per_image_mean_iou", "per_image_miss_rate",
                             "per_image_false_pos_rate")}})
json.dump(rows, open(out / "random_baseline_summary.json", "w"), indent=2)
print(f"\nSaved: {out / 'random_baseline_summary.json'}")
print("Compare against variance-ranked rows in results/sam_main_sweep/")
PYEOF

log "Done. Results in $OUT/"
