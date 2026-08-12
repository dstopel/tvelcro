#!/bin/bash
# =============================================================================
# exp_general_vs_domain.sh - Domain-specialization ablation
#
# Freezes heads ranked by the COCO 'general' profile (DOMAIN_GROUPS general =
# all val2017), then evaluates on the 5 specialized domains at the same
# fractions as the main sweep. Compare against exp_main_sweep.sh results to
# quantify how much domain-specific profiling buys.
#
# Run from repo root:
#   bash experiments/sam/exp_general_vs_domain.sh
#   # Reuse an existing general profile (dir must contain general/ subdir):
#   GENERAL_PROFILE_DIR=./results_sam_velcro/profile bash experiments/sam/exp_general_vs_domain.sh
#
# Expects $IMG_DIR/<domain>_eval/ image dirs; if no GENERAL_PROFILE_DIR is
# given, COCO val2017 + annotations must be under $DATA_ROOT
# (sam/download_data.sh fetches them).
#
# ETA (A10 24GB, N=300): general profile ~15 min (skipped if reused);
# eval ~20-30 min per (domain, fraction) cell. 5 domains x 3 fractions:
# ~5-8 hours total.
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
GENERAL_PROFILE_DIR="${GENERAL_PROFILE_DIR:-}"

export PYTHONPATH=.
OUT="${OUT_ROOT}/sam_general_vs_domain"
LOG_DIR="${OUT}/logs"

if [ ! -f sam/sam_velcro.py ]; then
    echo "ERROR: run from repo root (sam/sam_velcro.py not found)" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: SAM checkpoint not found at $CKPT (run sam/download_data.sh)" >&2
    exit 1
fi

mkdir -p "$OUT/profile" "$LOG_DIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Phase 1: General COCO profile (symlink existing, or profile fresh) ------
if [ ! -f "$OUT/profile/general/block0_head0_variance.npy" ]; then
    if [ -n "$GENERAL_PROFILE_DIR" ] && [ -d "$GENERAL_PROFILE_DIR/general" ]; then
        ln -sfn "$(cd "$GENERAL_PROFILE_DIR/general" && pwd)" "$OUT/profile/general"
        log "Symlinked existing general profile: $GENERAL_PROFILE_DIR/general"
    else
        if [ ! -d "$DATA_ROOT/val2017" ]; then
            echo "ERROR: no general profile and no COCO val2017 at $DATA_ROOT/val2017" >&2
            echo "  Either set GENERAL_PROFILE_DIR or run sam/download_data.sh" >&2
            exit 1
        fi
        log "Profiling 'general' from COCO val2017 (N=$N)..."
        python3 sam/sam_velcro.py --mode profile \
            --checkpoint "$CKPT" \
            --coco_dir "$DATA_ROOT" \
            --domain_groups general \
            --max_images "$N" \
            --out_dir "$OUT/profile" \
            2>&1 | tee "$LOG_DIR/profile_general.log"
    fi
else
    log "General profile exists: $OUT/profile/general"
fi

# --- Phase 2: Freeze from general profile, eval on each specialized domain ---
for F in $FRACS; do
    FREEZE_TAG="freeze_general_f${F}"
    if [ ! -f "$OUT/$FREEZE_TAG/freeze_manifest.json" ]; then
        log "Freezing from general profile @ f=$F..."
        python3 sam/sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group general \
            --freeze_fraction "$F" \
            --seed "$SEED" \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee "$LOG_DIR/${FREEZE_TAG}.log"
    fi

    for D in $DOMAINS; do
        if [ ! -d "$IMG_DIR/${D}_eval" ]; then
            log "SKIP eval $D @ f=$F (no image dir at $IMG_DIR/${D}_eval)"
            continue
        fi
        EVAL_TAG="eval_general_on_${D}_f${F}"
        if [ -f "$OUT/$EVAL_TAG/$D/eval_results.json" ]; then
            log "SKIP $EVAL_TAG (results exist)"
            continue
        fi
        log "--- general profile on $D @ f=$F ---"
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

# --- Phase 3: Summary table --------------------------------------------------
log "Summary (compare with results/sam_main_sweep eval_<domain>_f<frac> rows):"
python3 - "$OUT" "$DOMAINS" "$FRACS" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
domains = sys.argv[2].split()
fracs = sys.argv[3].split()

print(f"{'Domain':<12} {'Frac':>6} {'Mean IoU':>9} {'Miss%':>7} {'FP%':>7}")
rows = []
for d in domains:
    for f in fracs:
        p = out / f"eval_general_on_{d}_f{f}" / d / "eval_results.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        print(f"{d:<12} {f:>6} {r.get('per_image_mean_iou', 0):>9.4f} "
              f"{r.get('per_image_miss_rate', 0)*100:>6.1f} "
              f"{r.get('per_image_false_pos_rate', 0)*100:>6.1f}")
        rows.append({"domain": d, "fraction": float(f), "profile": "general", **{
            k: r.get(k) for k in ("per_image_mean_iou", "per_image_miss_rate",
                                  "per_image_false_pos_rate", "mean_feature_cosine_sim")}})
json.dump(rows, open(out / "general_vs_domain_summary.json", "w"), indent=2)
print(f"\nSaved: {out / 'general_vs_domain_summary.json'}")
PYEOF

log "Done. Results in $OUT/"
