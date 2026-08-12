#!/bin/bash
# exp_ratio_sweep.sh — Run ratio-based freeze with per-domain recommended thresholds
#
# Uses the output of analyze_freeze_params.py to select three operating points
# (conservative, recommended, aggressive) per domain instead of a blanket sweep.
#
# Usage:
#   bash scripts/exp_ratio_sweep.sh
#
#   # With custom recommendations file:
#   RECOMMENDATIONS=./path/to/freeze_recommendations.json bash scripts/exp_ratio_sweep.sh
#
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
OUT="${OUT:-./results_ratio_sweep}"
MAX_EVAL="${MAX_EVAL:-300}"
RECOMMENDATIONS="${RECOMMENDATIONS:-./results_freeze_recommendations.json}"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [ ! -f "$RECOMMENDATIONS" ]; then
    echo "ERROR: Recommendations file not found: $RECOMMENDATIONS"
    echo "  Run analyze_freeze_params.sh first, or set RECOMMENDATIONS= to point to the JSON."
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Extract per-domain thresholds from recommendations JSON
# ---------------------------------------------------------------------------
log "Reading per-domain thresholds from: $RECOMMENDATIONS"

DOMAIN_CONFIG=$(python3 - "$RECOMMENDATIONS" <<'PYEOF'
import json, sys

rec_path = sys.argv[1]
with open(rec_path) as f:
    rec = json.load(f)

# Output format: domain|profile_dir_hint|conservative_thresh|recommended_thresh|aggressive_thresh
for d in rec["domains"]:
    domain = d["domain"]
    ops = d["operating_points"]
    c = ops["conservative"]["ratio_threshold"]
    r = ops["recommended"]["ratio_threshold"]
    a = ops["aggressive"]["ratio_threshold"]
    print(f"{domain}|{c:.4f}|{r:.4f}|{a:.4f}")
PYEOF
)

if [ -z "$DOMAIN_CONFIG" ]; then
    log "ERROR: No domains found in recommendations file."
    exit 1
fi

log "Domain configurations:"
echo "$DOMAIN_CONFIG" | while IFS='|' read -r domain c_thresh r_thresh a_thresh; do
    log "  $domain: conservative=$c_thresh  recommended=$r_thresh  aggressive=$a_thresh"
done
echo ""

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
echo "$DOMAIN_CONFIG" | while IFS='|' read -r GROUP C_THRESH R_THRESH A_THRESH; do

    # Resolve PROFILE_DIR and IMG_DIR based on domain
    case "$GROUP" in
        kvasir|isic)
            PROFILE_DIR=./results_cross_domain/profile
            IMG_DIR=./image_dirs_cross_domain
            ;;
        busi|pcb|deepglobe)
            PROFILE_DIR=./results_wave2/profile
            IMG_DIR=./image_dirs_wave2
            ;;
        *)
            log "WARNING: Unknown domain $GROUP — trying default paths"
            PROFILE_DIR=./results_cross_domain/profile
            IMG_DIR=./image_dirs_cross_domain
            ;;
    esac

    # Skip if no image dir
    if [ ! -d "$IMG_DIR/${GROUP}_eval" ]; then
        log "SKIP $GROUP — no image dir at $IMG_DIR/${GROUP}_eval"
        continue
    fi

    for LEVEL in conservative recommended aggressive; do
        case "$LEVEL" in
            conservative) THRESH="$C_THRESH" ;;
            recommended)  THRESH="$R_THRESH" ;;
            aggressive)   THRESH="$A_THRESH" ;;
        esac

        FREEZE_TAG="freeze_${GROUP}_${LEVEL}"
        EVAL_TAG="eval_${GROUP}_${LEVEL}"

        # Skip if eval results already exist
        if [ -f "$OUT/$EVAL_TAG/$GROUP/eval_results.json" ]; then
            log "SKIP $GROUP $LEVEL (ratio=$THRESH) — eval results already exist"
            continue
        fi

        log "START $GROUP $LEVEL (ratio_threshold=$THRESH)"

        # Freeze with ratio method
        log "  Freezing ..."
        python sam_velcro.py --mode freeze \
            --freeze_method ratio \
            --ratio_threshold "$THRESH" \
            --profile_dir "$PROFILE_DIR" \
            --profile_group "$GROUP" \
            --general_group general \
            --out_dir "$OUT/$FREEZE_TAG"

        # Eval
        log "  Evaluating ..."
        python sam_velcro.py --mode eval \
            --checkpoint "$CKPT" \
            --image_dir "$IMG_DIR" \
            --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
            --eval_groups "$GROUP" \
            --max_images "$MAX_EVAL" \
            --points_per_side 16 \
            --diagnostic_level 0 \
            --out_dir "$OUT/$EVAL_TAG"

        log "DONE $GROUP $LEVEL (ratio=$THRESH)"
    done
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
log "Generating summary table ..."

python3 - "$OUT" "$RECOMMENDATIONS" <<'PYEOF'
import json, sys, os

out_dir = sys.argv[1]
rec_path = sys.argv[2]

with open(rec_path) as f:
    rec = json.load(f)

domains = [d["domain"] for d in rec["domains"]]
levels = ["conservative", "recommended", "aggressive"]

header = f"{'Domain':<12} {'Level':<14} {'Threshold':<10} {'Frozen':<8} {'Attn%':<8} {'Enc%':<8} {'Mean_IoU':<10} {'Miss%':<8} {'FP%':<8}"
sep = "-" * len(header)

print("\n" + sep)
print("RATIO-BASED FREEZE — PER-DOMAIN OPTIMAL THRESHOLDS")
print(sep)
print(header)
print(sep)

for d_info in rec["domains"]:
    group = d_info["domain"]
    for level in levels:
        thresh = d_info["operating_points"][level]["ratio_threshold"]

        freeze_tag = f"freeze_{group}_{level}"
        eval_tag = f"eval_{group}_{level}"

        eval_path = os.path.join(out_dir, eval_tag, group, "eval_results.json")
        manifest_path = os.path.join(out_dir, freeze_tag, "freeze_manifest.json")

        # From manifest
        num_frozen = "—"
        attn_pct = "—"
        enc_pct = "—"
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                m = json.load(f)
            num_frozen = str(m.get("summary", {}).get("frozen", "?"))
            attn_pct = f"{m.get('flops', {}).get('attention_flop_reduction_pct', 0):.1f}"
            enc_pct = f"{m.get('flops', {}).get('encoder_flop_reduction_pct', 0):.1f}"

        # From eval
        mean_iou = "—"
        miss_pct = "—"
        fp_pct = "—"
        if os.path.isfile(eval_path):
            with open(eval_path) as f:
                r = json.load(f)
            iou_val = r.get("per_image_mean_iou", r.get("mean_iou", r.get("iou_mean", None)))
            miss_val = r.get("per_image_miss_rate", r.get("miss_pct", r.get("miss_rate", None)))
            fp_val = r.get("per_image_false_pos_rate", r.get("fp_pct", r.get("false_positive_rate", None)))
            if iou_val is not None:
                mean_iou = f"{iou_val:.4f}"
            if miss_val is not None:
                miss_pct = f"{miss_val * 100 if miss_val < 1 else miss_val:.1f}"
            if fp_val is not None:
                fp_pct = f"{fp_val * 100 if fp_val < 1 else fp_val:.1f}"

        print(f"{group:<12} {level:<14} {thresh:<10.4f} {num_frozen:<8} {attn_pct:<8} {enc_pct:<8} {mean_iou:<10} {miss_pct:<8} {fp_pct:<8}")
    print()

print(sep)
print()

# Identify best level per domain (highest IoU)
print("BEST LEVEL PER DOMAIN:")
print("-" * 60)
for d_info in rec["domains"]:
    group = d_info["domain"]
    best_level = None
    best_iou = -1
    for level in levels:
        eval_path = os.path.join(out_dir, f"eval_{group}_{level}", group, "eval_results.json")
        if os.path.isfile(eval_path):
            with open(eval_path) as f:
                r = json.load(f)
            iou_val = r.get("per_image_mean_iou", r.get("mean_iou", 0))
            if iou_val > best_iou:
                best_iou = iou_val
                best_level = level
    if best_level:
        thresh = d_info["operating_points"][best_level]["ratio_threshold"]
        heads = d_info["operating_points"][best_level]["heads_frozen"]
        print(f"  {group:<12} → {best_level} (ratio={thresh:.4f}, {heads} heads, IoU={best_iou:.4f})")
    else:
        print(f"  {group:<12} → no eval results yet")

print()
PYEOF

log "All done."
