#!/bin/bash
# =============================================================================
# exp_wave2_resume.sh — Resume wave 2 eval with --points_per_side 16
# Fixes: OOM on PCB large images, stuck eval taking hours
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="./data/sam_vit_h_4b8939.pth"
IMG_DIR="./image_dirs_wave2"
OUT="./results_wave2"
LOG_DIR="${OUT}/logs"
MAX_EVAL=300

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/resume_${TIMESTAMP}.log"

log() {
    local msg="$(date +%H:%M:%S) | $1"
    echo "$msg" | tee -a "$MAIN_LOG"
}

log_section() {
    echo "" | tee -a "$MAIN_LOG"
    echo "==============================================================" | tee -a "$MAIN_LOG"
    log "$1"
    echo "==============================================================" | tee -a "$MAIN_LOG"
}

log_section "WAVE 2 RESUME: PCB + DeepGlobe (points_per_side=16)"

for GROUP in busi pcb deepglobe; do
    if [ ! -d "$IMG_DIR/${GROUP}_eval" ]; then
        log "SKIP: $GROUP (no eval image dir)"
        continue
    fi

    for FRAC in 0.125 0.25 0.375; do
        # ── (a) Domain-specific profile → freeze → eval ──────────────────

        FREEZE_TAG="freeze_${GROUP}_f${FRAC}"
        EVAL_TAG="eval_${GROUP}_f${FRAC}"

        if [ -f "$OUT/$EVAL_TAG/$GROUP/eval_results.json" ]; then
            log "SKIP: $EVAL_TAG (results exist)"
        else
            log "--- $GROUP @ f=$FRAC (domain-specific profile) ---"

            if [ ! -f "$OUT/$FREEZE_TAG/freeze_manifest.json" ]; then
                python sam_velcro.py --mode freeze \
                    --profile_dir "$OUT/profile" \
                    --profile_group "$GROUP" \
                    --freeze_fraction "$FRAC" \
                    --out_dir "$OUT/$FREEZE_TAG" \
                    2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG}.log"
            fi

            python sam_velcro.py --mode eval \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
                --eval_groups "$GROUP" \
                --max_images "$MAX_EVAL" \
                --points_per_side 16 \
                --diagnostic_level 0 \
                --out_dir "$OUT/$EVAL_TAG" \
                2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG}.log"

            log "  Done: $EVAL_TAG"
        fi

        # ── (b) General profile → freeze → eval on same domain ───────────

        FREEZE_TAG_GEN="freeze_general_on_${GROUP}_f${FRAC}"
        EVAL_TAG_GEN="eval_general_on_${GROUP}_f${FRAC}"

        if [ -f "$OUT/$EVAL_TAG_GEN/$GROUP/eval_results.json" ]; then
            log "SKIP: $EVAL_TAG_GEN (results exist)"
        else
            log "--- $GROUP @ f=$FRAC (general profile → cross-eval) ---"

            if [ ! -f "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" ]; then
                python sam_velcro.py --mode freeze \
                    --profile_dir "$OUT/profile" \
                    --profile_group general \
                    --freeze_fraction "$FRAC" \
                    --out_dir "$OUT/$FREEZE_TAG_GEN" \
                    2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG_GEN}.log"
            fi

            python sam_velcro.py --mode eval \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --freeze_manifest "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" \
                --eval_groups "$GROUP" \
                --max_images "$MAX_EVAL" \
                --points_per_side 16 \
                --diagnostic_level 0 \
                --out_dir "$OUT/$EVAL_TAG_GEN" \
                2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

            log "  Done: $EVAL_TAG_GEN"
        fi
    done
done

# ── Phase 3: Summary ─────────────────────────────────────────────────────────

log_section "PHASE 3: Summary"

python -c "
import json
from pathlib import Path

out_dir = Path('$OUT')
groups = ['busi', 'pcb', 'deepglobe']
fractions = [0.125, 0.25, 0.375]

print()
print('=' * 110)
print('WAVE 2 CROSS-DOMAIN EXPERIMENT — SUMMARY')
print('=' * 110)
print()
print(f'{\"Domain\":<15} {\"Frac\":>5} {\"Profile\":>12} {\"Mean IoU\":>10} {\"Miss%\":>8} {\"FP%\":>8} {\"Skipped\":>8} {\"Frozen\":>8} {\"Attn%\":>8}')
print('-' * 110)

rows = []
for group in groups:
    for frac in fractions:
        for profile_type, tag in [('domain', f'eval_{group}_f{frac}'), ('general', f'eval_general_on_{group}_f{frac}')]:
            results_path = out_dir / tag / group / 'eval_results.json'
            manifest_pattern = f'freeze_{group}_f{frac}' if profile_type == 'domain' else f'freeze_general_on_{group}_f{frac}'
            manifest_path = out_dir / manifest_pattern / 'freeze_manifest.json'

            iou = miss = fp = skipped = frozen = attn_pct = 'N/A'
            if results_path.exists():
                r = json.load(open(results_path))
                iou = f\"{r.get('per_image_mean_iou', r.get('overall_mean_iou', 0)):.4f}\"
                miss = f\"{r.get('per_image_miss_rate', 0)*100:.1f}\"
                fp = f\"{r.get('per_image_false_pos_rate', 0)*100:.1f}\"
                skipped = str(r.get('skipped_images', 0))
            if manifest_path.exists():
                m = json.load(open(manifest_path))
                frozen = str(m.get('summary', {}).get('frozen', '?'))
                attn_pct = f\"{m.get('flops', {}).get('attention_flop_reduction_pct', 0):.1f}\"

            print(f'{group:<15} {frac:>5.3f} {profile_type:>12} {iou:>10} {miss:>8} {fp:>8} {skipped:>8} {frozen:>8} {attn_pct:>8}')
            rows.append({
                'domain': group, 'fraction': frac, 'profile': profile_type,
                'mean_iou': iou, 'miss_rate': miss, 'fp_rate': fp,
                'skipped': skipped, 'frozen_heads': frozen, 'attn_flop_pct': attn_pct
            })
    print()

summary_path = out_dir / 'wave2_summary.json'
json.dump(rows, open(summary_path, 'w'), indent=2)
print(f'Summary saved: {summary_path}')
print('=' * 110)
" 2>&1 | tee -a "$MAIN_LOG" | tee "$OUT/wave2_summary.txt"

log_section "DONE"
log "Results: $OUT/"
log "Summary: $OUT/wave2_summary.txt"
