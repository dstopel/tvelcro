#!/bin/bash
# =============================================================================
# exp_cross_domain_resume_isic.sh — Resume from isic f=0.125 to end
# Lighter eval settings (diagnostic_level 0, no visualizations) to avoid OOM.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

CKPT="./data/sam_vit_h_4b8939.pth"
IMG_DIR="./image_dirs_cross_domain"
OUT="./results_cross_domain"
LOG_DIR="${OUT}/logs"
MAX_EVAL=300

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/resume_isic_${TIMESTAMP}.log"

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

# ── Phase 2: Freeze + Eval (remaining cells) ─────────────────────────────────

log_section "PHASE 2: Freeze + Eval (resume from isic f=0.125)"

# --- isic @ f=0.125 (domain-specific profile) ---

FREEZE_TAG="freeze_isic_f0.125"
EVAL_TAG="eval_isic_f0.125"

if [ -f "$OUT/$EVAL_TAG/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG (results exist)"
else
    log "--- isic @ f=0.125 (domain-specific profile) ---"

    if [ ! -f "$OUT/$FREEZE_TAG/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group isic \
            --freeze_fraction 0.125 \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG}.log"

    log "  Done: $EVAL_TAG"
fi

# --- isic @ f=0.125 (general profile → cross-eval) ---

FREEZE_TAG_GEN="freeze_general_on_isic_f0.125"
EVAL_TAG_GEN="eval_general_on_isic_f0.125"

if [ -f "$OUT/$EVAL_TAG_GEN/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG_GEN (results exist)"
else
    log "--- isic @ f=0.125 (general profile → cross-eval) ---"

    if [ ! -f "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group general \
            --freeze_fraction 0.125 \
            --out_dir "$OUT/$FREEZE_TAG_GEN" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG_GEN}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG_GEN" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

    log "  Done: $EVAL_TAG_GEN"
fi

# --- isic @ f=0.25 (domain-specific profile) ---

FREEZE_TAG="freeze_isic_f0.25"
EVAL_TAG="eval_isic_f0.25"

if [ -f "$OUT/$EVAL_TAG/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG (results exist)"
else
    log "--- isic @ f=0.25 (domain-specific profile) ---"

    if [ ! -f "$OUT/$FREEZE_TAG/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group isic \
            --freeze_fraction 0.25 \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG}.log"

    log "  Done: $EVAL_TAG"
fi

# --- isic @ f=0.25 (general profile → cross-eval) ---

FREEZE_TAG_GEN="freeze_general_on_isic_f0.25"
EVAL_TAG_GEN="eval_general_on_isic_f0.25"

if [ -f "$OUT/$EVAL_TAG_GEN/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG_GEN (results exist)"
else
    log "--- isic @ f=0.25 (general profile → cross-eval) ---"

    if [ ! -f "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group general \
            --freeze_fraction 0.25 \
            --out_dir "$OUT/$FREEZE_TAG_GEN" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG_GEN}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG_GEN" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

    log "  Done: $EVAL_TAG_GEN"
fi

# --- isic @ f=0.375 (domain-specific profile) ---

FREEZE_TAG="freeze_isic_f0.375"
EVAL_TAG="eval_isic_f0.375"

if [ -f "$OUT/$EVAL_TAG/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG (results exist)"
else
    log "--- isic @ f=0.375 (domain-specific profile) ---"

    if [ ! -f "$OUT/$FREEZE_TAG/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group isic \
            --freeze_fraction 0.375 \
            --out_dir "$OUT/$FREEZE_TAG" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG}.log"

    log "  Done: $EVAL_TAG"
fi

# --- isic @ f=0.375 (general profile → cross-eval) ---

FREEZE_TAG_GEN="freeze_general_on_isic_f0.375"
EVAL_TAG_GEN="eval_general_on_isic_f0.375"

if [ -f "$OUT/$EVAL_TAG_GEN/eval_results.json" ]; then
    log "SKIP: $EVAL_TAG_GEN (results exist)"
else
    log "--- isic @ f=0.375 (general profile → cross-eval) ---"

    if [ ! -f "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" ]; then
        python sam_velcro.py --mode freeze \
            --profile_dir "$OUT/profile" \
            --profile_group general \
            --freeze_fraction 0.375 \
            --out_dir "$OUT/$FREEZE_TAG_GEN" \
            2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG_GEN}.log"
    fi

    python sam_velcro.py --mode eval \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --freeze_manifest "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" \
        --eval_groups isic \
        --max_images "$MAX_EVAL" \
        --diagnostic_level 0 \
        --out_dir "$OUT/$EVAL_TAG_GEN" \
        2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

    log "  Done: $EVAL_TAG_GEN"
fi

# ── Phase 3: Plots + summary ─────────────────────────────────────────────────

log_section "PHASE 3: Generate comparison plots + summary"

log "Generating sweep plots..."
python plot_sweep_results.py \
    --results_dir "$OUT" \
    --groups "kvasir,isic" \
    --fractions "0.125,0.25,0.375" \
    2>&1 | tee -a "${LOG_DIR}/plot_sweep.log"

log ""
log "Generating cross-domain comparison summary..."

python -c "
import json, os, sys
from pathlib import Path

out_dir = Path('$OUT')
groups = ['kvasir', 'isic']
fractions = [0.125, 0.25, 0.375]

print()
print('=' * 110)
print('CROSS-DOMAIN LOCALITY EXPERIMENT — SUMMARY')
print('=' * 110)
print()

hdr = f'{\"Domain\":<15} {\"Frac\":>5} {\"Profile\":>12} {\"Mean IoU\":>10} {\"Miss%\":>8} {\"FP%\":>8} {\"Feat Sim\":>10} {\"Frozen\":>8} {\"Attn%\":>8}'
print(hdr)
print('-' * 110)

rows = []
for group in groups:
    for frac in fractions:
        for profile_type, tag_pattern in [('domain', 'eval_{g}_f{f}'), ('general', 'eval_general_on_{g}_f{f}')]:
            tag = tag_pattern.format(g=group, f=frac)
            results_path = out_dir / tag / 'eval_results.json'
            manifest_pattern = 'freeze_{g}_f{f}' if profile_type == 'domain' else 'freeze_general_on_{g}_f{f}'
            manifest_path = out_dir / manifest_pattern.format(g=group, f=frac) / 'freeze_manifest.json'

            iou = miss = fp = feat_sim = frozen = attn_pct = 'N/A'
            if results_path.exists():
                r = json.load(open(results_path))
                iou = f\"{r.get('per_image_mean_iou', r.get('overall_mean_iou', 0)):.4f}\"
                miss = f\"{r.get('per_image_miss_rate', 0)*100:.1f}\"
                fp = f\"{r.get('per_image_false_pos_rate', 0)*100:.1f}\"
                feat_sim = f\"{r.get('mean_feature_cosine_sim', 0):.4f}\" if r.get('mean_feature_cosine_sim') else 'N/A'
            if manifest_path.exists():
                m = json.load(open(manifest_path))
                s = m.get('summary', {})
                frozen = str(s.get('frozen', '?'))
                fl = m.get('flops', {})
                attn_pct = f\"{fl.get('attention_flop_reduction_pct', 0):.1f}\"

            label = profile_type
            print(f'{group:<15} {frac:>5.3f} {label:>12} {iou:>10} {miss:>8} {fp:>8} {feat_sim:>10} {frozen:>8} {attn_pct:>8}')
            rows.append({
                'domain': group, 'fraction': frac, 'profile': profile_type,
                'mean_iou': iou, 'miss_rate': miss, 'fp_rate': fp,
                'feature_sim': feat_sim, 'frozen_heads': frozen, 'attn_flop_pct': attn_pct
            })
    print()

summary_path = out_dir / 'cross_domain_summary.json'
json.dump(rows, open(summary_path, 'w'), indent=2)
print(f'Summary saved: {summary_path}')

print()
print('KEY QUESTION: Does domain-specific profiling yield higher IoU')
print('than general profiling on specialized datasets?')
print('Compare \"domain\" vs \"general\" rows for each (domain, fraction) pair.')
print('=' * 110)
" 2>&1 | tee -a "$MAIN_LOG" | tee "$OUT/cross_domain_summary.txt"

# ── Done ──────────────────────────────────────────────────────────────────────

log_section "EXPERIMENT COMPLETE"
log "Results:   $OUT/"
log "Summary:   $OUT/cross_domain_summary.txt"
log "Full log:  $MAIN_LOG"
