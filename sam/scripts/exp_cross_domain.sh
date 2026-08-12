#!/bin/bash
# =============================================================================
# exp_cross_domain.sh — Cross-domain locality experiment for SAM ViT-H
#
# Thesis: narrower visual domains produce MORE value locality (more freezable
# attention heads) than general-purpose image collections.
#
# Downloads specialized datasets (medical, driving), profiles them, reuses the
# existing general COCO profile, then runs freeze+eval to compare:
#   1. How many heads are freezable per domain (variance analysis)
#   2. Whether domain-specific profiling yields better IoU than general profiling
#
# Datasets:
#   - Kvasir-SEG  (GI polyps — extremely narrow domain, 1000 images)
#   - ISIC 2018   (dermoscopy skin lesions — very narrow, 2594 images)
#   - Cityscapes  (urban driving — moderately narrow, requires manual download)
#
# Usage (on GPU pod — tmux recommended):
#   # Minimal (Kvasir + ISIC only, general profile at default path):
#   bash scripts/exp_cross_domain.sh
#
#   # Full (with Cityscapes, custom general profile path):
#   GENERAL_PROFILE_DIR=/path/to/results/profile \
#     bash scripts/exp_cross_domain.sh
#
#   # Quick test (fewer images):
#   MAX_PROFILE=50 MAX_EVAL=50 \
#     bash scripts/exp_cross_domain.sh
#
# Prerequisites:
#   - GPU with >= 6 GB VRAM (SAM ViT-H FP32 = ~2.5 GB)
#   - Python with: segment-anything, torch, numpy, scipy, matplotlib, cv2, PIL
#   - wget, unzip
#   - For Cityscapes: manually download leftImg8bit_val.zip from
#     https://www.cityscapes-dataset.com/downloads/ and place in DATA_DIR
#
# Expected runtime (A10 24GB):
#   Phase 0 (download):  ~5-15 min (depends on network, Kvasir=46MB, ISIC=1.2GB)
#   Phase 1 (profile):   ~10-15 min per domain (300 images each)
#   Phase 2 (freeze+eval): ~20-30 min per (domain, fraction) cell
#   Phase 3 (plots):     ~1 min
#   Total (2 domains × 3 fractions × 2 profiles): ~3-5 hours
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_DIR="${DATA_DIR:-./data}"
IMG_DIR="${IMG_DIR:-./image_dirs_cross_domain}"
OUT="${OUT:-./results_cross_domain}"
LOG_DIR="${OUT}/logs"

# Path to existing general COCO profile (will NOT re-profile general)
# On the university VM this lives at: sam-velcro/results_sam_velcro/profile
GENERAL_PROFILE_DIR="${GENERAL_PROFILE_DIR:-./results_sam_velcro/profile}"

MAX_PROFILE="${MAX_PROFILE:-300}"
MAX_EVAL="${MAX_EVAL:-300}"
SEED="${SEED:-42}"

mkdir -p "$DATA_DIR" "$IMG_DIR" "$OUT" "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/cross_domain_${TIMESTAMP}.log"

# ── Logging ───────────────────────────────────────────────────────────────────

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

# ── Phase 0: Download datasets ───────────────────────────────────────────────

log_section "PHASE 0: Download datasets + organize image directories"

# 0.1 SAM checkpoint
if [ ! -f "$CKPT" ]; then
    log "Downloading SAM ViT-H checkpoint..."
    if ! wget --no-check-certificate -O "$CKPT" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth; then
        log "ERROR: Failed to download SAM checkpoint. Try manually:"
        log "  wget -O $CKPT https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        exit 1
    fi
    log "  Done: $CKPT"
else
    log "SAM checkpoint: $CKPT (exists)"
fi

# 0.2 Kvasir-SEG (GI polyp segmentation — 46 MB, 1000 images)
KVASIR_RAW="${DATA_DIR}/kvasir-seg"
if [ ! -d "$KVASIR_RAW/images" ]; then
    log "Downloading Kvasir-SEG (46 MB)..."
    if ! wget --no-check-certificate -O "${DATA_DIR}/kvasir-seg.zip" \
        "https://datasets.simula.no/downloads/kvasir-seg.zip"; then
        log "ERROR: Failed to download Kvasir-SEG. Try manually:"
        log "  wget -O ${DATA_DIR}/kvasir-seg.zip https://datasets.simula.no/downloads/kvasir-seg.zip"
        exit 1
    fi
    unzip -q "${DATA_DIR}/kvasir-seg.zip" -d "$DATA_DIR"
    rm -f "${DATA_DIR}/kvasir-seg.zip"
    # Handle nested directory (zip extracts to Kvasir-SEG/ or kvasir-seg/)
    for candidate in "$DATA_DIR/Kvasir-SEG" "$DATA_DIR/kvasir-seg"; do
        if [ -d "$candidate/images" ]; then
            if [ "$candidate" != "$KVASIR_RAW" ]; then
                mv "$candidate" "$KVASIR_RAW"
            fi
            break
        fi
    done
    if [ ! -d "$KVASIR_RAW/images" ]; then
        log "ERROR: Kvasir-SEG extracted but images/ not found. Check $DATA_DIR for extracted contents."
        ls "$DATA_DIR" | tee -a "$MAIN_LOG"
        exit 1
    fi
    log "  Extracted: $(ls "$KVASIR_RAW/images" | wc -l) images"
else
    log "Kvasir-SEG: $KVASIR_RAW/images ($(ls "$KVASIR_RAW/images" | wc -l) images, exists)"
fi

# 0.3 ISIC 2018 (dermoscopy skin lesions — ~1.2 GB, 2594 images)
ISIC_RAW="${DATA_DIR}/ISIC2018"
if [ ! -d "$ISIC_RAW" ] || [ "$(find "$ISIC_RAW" -name '*.jpg' -o -name '*.png' 2>/dev/null | head -1)" = "" ]; then
    log "Downloading ISIC 2018 Training Input (~1.2 GB)..."
    if ! wget --no-check-certificate -O "${DATA_DIR}/ISIC2018_Task1-2_Training_Input.zip" \
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"; then
        log "ERROR: Failed to download ISIC 2018. Try manually:"
        log "  wget -O ${DATA_DIR}/ISIC2018_Task1-2_Training_Input.zip https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"
        exit 1
    fi
    unzip -q "${DATA_DIR}/ISIC2018_Task1-2_Training_Input.zip" -d "$DATA_DIR"
    rm -f "${DATA_DIR}/ISIC2018_Task1-2_Training_Input.zip"
    # The zip extracts to ISIC2018_Task1-2_Training_Input/ — rename for simplicity
    if [ -d "${DATA_DIR}/ISIC2018_Task1-2_Training_Input" ] && [ ! -d "$ISIC_RAW" ]; then
        mv "${DATA_DIR}/ISIC2018_Task1-2_Training_Input" "$ISIC_RAW"
    fi
    log "  Extracted: $(find "$ISIC_RAW" -name '*.jpg' -o -name '*.png' | wc -l) images"
else
    log "ISIC 2018: $ISIC_RAW ($(find "$ISIC_RAW" -name '*.jpg' -o -name '*.png' | wc -l) images, exists)"
fi

# 0.4 Cityscapes (requires manual download — check if present)
CITY_RAW="${DATA_DIR}/cityscapes"
CITY_AVAILABLE=0
if [ -d "$CITY_RAW/leftImg8bit/val" ]; then
    CITY_COUNT="$(find "$CITY_RAW/leftImg8bit/val" -name '*.png' | wc -l)"
    if [ "$CITY_COUNT" -gt 0 ]; then
        CITY_AVAILABLE=1
        log "Cityscapes: $CITY_RAW ($CITY_COUNT images, found)"
    fi
fi
if [ "$CITY_AVAILABLE" -eq 0 ]; then
    log "WARNING: Cityscapes not found at $CITY_RAW/leftImg8bit/val/"
    log "  To include Cityscapes:"
    log "    1. Register at https://www.cityscapes-dataset.com/"
    log "    2. Download leftImg8bit_val.zip"
    log "    3. Extract to $CITY_RAW/"
    log "  Continuing without Cityscapes..."
fi

# 0.5 Build image directory splits (profile/eval, 300 each, deterministic)
log ""
log "Setting up image directory splits (profile=$MAX_PROFILE, eval=$MAX_EVAL, seed=$SEED)..."

setup_split() {
    local src_dir="$1"
    local group_name="$2"
    local n_profile="$3"
    local n_eval="$4"

    local prof_dir="${IMG_DIR}/${group_name}_profile"
    local eval_dir="${IMG_DIR}/${group_name}_eval"

    # Check if splits already have real (non-broken) images
    if [ -d "$prof_dir" ] && [ -d "$eval_dir" ]; then
        local real_count="$(find "$prof_dir" -maxdepth 2 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) 2>/dev/null | wc -l)"
        if [ "$real_count" -gt 0 ]; then
            log "  $group_name splits: already exist (profile=$real_count images)"
            return
        fi
        # Broken symlinks or empty — wipe and re-create
        rm -rf "$prof_dir" "$eval_dir"
    fi

    mkdir -p "$prof_dir" "$eval_dir"

    # Collect all images, sort deterministically, then split
    local all_images
    all_images="$(find "$src_dir" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) | sort)"
    local total
    total="$(echo "$all_images" | wc -l)"

    # Deterministic shuffle using sort with hash-based key
    local shuffled
    shuffled="$(echo "$all_images" | awk -v seed="$SEED" '{print seed"_"NR"_"$0, $0}' | sort -t' ' -k1,1 | awk '{print $2}')"

    # Use hard links (same filesystem) with cp fallback
    local count=0
    while IFS= read -r img_path; do
        count=$((count + 1))
        if [ "$count" -le "$n_profile" ]; then
            ln "$img_path" "$prof_dir/$(basename "$img_path")" 2>/dev/null \
                || cp "$img_path" "$prof_dir/$(basename "$img_path")"
        elif [ "$count" -le $((n_profile + n_eval)) ]; then
            ln "$img_path" "$eval_dir/$(basename "$img_path")" 2>/dev/null \
                || cp "$img_path" "$eval_dir/$(basename "$img_path")"
        else
            break
        fi
    done <<< "$shuffled"

    local p_count="$(find "$prof_dir" -maxdepth 1 -type f | wc -l)"
    local e_count="$(find "$eval_dir" -maxdepth 1 -type f | wc -l)"
    log "  $group_name: profile=$p_count, eval=$e_count (from $total total)"
}

# Kvasir split
setup_split "$KVASIR_RAW/images" "kvasir" "$MAX_PROFILE" "$MAX_EVAL"

# ISIC split
setup_split "$ISIC_RAW" "isic" "$MAX_PROFILE" "$MAX_EVAL"

# Cityscapes split (if available)
if [ "$CITY_AVAILABLE" -eq 1 ]; then
    setup_split "$CITY_RAW/leftImg8bit/val" "cityscapes" "$MAX_PROFILE" "$MAX_EVAL"
fi

log ""
log "Specialized domains: kvasir isic cityscapes"

# 0.6 Verify general profile exists
if [ ! -d "$GENERAL_PROFILE_DIR/general" ]; then
    log "ERROR: General COCO profile not found at $GENERAL_PROFILE_DIR/general/"
    log "  Expected: 1024 .npy files (512 mean + 512 variance)"
    log "  Set GENERAL_PROFILE_DIR to the directory containing the general/ subdirectory"
    exit 1
fi
GENERAL_NPY_COUNT="$(find "$GENERAL_PROFILE_DIR/general" -name '*.npy' | wc -l)"
log "General COCO profile: $GENERAL_PROFILE_DIR/general/ ($GENERAL_NPY_COUNT .npy files)"

# Symlink general profile into our output directory
mkdir -p "$OUT/profile"
if [ ! -e "$OUT/profile/general" ]; then
    ln -sf "$(cd "$GENERAL_PROFILE_DIR/general" && pwd)" "$OUT/profile/general"
    log "  Symlinked general profile into $OUT/profile/general"
fi

# ── Phase 1: Profile specialized domains ─────────────────────────────────────

log_section "PHASE 1: Profile specialized domains ($MAX_PROFILE images each)"

for GROUP in kvasir isic cityscapes; do
    # Skip groups whose image dirs don't exist (e.g. cityscapes not downloaded)
    if [ ! -d "$IMG_DIR/${GROUP}_profile" ]; then
        log "SKIP: $GROUP (no image dir at $IMG_DIR/${GROUP}_profile)"
        continue
    fi

    if [ -d "$OUT/profile/$GROUP" ] && [ "$(find "$OUT/profile/$GROUP" -name '*.npy' | wc -l)" -ge 1024 ]; then
        log "SKIP: $GROUP profile already complete ($(find "$OUT/profile/$GROUP" -name '*.npy' | wc -l) .npy files)"
        continue
    fi

    log "Profiling: $GROUP"
    T0=$(date +%s)

    python sam_velcro.py --mode profile \
        --checkpoint "$CKPT" \
        --image_dir "$IMG_DIR" \
        --domain_groups "$GROUP" \
        --max_images "$MAX_PROFILE" \
        --out_dir "$OUT/profile" \
        2>&1 | tee -a "${LOG_DIR}/profile_${GROUP}.log"

    ELAPSED=$(($(date +%s) - T0))
    log "  $GROUP profiled in ${ELAPSED}s"
done

# ── Phase 1.5: Profile analysis ──────────────────────────────────────────────

log_section "PHASE 1.5: Profile analysis (all domains including general)"

ALL_GROUPS="general,kvasir,isic,cityscapes"

log "Analyzing groups: $ALL_GROUPS"

python analyze_profile.py \
    --profile_dir "$OUT/profile" \
    --groups "$ALL_GROUPS" \
    --out "$OUT/plots/profile_analysis.txt" \
    2>&1 | tee -a "${LOG_DIR}/analyze_profile.log"

log "  Report: $OUT/plots/profile_analysis.txt"

# Variance distribution plots
log "Generating variance distribution plots..."
python plot_variance_distributions.py \
    --profile_dir "$OUT/profile" \
    --groups "$ALL_GROUPS" \
    --out_dir "$OUT/plots/variance" \
    2>&1 | tee -a "${LOG_DIR}/variance_plots.log"

log "  Plots: $OUT/plots/variance/"

# ── Phase 2: Freeze + Eval grid ──────────────────────────────────────────────

log_section "PHASE 2: Freeze + Eval (specialized profile vs general profile)"
log "Domains: kvasir isic cityscapes"
log "Fractions: 0.125 0.25 0.375"
log "Each domain gets TWO eval runs per fraction:"
log "  (a) freeze from domain-specific profile"
log "  (b) freeze from general COCO profile (cross-eval)"
log ""

for GROUP in kvasir isic cityscapes; do
    # Skip groups whose image dirs don't exist
    if [ ! -d "$IMG_DIR/${GROUP}_eval" ]; then
        log "SKIP: $GROUP (no eval image dir)"
        continue
    fi

    for FRAC in 0.125 0.25 0.375; do
        # ── (a) Domain-specific profile → freeze → eval ──────────────────

        FREEZE_TAG="freeze_${GROUP}_f${FRAC}"
        EVAL_TAG="eval_${GROUP}_f${FRAC}"

        if [ -f "$OUT/$EVAL_TAG/eval_results.json" ]; then
            log "SKIP: $EVAL_TAG (results exist)"
        else
            log "--- $GROUP @ f=$FRAC (domain-specific profile) ---"

            # Freeze
            python sam_velcro.py --mode freeze \
                --profile_dir "$OUT/profile" \
                --profile_group "$GROUP" \
                --freeze_fraction "$FRAC" \
                --out_dir "$OUT/$FREEZE_TAG" \
                2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG}.log"

            # Eval
            python sam_velcro.py --mode eval \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --freeze_manifest "$OUT/$FREEZE_TAG/freeze_manifest.json" \
                --eval_groups "$GROUP" \
                --max_images "$MAX_EVAL" \
                --save_visualizations \
                --diagnostic_level 2 \
                --out_dir "$OUT/$EVAL_TAG" \
                2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG}.log"

            log "  Done: $EVAL_TAG"
        fi

        # ── (b) General profile → freeze → eval on same domain ───────────

        FREEZE_TAG_GEN="freeze_general_on_${GROUP}_f${FRAC}"
        EVAL_TAG_GEN="eval_general_on_${GROUP}_f${FRAC}"

        if [ -f "$OUT/$EVAL_TAG_GEN/eval_results.json" ]; then
            log "SKIP: $EVAL_TAG_GEN (results exist)"
        else
            log "--- $GROUP @ f=$FRAC (general profile → cross-eval) ---"

            # Freeze from general profile
            python sam_velcro.py --mode freeze \
                --profile_dir "$OUT/profile" \
                --profile_group general \
                --freeze_fraction "$FRAC" \
                --out_dir "$OUT/$FREEZE_TAG_GEN" \
                2>&1 | tee -a "${LOG_DIR}/${FREEZE_TAG_GEN}.log"

            # Eval on specialized domain images
            python sam_velcro.py --mode eval \
                --checkpoint "$CKPT" \
                --image_dir "$IMG_DIR" \
                --freeze_manifest "$OUT/$FREEZE_TAG_GEN/freeze_manifest.json" \
                --eval_groups "$GROUP" \
                --max_images "$MAX_EVAL" \
                --save_visualizations \
                --diagnostic_level 2 \
                --out_dir "$OUT/$EVAL_TAG_GEN" \
                2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

            log "  Done: $EVAL_TAG_GEN"
        fi
    done
done

# ── Phase 3: Sweep plots + summary ───────────────────────────────────────────

log_section "PHASE 3: Generate comparison plots + summary"

# 3.1 Standard sweep plots (domain-specific profiles)
log "Generating sweep plots for domain-specific profiles..."
python plot_sweep_results.py \
    --results_dir "$OUT" \
    --groups "kvasir,isic,cityscapes" \
    --fractions "0.125,0.25,0.375" \
    2>&1 | tee -a "${LOG_DIR}/plot_sweep.log"

# 3.2 Cross-domain comparison summary table
log ""
log "Generating cross-domain comparison summary..."

python -c "
import json, os, sys
from pathlib import Path

out_dir = Path('$OUT')
groups = ['kvasir', 'isic', 'cityscapes']
fractions = [0.125, 0.25, 0.375]

print()
print('=' * 110)
print('CROSS-DOMAIN LOCALITY EXPERIMENT — SUMMARY')
print('=' * 110)
print()

# Header
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

# Save as JSON
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
log "Profiles:  $OUT/profile/"
log "Plots:     $OUT/plots/"
log "Summary:   $OUT/cross_domain_summary.txt"
log "Full log:  $MAIN_LOG"
log ""
log "Domains tested: kvasir isic cityscapes"
log "Fractions: 0.125 0.25 0.375"
log ""
log "Next steps:"
log "  1. Check $OUT/plots/profile_analysis.txt for Spearman correlations"
log "     (expect kvasir vs general < dogs vs general)"
log "  2. Check $OUT/cross_domain_summary.txt for IoU comparison"
log "     (expect domain-specific profile >= general profile IoU on narrow domains)"
log "  3. Check $OUT/plots/variance/ for distribution differences"
log "     (expect kvasir/isic to have tighter windowed-head distributions)"
