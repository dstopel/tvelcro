#!/bin/bash
# =============================================================================
# exp_cross_domain_wave2.sh — Wave 2: BUSI, PCB, DeepGlobe
#
# Thesis: narrower/more specialized visual domains produce MORE value locality.
# Wave 1 proved it with medical (Kvasir, ISIC). Wave 2 extends to:
#   - BUSI (breast ultrasound — grayscale, different imaging physics)
#   - PCB Defects (circuit board inspection — green boards, extreme repetition)
#   - DeepGlobe (satellite imagery — constrained overhead perspective)
#
# Reuses the existing general COCO profile (no re-profiling general).
#
# Usage:
#   bash scripts/exp_cross_domain_wave2.sh
#
# Prerequisites:
#   - GPU with >= 6 GB VRAM (SAM ViT-H FP32 = ~2.5 GB)
#   - Python with: segment-anything, torch, numpy, scipy, matplotlib, cv2, PIL
#   - wget, unzip, tar
#   - General COCO profile already exists at $GENERAL_PROFILE_DIR/general/
#   - For DeepGlobe: Kaggle CLI or manual download.
#
# Expected runtime (A10 24GB):
#   Phase 0 (download):     ~5-10 min (BUSI=260MB, PCB=100MB, DeepGlobe=~500MB)
#   Phase 1 (profile):      ~10-15 min per domain (300 images each)
#   Phase 2 (freeze+eval):  ~15-20 min per cell (diagnostic_level 0 to avoid OOM)
#   Total (3 domains × 3 fracs × 2 profiles): ~4-6 hours
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── Configuration ─────────────────────────────────────────────────────────────

CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_DIR="${DATA_DIR:-./data}"
IMG_DIR="${IMG_DIR:-./image_dirs_wave2}"
OUT="${OUT:-./results_wave2}"
LOG_DIR="${OUT}/logs"

# Path to existing general COCO profile
GENERAL_PROFILE_DIR="${GENERAL_PROFILE_DIR:-./results_sam_velcro/profile}"

MAX_PROFILE="${MAX_PROFILE:-300}"
MAX_EVAL="${MAX_EVAL:-300}"
SEED="${SEED:-42}"

mkdir -p "$DATA_DIR" "$IMG_DIR" "$OUT" "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/wave2_${TIMESTAMP}.log"

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
        log "ERROR: Failed to download SAM checkpoint."
        exit 1
    fi
    log "  Done: $CKPT"
else
    log "SAM checkpoint: $CKPT (exists)"
fi

# 0.2 BUSI — Breast Ultrasound Images (260 MB, 780 images)
# Source: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
# Structure after download: Dataset_BUSI_with_GT/{benign,malignant,normal}/*.png
BUSI_RAW="${DATA_DIR}/busi"
if [ ! -d "$BUSI_RAW" ] || [ "$(find "$BUSI_RAW" -name '*.png' 2>/dev/null | wc -l)" -lt 100 ]; then
    log "Downloading BUSI (Breast Ultrasound)..."
    # Try kaggle CLI first
    if command -v kaggle &>/dev/null; then
        kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset \
            -p "$DATA_DIR" --unzip 2>&1 | tee -a "$MAIN_LOG"
        # Move to standard location
        if [ -d "${DATA_DIR}/Dataset_BUSI_with_GT" ]; then
            mv "${DATA_DIR}/Dataset_BUSI_with_GT" "$BUSI_RAW"
        fi
    else
        log "  Kaggle CLI not found. Trying direct download..."
        # Alternative: direct URL (may require manual download)
        if ! wget --no-check-certificate -O "${DATA_DIR}/busi.zip" \
            "https://scholar.cu.edu.eg/Dataset_BUSI.zip" 2>/dev/null; then
            log "  WARNING: Auto-download failed for BUSI."
            log "  Manual download required:"
            log "    Option A: pip install kaggle && kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset"
            log "    Option B: Download from https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"
            log "    Then extract to: $BUSI_RAW/"
            log "  Continuing without BUSI..."
        else
            unzip -q "${DATA_DIR}/busi.zip" -d "$DATA_DIR"
            rm -f "${DATA_DIR}/busi.zip"
            # Find and rename the extracted directory
            for candidate in "${DATA_DIR}/Dataset_BUSI_with_GT" "${DATA_DIR}/BUSI" "${DATA_DIR}/busi"; do
                if [ -d "$candidate" ] && [ "$(find "$candidate" -name '*.png' | wc -l)" -gt 100 ]; then
                    if [ "$candidate" != "$BUSI_RAW" ]; then
                        mv "$candidate" "$BUSI_RAW"
                    fi
                    break
                fi
            done
        fi
    fi
    if [ -d "$BUSI_RAW" ]; then
        log "  BUSI: $(find "$BUSI_RAW" -name '*.png' | wc -l) images"
    fi
else
    log "BUSI: $BUSI_RAW ($(find "$BUSI_RAW" -name '*.png' | wc -l) images, exists)"
fi

# 0.3 PCB Defects — Circuit Board Inspection (~100 MB, ~1300 images)
# Source: https://www.kaggle.com/datasets/akhatova/pcb-defects
# Structure: PCB_DATASET/images/*.jpg
PCB_RAW="${DATA_DIR}/pcb"
if [ ! -d "$PCB_RAW" ] || [ "$(find "$PCB_RAW" -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)" -lt 100 ]; then
    log "Downloading PCB Defects (~100 MB)..."
    if command -v kaggle &>/dev/null; then
        kaggle datasets download -d akhatova/pcb-defects \
            -p "$DATA_DIR" --unzip 2>&1 | tee -a "$MAIN_LOG"
        # Find extracted directory
        for candidate in "${DATA_DIR}/PCB_DATASET" "${DATA_DIR}/pcb-defects" "${DATA_DIR}/pcb"; do
            if [ -d "$candidate" ] && [ "$(find "$candidate" -name '*.jpg' -o -name '*.png' | wc -l)" -gt 50 ]; then
                if [ "$candidate" != "$PCB_RAW" ]; then
                    mv "$candidate" "$PCB_RAW"
                fi
                break
            fi
        done
        # Also check if images landed directly in data/ subdirs
        if [ ! -d "$PCB_RAW" ] || [ "$(find "$PCB_RAW" -name '*.jpg' -o -name '*.png' 2>/dev/null | wc -l)" -lt 50 ]; then
            # Search for PCB images in any extracted subdir
            for candidate in $(find "$DATA_DIR" -maxdepth 2 -type d -name "*PCB*" -o -name "*pcb*" 2>/dev/null); do
                if [ "$(find "$candidate" -name '*.jpg' -o -name '*.png' | wc -l)" -gt 50 ]; then
                    mv "$candidate" "$PCB_RAW"
                    break
                fi
            done
        fi
    else
        log "  WARNING: Kaggle CLI not found for PCB Defects."
        log "  Manual download:"
        log "    kaggle datasets download -d akhatova/pcb-defects -p $DATA_DIR --unzip"
        log "    mv $DATA_DIR/PCB_DATASET $PCB_RAW"
        log "  Continuing without PCB..."
    fi
    if [ -d "$PCB_RAW" ]; then
        log "  PCB: $(find "$PCB_RAW" -name '*.jpg' -o -name '*.png' | wc -l) images"
    fi
else
    log "PCB: $PCB_RAW ($(find "$PCB_RAW" -name '*.jpg' -o -name '*.png' | wc -l) images, exists)"
fi

# 0.4 DeepGlobe Road — Satellite Road Extraction (~500 MB, 6226 tiles)
# Source: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
DEEPGLOBE_RAW="${DATA_DIR}/deepglobe"
if [ ! -d "$DEEPGLOBE_RAW" ] || [ "$(find "$DEEPGLOBE_RAW" -name '*.png' -o -name '*.jpg' 2>/dev/null | wc -l)" -lt 100 ]; then
    log "Downloading DeepGlobe Road..."
    if command -v kaggle &>/dev/null; then
        kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset \
            -p "$DATA_DIR" --unzip 2>&1 | tee -a "$MAIN_LOG"
        # Find extracted directory — Kaggle may extract to train/ test/ valid/ directly
        for candidate in "${DATA_DIR}/deepglobe-road-extraction-dataset" "${DATA_DIR}/DeepGlobe" "${DATA_DIR}/deepglobe"; do
            if [ -d "$candidate" ] && [ "$(find "$candidate" -name '*_sat.*' | wc -l)" -gt 100 ]; then
                if [ "$candidate" != "$DEEPGLOBE_RAW" ]; then
                    mv "$candidate" "$DEEPGLOBE_RAW"
                fi
                break
            fi
        done
        # If files landed in data/train/ directly (no parent dir)
        if [ ! -d "$DEEPGLOBE_RAW" ] || [ "$(find "$DEEPGLOBE_RAW" -name '*_sat.*' 2>/dev/null | wc -l)" -lt 100 ]; then
            if [ -d "${DATA_DIR}/train" ] && [ "$(find "${DATA_DIR}/train" -name '*_sat.*' | wc -l)" -gt 100 ]; then
                mkdir -p "$DEEPGLOBE_RAW"
                mv "${DATA_DIR}/train" "${DATA_DIR}/test" "${DATA_DIR}/valid" "$DEEPGLOBE_RAW/" 2>/dev/null
            fi
        fi
    else
        log "  WARNING: Kaggle CLI not found for DeepGlobe."
        log "  Manual download required:"
        log "    pip install kaggle && kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset"
        log "    Then extract to: $DEEPGLOBE_RAW/"
        log "  Continuing without DeepGlobe..."
    fi
    if [ -d "$DEEPGLOBE_RAW" ]; then
        log "  DeepGlobe: $(find "$DEEPGLOBE_RAW" -name '*.png' -o -name '*.jpg' | wc -l) images"
    fi
else
    log "DeepGlobe: $DEEPGLOBE_RAW ($(find "$DEEPGLOBE_RAW" -name '*.png' -o -name '*.jpg' | wc -l) images, exists)"
fi

# 0.5 Build image directory splits
log ""
log "Setting up image directory splits (profile=$MAX_PROFILE, eval=$MAX_EVAL, seed=$SEED)..."

setup_split() {
    local src_dir="$1"
    local group_name="$2"
    local n_profile="$3"
    local n_eval="$4"

    local prof_dir="${IMG_DIR}/${group_name}_profile"
    local eval_dir="${IMG_DIR}/${group_name}_eval"

    # Check if splits already have real images
    if [ -d "$prof_dir" ] && [ -d "$eval_dir" ]; then
        local real_count="$(find "$prof_dir" -maxdepth 2 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) 2>/dev/null | wc -l)"
        if [ "$real_count" -gt 0 ]; then
            log "  $group_name splits: already exist (profile=$real_count images)"
            return
        fi
        rm -rf "$prof_dir" "$eval_dir"
    fi

    mkdir -p "$prof_dir" "$eval_dir"

    # Collect all images (exclude masks — filter out files with _mask, _segmentation, _GT in name)
    local all_images
    all_images="$(find "$src_dir" -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) \
        | grep -v -i '_mask\|_segmentation\|_GT\|ground.truth\|GroundTruth' \
        | sort)"
    local total
    total="$(echo "$all_images" | grep -c . || echo 0)"

    if [ "$total" -lt 10 ]; then
        log "  WARNING: $group_name has only $total images in $src_dir — skipping"
        rm -rf "$prof_dir" "$eval_dir"
        return
    fi

    # Deterministic shuffle using md5 hash (handles filenames with spaces)
    local shuffled
    shuffled="$(echo "$all_images" | while IFS= read -r f; do printf '%s %s\n' "$(printf '%s' "${SEED}_${f}" | md5sum | cut -c1-8)" "$f"; done | sort -k1,1 | cut -d' ' -f2-)"

    local count=0
    while IFS= read -r img_path; do
        [ -z "$img_path" ] && continue
        count=$((count + 1))
        # Replace spaces/parens in filename for safety
        local safe_name
        safe_name="$(basename "$img_path" | tr ' ()' '_[]')"
        if [ "$count" -le "$n_profile" ]; then
            cp -- "$img_path" "$prof_dir/$safe_name"
        elif [ "$count" -le $((n_profile + n_eval)) ]; then
            cp -- "$img_path" "$eval_dir/$safe_name"
        else
            break
        fi
    done <<< "$shuffled"

    local p_count="$(find "$prof_dir" -maxdepth 1 -type f | wc -l)"
    local e_count="$(find "$eval_dir" -maxdepth 1 -type f | wc -l)"
    log "  $group_name: profile=$p_count, eval=$e_count (from $total total)"
}

# BUSI split (exclude mask files which have _mask in name)
if [ -d "$BUSI_RAW" ]; then
    setup_split "$BUSI_RAW" "busi" "$MAX_PROFILE" "$MAX_EVAL"
fi

# PCB split
if [ -d "$PCB_RAW" ]; then
    setup_split "$PCB_RAW" "pcb" "$MAX_PROFILE" "$MAX_EVAL"
fi

# DeepGlobe split (use satellite tiles, exclude masks)
if [ -d "$DEEPGLOBE_RAW" ]; then
    setup_split "$DEEPGLOBE_RAW" "deepglobe" "$MAX_PROFILE" "$MAX_EVAL"
fi

log ""

# 0.6 Verify general profile exists
if [ ! -d "$GENERAL_PROFILE_DIR/general" ]; then
    log "ERROR: General COCO profile not found at $GENERAL_PROFILE_DIR/general/"
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

for GROUP in busi pcb deepglobe; do
    # Skip groups whose image dirs don't exist
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

log_section "PHASE 1.5: Profile analysis"

# Build groups list from what actually has profile data
PROFILED_GROUPS="general"
for G in busi pcb deepglobe; do
    if [ -d "$OUT/profile/$G" ] && [ "$(find "$OUT/profile/$G" -name '*.npy' | wc -l)" -ge 1024 ]; then
        PROFILED_GROUPS="$PROFILED_GROUPS,$G"
    fi
done

log "Analyzing groups: $PROFILED_GROUPS"

python analyze_profile.py \
    --profile_dir "$OUT/profile" \
    --groups "$PROFILED_GROUPS" \
    --out "$OUT/plots/profile_analysis.txt" \
    2>&1 | tee -a "${LOG_DIR}/analyze_profile.log"

log "  Report: $OUT/plots/profile_analysis.txt"

log "Generating variance distribution plots..."
python plot_variance_distributions.py \
    --profile_dir "$OUT/profile" \
    --groups "$PROFILED_GROUPS" \
    --out_dir "$OUT/plots/variance" \
    2>&1 | tee -a "${LOG_DIR}/variance_plots.log"

log "  Plots: $OUT/plots/variance/"

# ── Phase 2: Freeze + Eval grid ──────────────────────────────────────────────

log_section "PHASE 2: Freeze + Eval (specialized profile vs general profile)"

for GROUP in busi pcb deepglobe; do
    # Skip groups whose eval dirs don't exist
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
                --diagnostic_level 0 \
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
                --diagnostic_level 0 \
                --out_dir "$OUT/$EVAL_TAG_GEN" \
                2>&1 | tee -a "${LOG_DIR}/${EVAL_TAG_GEN}.log"

            log "  Done: $EVAL_TAG_GEN"
        fi
    done
done

# ── Phase 3: Plots + summary ─────────────────────────────────────────────────

log_section "PHASE 3: Generate comparison plots + summary"

# Build evaluated groups list
EVAL_GROUPS=""
for G in busi pcb deepglobe; do
    if [ -d "$OUT/eval_${G}_f0.125" ]; then
        if [ -z "$EVAL_GROUPS" ]; then
            EVAL_GROUPS="$G"
        else
            EVAL_GROUPS="$EVAL_GROUPS,$G"
        fi
    fi
done

if [ -n "$EVAL_GROUPS" ]; then
    log "Generating sweep plots for: $EVAL_GROUPS"
    python plot_sweep_results.py \
        --results_dir "$OUT" \
        --groups "$EVAL_GROUPS" \
        --fractions "0.125,0.25,0.375" \
        2>&1 | tee -a "${LOG_DIR}/plot_sweep.log"
fi

log ""
log "Generating cross-domain comparison summary..."

python -c "
import json, os, sys
from pathlib import Path

out_dir = Path('$OUT')
groups = ['busi', 'pcb', 'deepglobe']
fractions = [0.125, 0.25, 0.375]

print()
print('=' * 110)
print('CROSS-DOMAIN LOCALITY EXPERIMENT (WAVE 2) — SUMMARY')
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

summary_path = out_dir / 'wave2_summary.json'
json.dump(rows, open(summary_path, 'w'), indent=2)
print(f'Summary saved: {summary_path}')

print()
print('KEY COMPARISONS:')
print('  - BUSI (ultrasound) vs general: expect lowest correlation (~0.72)')
print('  - PCB (industrial) vs general: expect moderate (~0.85)')
print('  - DeepGlobe (satellite) vs general: expect moderate (~0.88)')
print('  - All three should show domain-specific > general profile IoU')
print('=' * 110)
" 2>&1 | tee -a "$MAIN_LOG" | tee "$OUT/wave2_summary.txt"

# ── Done ──────────────────────────────────────────────────────────────────────

log_section "EXPERIMENT COMPLETE"
log "Results:    $OUT/"
log "Profiles:   $OUT/profile/"
log "Plots:      $OUT/plots/"
log "Summary:    $OUT/wave2_summary.txt"
log "Full log:   $MAIN_LOG"
log ""
log "Next steps:"
log "  1. Check $OUT/plots/profile_analysis.txt for Spearman correlations"
log "  2. Compare with Wave 1: kvasir=0.983, isic=0.863"
log "  3. BUSI should be the lowest correlation of all domains tested"
