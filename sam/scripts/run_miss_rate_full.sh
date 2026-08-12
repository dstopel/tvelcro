#!/usr/bin/env bash
# run_miss_rate_full.sh — Check GT masks, download if needed, then run analysis
#
# Single script that does everything:
#   1. Checks GT mask availability for all 5 domains
#   2. Downloads missing GT masks (ISIC GT zip, etc.)
#   3. Runs miss rate decomposition across all domains × freeze fractions
#
# Usage:
#   bash scripts/run_miss_rate_full.sh
#
# Environment overrides:
#   DATA_ROOT=./data  CKPT=./data/sam_vit_h_4b8939.pth  MAX_IMAGES=0  bash scripts/run_miss_rate_full.sh

set -euo pipefail

# ============================================================================
# Configuration (all overridable via environment)
# ============================================================================

DATA_ROOT="${DATA_ROOT:-./data}"
CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
OUT_DIR="${OUT_DIR:-./results_miss_analysis}"
MAX_IMAGES="${MAX_IMAGES:-0}"  # 0 = all
DEVICE="${DEVICE:-cuda}"
DOMAINS="${DOMAINS:-kvasir,isic,busi,deepglobe,pcb}"
FRACTIONS="${FRACTIONS:-0.125,0.25,0.375}"
FORCE="${FORCE:-false}"

# Image directory locations (wave1 and wave2)
IMG_DIR_W1="${IMG_DIR_W1:-./image_dirs_cross_domain}"
IMG_DIR_W2="${IMG_DIR_W2:-./image_dirs_wave2}"

# Manifest search paths (in priority order)
MANIFEST_DIRS=(
    "./results_cross_domain"
    "./results_wave2"
    "./results_sam_velcro"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LOG_DIR="$OUT_DIR"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/run_miss_rate_full.log"

# ============================================================================
# Helpers
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "$1" | tee -a "$LOGFILE"; }
pass() { log "  ${GREEN}✓${NC} $1"; }
fail() { log "  ${RED}✗${NC} $1"; }
warn() { log "  ${YELLOW}!${NC} $1"; }
header() { log "\n${BOLD}${CYAN}$1${NC}"; }

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# ============================================================================
# PHASE 1: Check GT Mask Availability
# ============================================================================

header "================================================================"
header "  PHASE 1: Checking GT Mask Availability"
header "  $(timestamp)"
header "================================================================"
log ""

GT_STATUS=()  # Track per-domain GT status: "ok" or "missing"

# --- Kvasir-SEG ---
log "[1/5] Kvasir-SEG (polyp masks)"
KVASIR_MASKS="$DATA_ROOT/kvasir-seg/masks"
if [ -d "$KVASIR_MASKS" ]; then
    n=$(find "$KVASIR_MASKS" \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -gt 100 ]; then
        pass "Found $n masks in $KVASIR_MASKS"
        GT_STATUS+=("ok")
    else
        warn "Only $n masks (expected ~1000)"
        GT_STATUS+=("partial")
    fi
else
    fail "Not found: $KVASIR_MASKS"
    GT_STATUS+=("missing")
fi

# --- ISIC 2018 ---
log ""
log "[2/5] ISIC 2018 (lesion segmentation)"
ISIC_GT="$DATA_ROOT/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
if [ -d "$ISIC_GT" ]; then
    n=$(find "$ISIC_GT" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -gt 100 ]; then
        pass "Found $n GT masks in $ISIC_GT"
        GT_STATUS+=("ok")
    else
        warn "Only $n masks (expected ~2594)"
        GT_STATUS+=("partial")
    fi
else
    fail "Not found: $ISIC_GT"
    GT_STATUS+=("missing")
fi

# --- BUSI ---
log ""
log "[3/5] BUSI (breast ultrasound)"
BUSI_DIR="$DATA_ROOT/busi"
if [ -d "$BUSI_DIR" ]; then
    n_masks=$(find "$BUSI_DIR" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_masks" -gt 50 ]; then
        pass "Found $n_masks mask files"
        GT_STATUS+=("ok")
    else
        warn "Only $n_masks mask files"
        GT_STATUS+=("partial")
    fi
else
    fail "Not found: $BUSI_DIR"
    GT_STATUS+=("missing")
fi

# --- DeepGlobe ---
log ""
log "[4/5] DeepGlobe (road masks)"
DEEPGLOBE_DIR="$DATA_ROOT/deepglobe"
if [ -d "$DEEPGLOBE_DIR" ]; then
    n_masks=$(find "$DEEPGLOBE_DIR" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_masks" -gt 100 ]; then
        pass "Found $n_masks road masks"
        GT_STATUS+=("ok")
    else
        warn "Only $n_masks mask files"
        GT_STATUS+=("partial")
    fi
else
    fail "Not found: $DEEPGLOBE_DIR"
    GT_STATUS+=("missing")
fi

# --- PCB ---
log ""
log "[5/5] PCB (XML annotations)"
PCB_DIR="$DATA_ROOT/pcb"
if [ -d "$PCB_DIR" ]; then
    n_xml=$(find "$PCB_DIR" -name '*.xml' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_xml" -gt 50 ]; then
        pass "Found $n_xml XML annotations"
        GT_STATUS+=("ok")
    else
        n_txt=$(find "$PCB_DIR" -name '*.txt' 2>/dev/null | grep -cv "readme\|README\|license" || echo 0)
        if [ "$n_txt" -gt 50 ]; then
            warn "No XML but found $n_txt TXT annotations (may need conversion)"
            GT_STATUS+=("partial")
        else
            warn "Only $n_xml XML files"
            GT_STATUS+=("partial")
        fi
    fi
else
    fail "Not found: $PCB_DIR"
    GT_STATUS+=("missing")
fi

# ============================================================================
# PHASE 2: Download Missing GT Masks
# ============================================================================

header ""
header "================================================================"
header "  PHASE 2: Downloading Missing GT Masks"
header "  $(timestamp)"
header "================================================================"
log ""

DOWNLOADS_NEEDED=false
for status in "${GT_STATUS[@]}"; do
    if [ "$status" = "missing" ]; then
        DOWNLOADS_NEEDED=true
        break
    fi
done

if ! $DOWNLOADS_NEEDED; then
    pass "All GT masks available — no downloads needed"
else
    # --- Download ISIC GT if missing ---
    if [ "${GT_STATUS[1]}" = "missing" ]; then
        log "Downloading ISIC 2018 Ground Truth (~200 MB)..."
        ISIC_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
        mkdir -p "$DATA_ROOT/ISIC2018"

        if wget -q --spider "$ISIC_URL" 2>/dev/null; then
            wget --progress=bar:force "$ISIC_URL" -O "$DATA_ROOT/ISIC2018/gt.zip" 2>&1 | tee -a "$LOGFILE"
            unzip -q "$DATA_ROOT/ISIC2018/gt.zip" -d "$DATA_ROOT/ISIC2018/"
            rm -f "$DATA_ROOT/ISIC2018/gt.zip"
            if [ -d "$ISIC_GT" ]; then
                n=$(find "$ISIC_GT" -name '*.png' | wc -l | tr -d ' ')
                pass "ISIC GT downloaded: $n masks"
            else
                fail "ISIC GT: zip extracted but expected directory not found"
                log "    Check contents: ls $DATA_ROOT/ISIC2018/"
            fi
        else
            fail "Cannot reach ISIC download URL"
            log ""
            log "    Manual download required:"
            log "      wget $ISIC_URL"
            log "      unzip -d $DATA_ROOT/ISIC2018/ ISIC2018_Task1_Training_GroundTruth.zip"
            log ""
        fi
    fi

    # --- Download Kvasir if missing ---
    if [ "${GT_STATUS[0]}" = "missing" ]; then
        log "Downloading Kvasir-SEG (~50 MB)..."
        KVASIR_URL="https://datasets.simula.no/downloads/kvasir-seg.zip"
        if wget -q --spider "$KVASIR_URL" 2>/dev/null; then
            wget --progress=bar:force "$KVASIR_URL" -O "$DATA_ROOT/kvasir-seg.zip" 2>&1 | tee -a "$LOGFILE"
            unzip -q "$DATA_ROOT/kvasir-seg.zip" -d "$DATA_ROOT/"
            rm -f "$DATA_ROOT/kvasir-seg.zip"
            # Normalize name
            for candidate in "$DATA_ROOT/Kvasir-SEG" "$DATA_ROOT/kvasir-seg"; do
                if [ -d "$candidate/masks" ]; then
                    [ "$candidate" != "$DATA_ROOT/kvasir-seg" ] && mv "$candidate" "$DATA_ROOT/kvasir-seg"
                    break
                fi
            done
            if [ -d "$KVASIR_MASKS" ]; then
                pass "Kvasir masks downloaded"
            else
                fail "Kvasir download: masks dir not found after extraction"
            fi
        else
            fail "Cannot reach Kvasir download URL"
        fi
    fi

    # --- BUSI / DeepGlobe / PCB require Kaggle ---
    for i in 2 3 4; do
        if [ "${GT_STATUS[$i]}" = "missing" ]; then
            case $i in
                2) ds_name="BUSI"; ds_slug="aryashah2k/breast-ultrasound-images-dataset" ;;
                3) ds_name="DeepGlobe"; ds_slug="balraj98/deepglobe-road-extraction-dataset" ;;
                4) ds_name="PCB"; ds_slug="akhatova/pcb-defects" ;;
            esac

            if command -v kaggle &>/dev/null; then
                log "Downloading $ds_name via Kaggle CLI..."
                kaggle datasets download -d "$ds_slug" -p "$DATA_ROOT" --unzip 2>&1 | tee -a "$LOGFILE"
                pass "$ds_name downloaded (may need path normalization)"
            else
                warn "$ds_name missing — Kaggle CLI not found"
                log "    Install: pip install kaggle"
                log "    Then: kaggle datasets download -d $ds_slug -p $DATA_ROOT --unzip"
            fi
        fi
    done
fi

# ============================================================================
# PHASE 3: Run Miss Rate Analysis
# ============================================================================

header ""
header "================================================================"
header "  PHASE 3: Running Miss Rate Decomposition"
header "  $(timestamp)"
header "================================================================"
log ""

# Verify checkpoint
if [ ! -f "$CKPT" ]; then
    fail "SAM checkpoint not found: $CKPT"
    log "  Set CKPT=/path/to/sam_vit_h_4b8939.pth"
    exit 1
fi
pass "Checkpoint: $CKPT"

# Determine image directory for a domain
get_image_dir() {
    local domain="$1"
    # Wave 1 domains
    if [ -d "$IMG_DIR_W1/${domain}_eval" ]; then
        echo "$IMG_DIR_W1"
        return
    fi
    # Wave 2 domains
    if [ -d "$IMG_DIR_W2/${domain}_eval" ]; then
        echo "$IMG_DIR_W2"
        return
    fi
    # Fallback: check both without _eval suffix
    if [ -d "$IMG_DIR_W1/$domain" ]; then
        echo "$IMG_DIR_W1"
        return
    fi
    if [ -d "$IMG_DIR_W2/$domain" ]; then
        echo "$IMG_DIR_W2"
        return
    fi
    echo ""
}

# Find freeze manifest for a domain + fraction
find_manifest() {
    local domain="$1"
    local frac="$2"

    for base_dir in "${MANIFEST_DIRS[@]}"; do
        # Pattern 1: {base}/{domain}/freeze_{domain}_f{frac}/freeze_manifest.json
        local p="$base_dir/$domain/freeze_${domain}_f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return

        # Pattern 2: {base}/{domain}/f{frac}/freeze_manifest.json
        p="$base_dir/$domain/f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return

        # Pattern 3: {base}/freeze_{domain}_f{frac}/freeze_manifest.json
        p="$base_dir/freeze_${domain}_f${frac}/freeze_manifest.json"
        [ -f "$p" ] && echo "$p" && return
    done

    echo ""
}

# Build max_images argument
MAX_IMG_ARG=""
if [ "$MAX_IMAGES" -gt 0 ] 2>/dev/null; then
    MAX_IMG_ARG="--max_images $MAX_IMAGES"
fi

# Run analysis
IFS=',' read -ra DOMAIN_LIST <<< "$DOMAINS"
IFS=',' read -ra FRAC_LIST <<< "$FRACTIONS"

PASS=0
SKIP=0
FAIL=0

log ""
log "Domains: ${DOMAIN_LIST[*]}"
log "Fractions: ${FRAC_LIST[*]}"
log "Total runs: $((${#DOMAIN_LIST[@]} * ${#FRAC_LIST[@]}))"
log ""

for domain in "${DOMAIN_LIST[@]}"; do
    img_dir=$(get_image_dir "$domain")
    if [ -z "$img_dir" ]; then
        warn "No eval images found for $domain — skipping"
        SKIP=$((SKIP + ${#FRAC_LIST[@]}))
        continue
    fi

    for frac in "${FRAC_LIST[@]}"; do
        run_out="$OUT_DIR/$domain/f${frac}"

        # Idempotency check
        if [ "$FORCE" != "true" ] && [ -f "$run_out/miss_analysis_results.json" ]; then
            log "  [SKIP] $domain @ f=$frac — already done (use FORCE=true to re-run)"
            SKIP=$((SKIP + 1))
            continue
        fi

        # Find manifest
        manifest=$(find_manifest "$domain" "$frac")
        if [ -z "$manifest" ]; then
            warn "[SKIP] $domain @ f=$frac — no manifest found"
            SKIP=$((SKIP + 1))
            continue
        fi

        log "  [RUN] $domain @ f=$frac"
        log "        manifest: $manifest"
        log "        img_dir:  $img_dir"

        if python3 scripts/miss_rate_analysis.py \
            --domain "$domain" \
            --checkpoint "$CKPT" \
            --freeze_manifest "$manifest" \
            --image_dir "$img_dir" \
            --data_root "$DATA_ROOT" \
            --out_dir "$run_out" \
            --device "$DEVICE" \
            $MAX_IMG_ARG \
            2>&1 | tee -a "$LOGFILE"; then
            pass "$domain @ f=$frac — done"
            PASS=$((PASS + 1))
        else
            fail "$domain @ f=$frac — FAILED (exit code $?)"
            FAIL=$((FAIL + 1))
        fi
        log ""
    done
done

# ============================================================================
# Summary
# ============================================================================

header ""
header "================================================================"
header "  COMPLETE — $(timestamp)"
header "================================================================"
log ""
log "  Results: $PASS passed, $SKIP skipped, $FAIL failed"
log "  Output:  $OUT_DIR/"
log "  Log:     $LOGFILE"
log ""

# Print summary table if results exist
if [ "$PASS" -gt 0 ]; then
    header "  Summary Table:"
    log ""
    printf "  %-12s %6s %7s %7s %7s %7s %6s\n" "Domain" "Frac" "Miss%" "Dom%" "Red%" "BG%" "%Dom"
    printf "  %-12s %6s %7s %7s %7s %7s %6s\n" "------" "----" "-----" "----" "----" "---" "----"

    for domain in "${DOMAIN_LIST[@]}"; do
        for frac in "${FRAC_LIST[@]}"; do
            json="$OUT_DIR/$domain/f${frac}/miss_analysis_results.json"
            [ -f "$json" ] || continue

            # Extract values (try jq, fallback to python)
            if command -v jq &>/dev/null; then
                total=$(jq -r '.summary.total_miss_rate' "$json")
                dom=$(jq -r '.summary.domain_miss_rate' "$json")
                red=$(jq -r '.summary.redundant_miss_rate' "$json")
                bg=$(jq -r '.summary.background_miss_rate' "$json")
                pct=$(jq -r '.summary.pct_misses_domain' "$json")
            else
                read -r total dom red bg pct < <(python3 -c "
import json, sys
d = json.load(open('$json'))['summary']
print(d['total_miss_rate'], d['domain_miss_rate'], d['redundant_miss_rate'], d['background_miss_rate'], d['pct_misses_domain'])
")
            fi

            printf "  %-12s %6s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %5.1f%%\n" \
                "$domain" "$frac" \
                "$(echo "$total * 100" | bc -l)" \
                "$(echo "$dom * 100" | bc -l)" \
                "$(echo "$red * 100" | bc -l)" \
                "$(echo "$bg * 100" | bc -l)" \
                "$pct"
        done
    done
    log ""
fi

log "Done."
