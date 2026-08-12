#!/usr/bin/env bash
# check_gt_masks.sh — Verify GT mask availability for miss rate analysis
#
# Checks whether ground-truth masks are accessible for each of the 5 domains.
# Downloads missing GT data where possible (ISIC requires a separate download).
#
# Usage:
#   bash scripts/check_gt_masks.sh [--download] [--data-root ./data]
#
# Options:
#   --download    Attempt to download missing GT masks
#   --data-root   Data directory (default: ./data)
#   --verbose     Show detailed file counts

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT="${DATA_ROOT:-./data}"
DOWNLOAD=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --download)   DOWNLOAD=true; shift ;;
        --data-root)  DATA_ROOT="$2"; shift 2 ;;
        --verbose)    VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/check_gt_masks.sh [--download] [--data-root ./data] [--verbose]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

count_files() {
    local dir="$1"
    local pattern="$2"
    find "$dir" -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

# ============================================================================
# Check each domain
# ============================================================================

echo ""
echo "================================================================"
echo "  GT Mask Availability Check"
echo "  Data root: $DATA_ROOT"
echo "================================================================"
echo ""

ALL_OK=true

# ---------- 1. Kvasir-SEG ----------
echo "[1/5] Kvasir-SEG (polyp masks)"

KVASIR_MASKS="$DATA_ROOT/kvasir-seg/masks"
if [ -d "$KVASIR_MASKS" ]; then
    n=$(count_files "$KVASIR_MASKS" "*.jpg")
    n_png=$(count_files "$KVASIR_MASKS" "*.png")
    total=$((n + n_png))
    if [ "$total" -gt 100 ]; then
        pass "Found $total mask files in $KVASIR_MASKS"
    else
        warn "Only $total masks found (expected ~1000)"
        ALL_OK=false
    fi
else
    fail "Directory not found: $KVASIR_MASKS"
    ALL_OK=false
    if $DOWNLOAD; then
        echo "    Downloading Kvasir-SEG..."
        mkdir -p "$DATA_ROOT"
        wget -q "https://datasets.simula.no/downloads/kvasir-seg.zip" -O "$DATA_ROOT/kvasir-seg.zip"
        unzip -q "$DATA_ROOT/kvasir-seg.zip" -d "$DATA_ROOT/"
        rm -f "$DATA_ROOT/kvasir-seg.zip"
        # Normalize directory name
        for candidate in "$DATA_ROOT/Kvasir-SEG" "$DATA_ROOT/kvasir-seg"; do
            if [ -d "$candidate/masks" ]; then
                if [ "$candidate" != "$DATA_ROOT/kvasir-seg" ]; then
                    mv "$candidate" "$DATA_ROOT/kvasir-seg"
                fi
                break
            fi
        done
        if [ -d "$KVASIR_MASKS" ]; then
            n=$(count_files "$KVASIR_MASKS" "*")
            pass "Downloaded: $n mask files"
        else
            fail "Download failed — masks dir still missing"
        fi
    fi
fi

# ---------- 2. ISIC 2018 ----------
echo ""
echo "[2/5] ISIC 2018 (lesion segmentation masks)"

ISIC_GT="$DATA_ROOT/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
if [ -d "$ISIC_GT" ]; then
    n=$(count_files "$ISIC_GT" "*.png")
    if [ "$n" -gt 100 ]; then
        pass "Found $n GT masks in $ISIC_GT"
    else
        warn "Only $n masks found (expected ~2594)"
        ALL_OK=false
    fi
else
    fail "Directory not found: $ISIC_GT"
    ALL_OK=false

    if $DOWNLOAD; then
        echo "    Downloading ISIC 2018 Ground Truth (~200 MB)..."
        mkdir -p "$DATA_ROOT/ISIC2018"
        ISIC_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
        if wget -q --spider "$ISIC_URL" 2>/dev/null; then
            wget -q "$ISIC_URL" -O "$DATA_ROOT/ISIC2018/gt.zip"
            unzip -q "$DATA_ROOT/ISIC2018/gt.zip" -d "$DATA_ROOT/ISIC2018/"
            rm -f "$DATA_ROOT/ISIC2018/gt.zip"
            if [ -d "$ISIC_GT" ]; then
                n=$(count_files "$ISIC_GT" "*.png")
                pass "Downloaded: $n GT masks"
            else
                fail "Download completed but expected directory not found"
                echo "    Check: ls $DATA_ROOT/ISIC2018/"
            fi
        else
            fail "Cannot reach ISIC download URL"
            echo ""
            echo "    ┌─────────────────────────────────────────────────────────────┐"
            echo "    │  MANUAL DOWNLOAD REQUIRED for ISIC GT masks:                │"
            echo "    │                                                             │"
            echo "    │  wget https://isic-challenge-data.s3.amazonaws.com/2018/\\   │"
            echo "    │       ISIC2018_Task1_Training_GroundTruth.zip               │"
            echo "    │                                                             │"
            echo "    │  unzip -d $DATA_ROOT/ISIC2018/ ISIC2018_Task1_Training_GroundTruth.zip│"
            echo "    │                                                             │"
            echo "    │  Expected result:                                           │"
            echo "    │    $ISIC_GT/ISIC_*_segmentation.png                         │"
            echo "    └─────────────────────────────────────────────────────────────┘"
            echo ""
        fi
    else
        echo ""
        echo "    To download ISIC GT masks, run:"
        echo "      bash scripts/check_gt_masks.sh --download"
        echo "    Or manually:"
        echo "      wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
        echo "      unzip -d $DATA_ROOT/ISIC2018/ ISIC2018_Task1_Training_GroundTruth.zip"
        echo ""
    fi
fi

# ---------- 3. BUSI ----------
echo ""
echo "[3/5] BUSI (breast ultrasound tumor masks)"

BUSI_DIR="$DATA_ROOT/busi"
if [ -d "$BUSI_DIR" ]; then
    n_masks=$(find "$BUSI_DIR" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ')
    n_images=$(find "$BUSI_DIR" -name '*.png' 2>/dev/null | grep -v '_mask' | wc -l | tr -d ' ')
    if [ "$n_masks" -gt 50 ]; then
        pass "Found $n_masks mask files alongside $n_images images"
        if $VERBOSE; then
            echo "      benign:    $(find "$BUSI_DIR/benign" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ') masks"
            echo "      malignant: $(find "$BUSI_DIR/malignant" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ') masks"
            echo "      normal:    $(find "$BUSI_DIR/normal" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ') masks (should be ~0)"
        fi
    else
        warn "Only $n_masks mask files found (expected ~650+)"
        ALL_OK=false
    fi
else
    fail "Directory not found: $BUSI_DIR"
    ALL_OK=false
    echo "    BUSI masks are included with the image download (Kaggle: aryashah2k/breast-ultrasound-images-dataset)"
    echo "    Download and extract to: $BUSI_DIR/{benign,malignant,normal}/"
fi

# ---------- 4. DeepGlobe ----------
echo ""
echo "[4/5] DeepGlobe (road segmentation masks)"

DEEPGLOBE_DIR="$DATA_ROOT/deepglobe"
if [ -d "$DEEPGLOBE_DIR" ]; then
    n_masks=$(find "$DEEPGLOBE_DIR" -name '*_mask*' 2>/dev/null | wc -l | tr -d ' ')
    n_sat=$(find "$DEEPGLOBE_DIR" -name '*_sat*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_masks" -gt 100 ]; then
        pass "Found $n_masks mask files (alongside $n_sat satellite images)"
    else
        warn "Only $n_masks mask files found (expected ~6000+)"
        ALL_OK=false
        echo "    DeepGlobe masks (*_mask.png) should be in the same directories as *_sat.png files"
        echo "    They are included in the Kaggle download: balraj98/deepglobe-road-extraction-dataset"
    fi
else
    fail "Directory not found: $DEEPGLOBE_DIR"
    ALL_OK=false
    echo "    Download from Kaggle: balraj98/deepglobe-road-extraction-dataset"
    echo "    Extract to: $DEEPGLOBE_DIR/"
fi

# ---------- 5. PCB ----------
echo ""
echo "[5/5] PCB (defect bounding box annotations)"

PCB_DIR="$DATA_ROOT/pcb"
if [ -d "$PCB_DIR" ]; then
    n_xml=$(find "$PCB_DIR" -name '*.xml' 2>/dev/null | wc -l | tr -d ' ')
    n_images=$(find "$PCB_DIR" \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_xml" -gt 50 ]; then
        pass "Found $n_xml XML annotations (PASCAL VOC format) for $n_images images"
    elif [ "$n_xml" -gt 0 ]; then
        warn "Only $n_xml XML annotations found"
        ALL_OK=false
    else
        # Check for alternative annotation formats
        n_txt=$(find "$PCB_DIR" -name '*.txt' 2>/dev/null | grep -v "readme\|README\|license" | wc -l | tr -d ' ')
        if [ "$n_txt" -gt 50 ]; then
            warn "No XML but found $n_txt .txt annotations (YOLO format?)"
            echo "    The miss_rate_analysis expects PASCAL VOC XML format."
            echo "    May need conversion from YOLO format."
            ALL_OK=false
        else
            fail "No annotation files found (XML or TXT)"
            ALL_OK=false
        fi
    fi
else
    fail "Directory not found: $PCB_DIR"
    ALL_OK=false
    echo "    Download from Kaggle: akhatova/pcb-defects"
    echo "    Extract to: $PCB_DIR/"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================"
if $ALL_OK; then
    echo -e "  ${GREEN}All GT masks available!${NC} Ready to run miss rate analysis."
else
    echo -e "  ${YELLOW}Some GT masks are missing.${NC}"
    echo ""
    echo "  To auto-download what's available:"
    echo "    bash scripts/check_gt_masks.sh --download --data-root $DATA_ROOT"
    echo ""
    echo "  For Kaggle datasets, ensure kaggle CLI is configured:"
    echo "    pip install kaggle"
    echo "    # Place ~/.kaggle/kaggle.json with your API key"
fi
echo "================================================================"
echo ""

if $ALL_OK; then
    exit 0
else
    exit 1
fi
