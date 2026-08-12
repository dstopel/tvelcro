#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Miss Rate Decomposition Analysis
# Runs miss_rate_analysis.py across all domains and freeze fractions.
# Designed for university VM with NVIDIA A10 24GB GPU.
# ==============================================================================

# --- Configuration (overridable via environment) ------------------------------
CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results_miss_analysis}"
MAX_IMAGES="${MAX_IMAGES:-0}"  # 0 = all images
DOMAINS="${DOMAINS:-kvasir,isic,busi,deepglobe,pcb}"
FRACTIONS="${FRACTIONS:-0.125,0.25,0.375}"

# --- Flags --------------------------------------------------------------------
DRY_RUN=0
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--force]"
            echo ""
            echo "  --dry-run   Print commands without executing"
            echo "  --force     Re-run even if output JSON already exists"
            echo ""
            echo "Environment variables:"
            echo "  CKPT         SAM checkpoint path (default: ./data/sam_vit_h_4b8939.pth)"
            echo "  DATA_ROOT    Data root for GT masks (default: ./data)"
            echo "  OUT_DIR      Output directory (default: ./results_miss_analysis)"
            echo "  MAX_IMAGES   Max images per domain, 0=all (default: 0)"
            echo "  DOMAINS      Comma-separated domains (default: kvasir,isic,busi,deepglobe,pcb)"
            echo "  FRACTIONS    Comma-separated freeze fractions (default: 0.125,0.25,0.375)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# --- Setup --------------------------------------------------------------------
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/miss_analysis.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*"
    echo "$msg" >&2
    echo "$msg" >> "$LOG_FILE"
}

err() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*"
    echo "$msg" >&2
    echo "$msg" >> "$LOG_FILE"
}

# --- Time estimate ------------------------------------------------------------
estimate_time() {
    local n_domains n_fractions total_runs
    IFS=',' read -ra dom_arr <<< "$DOMAINS"
    IFS=',' read -ra frac_arr <<< "$FRACTIONS"
    n_domains=${#dom_arr[@]}
    n_fractions=${#frac_arr[@]}
    total_runs=$((n_domains * n_fractions))

    if [[ "$MAX_IMAGES" -eq 0 ]]; then
        log "Estimated time: ~5-8 minutes per domain-fraction pair (full dataset)"
        log "Total runs: $total_runs => ~$((total_runs * 6)) minutes estimated"
    else
        local est_per_run=$(( (MAX_IMAGES * 3 + 59) / 60 ))  # ~3s per image
        [[ $est_per_run -lt 1 ]] && est_per_run=1
        log "Estimated time: ~${est_per_run} minutes per run ($MAX_IMAGES images)"
        log "Total runs: $total_runs => ~$((total_runs * est_per_run)) minutes estimated"
    fi
}

# --- Verify checkpoint --------------------------------------------------------
if [[ ! -f "$CKPT" ]]; then
    err "SAM checkpoint not found: $CKPT"
    exit 1
fi
log "SAM checkpoint: $CKPT"

# --- Image directory resolution -----------------------------------------------
get_image_dir() {
    local domain="$1"
    case "$domain" in
        kvasir|isic)
            # Wave 1: prefer cross_domain, fall back to wave2
            if [[ -d "./image_dirs_cross_domain/${domain}_eval" ]]; then
                echo "./image_dirs_cross_domain/${domain}_eval"
            elif [[ -d "./image_dirs_wave2/${domain}_eval" ]]; then
                echo "./image_dirs_wave2/${domain}_eval"
            else
                echo ""
            fi
            ;;
        busi|pcb|deepglobe)
            # Wave 2 domains
            if [[ -d "./image_dirs_wave2/${domain}_eval" ]]; then
                echo "./image_dirs_wave2/${domain}_eval"
            else
                echo ""
            fi
            ;;
        *)
            echo ""
            ;;
    esac
}

# --- GT mask availability check -----------------------------------------------
check_gt_masks() {
    local domain="$1"
    case "$domain" in
        kvasir)
            if [[ -d "./data/kvasir-seg/masks" ]]; then
                return 0
            else
                warn "kvasir GT masks not found at ./data/kvasir-seg/masks/"
                return 1
            fi
            ;;
        isic)
            if [[ -d "./data/ISIC2018/ISIC2018_Task1_Training_GroundTruth" ]]; then
                return 0
            else
                warn "ISIC GT masks not found."
                warn "To download ISIC 2018 Task 1 ground truth:"
                warn "  1. Visit https://challenge.isic-archive.com/data/#2018"
                warn "  2. Download ISIC2018_Task1_Training_GroundTruth.zip"
                warn "  3. Extract to ./data/ISIC2018/ISIC2018_Task1_Training_GroundTruth/"
                warn "Continuing without ISIC GT check (analysis may handle missing masks)."
                return 0  # Allow run, let python script handle
            fi
            ;;
        busi)
            if [[ -d "./data/busi" ]] && compgen -G "./data/busi/*_mask*.png" > /dev/null 2>&1; then
                return 0
            elif [[ -d "./data/busi" ]]; then
                # Check recursively
                if find "./data/busi" -name "*_mask*.png" -print -quit 2>/dev/null | grep -q .; then
                    return 0
                fi
                warn "busi directory exists but no *_mask*.png files found"
                return 1
            else
                warn "busi GT not found at ./data/busi/"
                return 1
            fi
            ;;
        deepglobe)
            if [[ -d "./data/deepglobe" ]] && find "./data/deepglobe" -name "*_mask*" -print -quit 2>/dev/null | grep -q .; then
                return 0
            else
                warn "deepglobe GT masks not found at ./data/deepglobe/"
                return 1
            fi
            ;;
        pcb)
            if [[ -d "./data/pcb" ]] && find "./data/pcb" -name "*.xml" -print -quit 2>/dev/null | grep -q .; then
                return 0
            else
                warn "pcb GT annotations (*.xml) not found at ./data/pcb/"
                return 1
            fi
            ;;
        *)
            warn "Unknown domain: $domain"
            return 1
            ;;
    esac
}

# --- Freeze manifest resolution -----------------------------------------------
find_manifest() {
    local domain="$1"
    local fraction="$2"

    # Candidate paths in priority order
    local candidates=(
        "./results_cross_domain/${domain}/freeze_${domain}_f${fraction}/freeze_manifest.json"
        "./results_wave2/${domain}/freeze_${domain}_f${fraction}/freeze_manifest.json"
        "./results_sam_velcro/${domain}/freeze_${domain}_f${fraction}/freeze_manifest.json"
    )

    for cand in "${candidates[@]}"; do
        if [[ -f "$cand" ]]; then
            echo "$cand"
            return 0
        fi
    done

    echo ""
    return 1
}

# --- JSON value extraction (jq with python fallback) --------------------------
json_get() {
    local file="$1"
    local key="$2"
    if command -v jq &>/dev/null; then
        jq -r ".$key // \"N/A\"" "$file" 2>/dev/null || echo "N/A"
    else
        python3 -c "
import json, sys
try:
    with open('$file') as f:
        d = json.load(f)
    val = d.get('$key', 'N/A')
    if isinstance(val, float):
        print(f'{val:.4f}')
    else:
        print(val)
except Exception:
    print('N/A')
" 2>/dev/null || echo "N/A"
    fi
}

# ==============================================================================
# Main execution
# ==============================================================================

log "=========================================="
log "Miss Rate Decomposition Analysis"
log "=========================================="
log "Domains:    $DOMAINS"
log "Fractions:  $FRACTIONS"
log "MAX_IMAGES: $MAX_IMAGES (0=all)"
log "OUT_DIR:    $OUT_DIR"
log "DRY_RUN:    $DRY_RUN"
log "FORCE:      $FORCE"
log ""

estimate_time

IFS=',' read -ra DOMAIN_LIST <<< "$DOMAINS"
IFS=',' read -ra FRACTION_LIST <<< "$FRACTIONS"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

declare -a RESULTS_FILES=()

for domain in "${DOMAIN_LIST[@]}"; do
    log ""
    log "--- Domain: $domain ---"

    # Resolve image directory
    IMG_DIR=$(get_image_dir "$domain")
    if [[ -z "$IMG_DIR" ]]; then
        err "Image directory not found for domain: $domain"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    log "Image dir: $IMG_DIR"

    # Check GT masks
    if ! check_gt_masks "$domain"; then
        warn "GT mask check failed for $domain — proceeding anyway"
    fi

    for fraction in "${FRACTION_LIST[@]}"; do
        # Fraction as percentage label (for display)
        frac_pct=$(python3 -c "print(f'{float($fraction)*100:.1f}%')" 2>/dev/null || echo "${fraction}")

        log ""
        log "  Fraction: $fraction ($frac_pct)"

        # Find freeze manifest
        MANIFEST=$(find_manifest "$domain" "$fraction") || true
        if [[ -z "$MANIFEST" ]]; then
            warn "  Freeze manifest not found for $domain f=$fraction — skipping"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        log "  Manifest: $MANIFEST"

        # Output path
        RUN_OUT_DIR="$OUT_DIR/$domain/f${fraction}"
        RESULT_JSON="$RUN_OUT_DIR/miss_analysis_results.json"

        # Idempotency check
        if [[ -f "$RESULT_JSON" && "$FORCE" -eq 0 ]]; then
            log "  Output already exists: $RESULT_JSON (use --force to override)"
            RESULTS_FILES+=("$RESULT_JSON:$domain:$fraction")
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        # Build command
        CMD="python scripts/miss_rate_analysis.py"
        CMD+=" --checkpoint $CKPT"
        CMD+=" --freeze_manifest $MANIFEST"
        CMD+=" --domain $domain"
        CMD+=" --image_dir $IMG_DIR"
        CMD+=" --data_root $DATA_ROOT"
        CMD+=" --max_images $MAX_IMAGES"
        CMD+=" --out_dir $RUN_OUT_DIR"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            log "  [DRY-RUN] $CMD"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        # Execute
        mkdir -p "$RUN_OUT_DIR"
        log "  Running: $CMD"
        set +e
        eval "$CMD" >> "$LOG_FILE" 2>&1
        EXIT_CODE=$?
        set -e

        if [[ $EXIT_CODE -eq 0 ]]; then
            log "  Completed successfully (exit code: 0)"
            PASS_COUNT=$((PASS_COUNT + 1))
            RESULTS_FILES+=("$RESULT_JSON:$domain:$fraction")
        else
            err "  Failed with exit code: $EXIT_CODE"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
done

# ==============================================================================
# Summary table
# ==============================================================================

log ""
log "=========================================="
log "Execution Summary: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  SKIP=$SKIP_COUNT"
log "=========================================="

print_summary_table() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║           Miss Rate Decomposition Summary                           ║"
    echo "╠════════════╦═══════╦═══════════╦════════════╦══════════════╦════════╣"
    echo "║ Domain     ║ Frac  ║ Total Miss║ Domain Miss║ Background   ║ % Dom  ║"
    echo "╠════════════╬═══════╬═══════════╬════════════╬══════════════╬════════╣"

    for entry in "${RESULTS_FILES[@]}"; do
        IFS=':' read -r json_file domain fraction <<< "$entry"

        if [[ ! -f "$json_file" ]]; then
            continue
        fi

        frac_pct=$(python3 -c "print(f'{float($fraction)*100:.1f}%')" 2>/dev/null || echo "${fraction}")

        total_miss=$(json_get "$json_file" "total_miss_rate")
        domain_miss=$(json_get "$json_file" "domain_miss_rate")
        bg_miss=$(json_get "$json_file" "background_miss_rate")
        pct_domain=$(json_get "$json_file" "pct_domain_attributable")

        # Format values as percentages if numeric
        fmt_val() {
            local val="$1"
            if [[ "$val" == "N/A" ]]; then
                echo "  N/A  "
            else
                python3 -c "
v = float('$val')
if v < 1.0:
    print(f'{v*100:5.1f}%  ')
else:
    print(f'{v:5.1f}%  ')
" 2>/dev/null || echo "  N/A  "
            fi
        }

        t_fmt=$(fmt_val "$total_miss")
        d_fmt=$(fmt_val "$domain_miss")
        b_fmt=$(fmt_val "$bg_miss")
        p_fmt=$(fmt_val "$pct_domain")

        printf "║ %-10s ║ %-5s ║ %9s ║ %10s ║ %12s ║ %6s║\n" \
            "$domain" "$frac_pct" "$t_fmt" "$d_fmt" "$b_fmt" "$p_fmt"
    done

    echo "╚════════════╩═══════╩═══════════╩════════════╩══════════════╩════════╝"
    echo ""
}

# Print to stdout and append to log
print_summary_table | tee -a "$LOG_FILE"

log "Full log: $LOG_FILE"
log "Done."
