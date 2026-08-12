#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# ── Configuration ────────────────────────────────────────────────────────────
CKPT="${CKPT:-./data/sam_vit_h_4b8939.pth}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./results_combined_freeze}"
DEVICE="${DEVICE:-cuda}"
MLP_FRACTIONS="${MLP_FRACTIONS:-0.125,0.25,0.375}"
PROFILE_IMAGES="${PROFILE_IMAGES:-300}"
EVAL_IMAGES="${EVAL_IMAGES:-0}"  # 0 = all
SKIP_PROFILE="${SKIP_PROFILE:-false}"
DRY_RUN="${DRY_RUN:-false}"

DOMAINS=(kvasir isic busi pcb deepglobe)
declare -A PASSED=()
declare -A FAILED=()

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_cmd() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

resolve_image_dir() {
    local domain="$1"
    local primary secondary
    case "$domain" in
        kvasir|isic)   primary=./image_dirs_cross_domain; secondary=./image_dirs_wave2 ;;
        busi|pcb|deepglobe) primary=./image_dirs_wave2; secondary=./image_dirs_cross_domain ;;
    esac
    for base in "$primary" "$secondary"; do
        for suffix in "${domain}_profile" "${domain}_eval" "$domain"; do
            [ -d "$base/$suffix" ] && echo "$base" && return 0
        done
    done
    return 1
}

resolve_attn_manifest() {
    local domain="$1"
    local candidates=(
        "./results_cross_domain/${domain}/freeze_${domain}_f0.25/freeze_manifest.json"
        "./results_cross_domain/freeze_${domain}_f0.25/freeze_manifest.json"
        "./results_wave2/${domain}/freeze_${domain}_f0.25/freeze_manifest.json"
        "./results_wave2/freeze_${domain}_f0.25/freeze_manifest.json"
    )
    for path in "${candidates[@]}"; do
        [ -f "$path" ] && echo "$path" && return 0
    done
    return 1
}

# ── Main loop ────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
log "Combined attention+MLP freeze experiment"
log "Domains: ${DOMAINS[*]}"
log "MLP fractions: $MLP_FRACTIONS"
[ "$DRY_RUN" = "true" ] && log "*** DRY-RUN MODE ***"
echo ""

for domain in kvasir isic busi pcb; do
    log "=== $domain ==="

    img_dir=$(resolve_image_dir "$domain") || { log "SKIP $domain: no image dir found"; FAILED[$domain]="no image dir"; continue; }
    manifest=$(resolve_attn_manifest "$domain") || { log "SKIP $domain: no attention manifest found"; FAILED[$domain]="no manifest"; continue; }

    log "  image_dir: $img_dir"
    log "  manifest:  $manifest"

    EXTRA_ARGS=()
    [ "$SKIP_PROFILE" = "true" ] && EXTRA_ARGS+=(--skip_profile)

    run_cmd python3 scripts/exp_combined_freeze.py \
        --domain "$domain" \
        --checkpoint "$CKPT" \
        --attn_manifest "$manifest" \
        --image_dir "$img_dir" \
        --data_root "$DATA_ROOT" \
        --mlp_fractions "$MLP_FRACTIONS" \
        --profile_images "$PROFILE_IMAGES" \
        --eval_images "$EVAL_IMAGES" \
        --out_dir "$OUT_DIR" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    && PASSED[$domain]="ok" \
    || FAILED[$domain]="exit $?"

    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
log "========================================"
log "SUMMARY"
log "========================================"
printf "%-12s %s\n" "DOMAIN" "STATUS"
printf "%-12s %s\n" "------" "------"
for domain in kvasir isic busi pcb; do
    if [ -n "${PASSED[$domain]:-}" ]; then
        printf "%-12s PASS\n" "$domain"
    else
        printf "%-12s FAIL (%s)\n" "$domain" "${FAILED[$domain]:-unknown}"
    fi
done
echo ""

# Print results from JSON outputs if available
if [ "$DRY_RUN" != "true" ]; then
    python3 -c "
import json, os, glob, sys
out = sys.argv[1]
print(f'{\"Domain\":<10} {\"MLP%\":<7} {\"Attn_heads\":<11} {\"MLP/block\":<10} {\"IoU\":<8} {\"Miss%\":<7} {\"Dom\":<5} {\"BG\":<6} {\"RefCov\":<7} {\"FrzCov\":<7} {\"Drop\":<6}')
print('-' * 95)
for d in ['kvasir', 'isic', 'busi', 'pcb']:
    pattern = os.path.join(out, d, 'combined_f*', 'results.json')
    for jf in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(jf))
            mlp_f = r.get('mlp_freeze_fraction', 0)
            attn_h = r.get('attn_heads_frozen', '?')
            mlp_n = r.get('mlp_neurons_frozen_per_block', '?')
            iou = r['mask_comparison']['mean_iou']
            miss = r['mask_comparison']['miss_rate']
            dom = r['miss_decomposition']['domain_misses']
            bg = r['miss_decomposition']['background_misses']
            ref_c = r['coverage_comparison']['ref_mean_coverage']
            frz_c = r['coverage_comparison']['frozen_mean_coverage']
            drop = r['coverage_comparison']['mean_drop']
            print(f'{d:<10} {mlp_f*100:>5.1f}% {attn_h:<11} {mlp_n:<10} {iou:<8.4f} {miss*100:<6.1f}% {dom:<5} {bg:<6} {ref_c*100:<6.1f}% {frz_c*100:<6.1f}% {drop*100:>4.1f}pp')
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
" "$OUT_DIR" 2>/dev/null || true
fi

log "Results: $OUT_DIR"
log "Done."
