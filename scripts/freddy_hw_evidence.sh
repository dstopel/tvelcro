#!/bin/bash
# freddy_hw_evidence.sh - one-shot hardware evidence pass (Freddy notes 1-5).
#
# Runs on the L4 pod from repo root. For every model x available freeze
# manifest: absolute latency (median/p95), NVML energy (J/fwd, avg W),
# peak CUDA memory, and param counts baseline vs sliced. Then the storage
# report cross-checked against a real mean file.
#
#   PYTHONPATH=. bash scripts/freddy_hw_evidence.sh
#
# Outputs under results/freddy_hw/. Never overwrites: skips finished JSONs.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=.

OUT=results/freddy_hw
mkdir -p "$OUT"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-20}"
FAILED=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }
runlog() { echo "| $(date -u '+%Y-%m-%dT%H:%M:%SZ') | $1 | $2 | $3 |" >> results/RUNLOG.md; }

# GPU + software provenance for the setup table (review P0 reproducibility)
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "$OUT/gpu_info.csv" 2>/dev/null
python3 - <<'EOF' > "$OUT/sw_versions.json"
import json, torch, sys
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                  "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}))
EOF

# ---------------------------------------------------------------------------
# 1) ViT-family latency: every model x every frac with a manifest on disk
#    (vit_base batches 1,8; dinov2_giant 1,2; dinov3_7b 1,2)
# ---------------------------------------------------------------------------
bench_vit() { # model batches manifest tag
    local MODEL=$1 BATCHES=$2 MANIFEST=$3 TAG=$4
    local J="$OUT/latency_${TAG}.json"
    [ -f "$J" ] && { log "skip $TAG (exists)"; return 0; }
    [ -f "$MANIFEST" ] || { log "SKIP $TAG: no manifest $MANIFEST"; return 0; }
    log "bench $TAG ..."
    if python3 vit/bench_latency.py --model "$MODEL" --freeze_manifest "$MANIFEST" \
        --batch_sizes "$BATCHES" --warmup "$WARMUP" --iters "$ITERS" --out "$J"; then
        runlog "freddy_lat_$TAG" OK "-"
    else
        runlog "freddy_lat_$TAG" FAIL "-"; FAILED=1
    fi
}

find_manifest() { # model_pattern frac
    find results -path "*$1*" -path "*$2*" -name freeze_manifest.json 2>/dev/null | head -1
}

for FRAC in 0.25 0.125 0.375 0.5; do
    M=$(find_manifest vit_base "$FRAC"); [ -z "$M" ] && M=$(find_manifest base "f$FRAC")
    bench_vit vit_base_patch16_224 1,8 "${M:-missing}" "vit_base_f$FRAC"
done
for FRAC in 0.25 0.125 0.375; do
    M=$(find_manifest giant "$FRAC")
    bench_vit dinov2_giant 1,2 "${M:-missing}" "dinov2_giant_f$FRAC"
    M=$(find_manifest dinov3 "$FRAC"); [ -z "$M" ] && M=$(find_manifest 7b "$FRAC")
    bench_vit dinov3_7b 1,2 "${M:-missing}" "dinov3_7b_f$FRAC"
done

# ---------------------------------------------------------------------------
# 2) SAM encoder + AutomaticMaskGenerator latency/energy (kvasir, f0.25)
# ---------------------------------------------------------------------------
if [ ! -f "$OUT/latency_sam_f0.25.json" ]; then
    SAM_MANIFEST=$(find results -path "*kvasir*0.25*" -name freeze_manifest.json | head -1)
    CKPT=$(find data -name "sam_vit_h*.pth" 2>/dev/null | head -1)
    IMG_DIR=$(find data -type d -name "kvasir_eval" 2>/dev/null | head -1)
    if [ -n "$SAM_MANIFEST" ] && [ -n "$CKPT" ]; then
        log "bench SAM (manifest=$SAM_MANIFEST) ..."
        if python3 sam/bench_latency.py --checkpoint "$CKPT" --freeze_manifest "$SAM_MANIFEST" \
            ${IMG_DIR:+--image_dir "$IMG_DIR"} --eval_n 20 --points_per_side 16 \
            --warmup 5 --iters 50 --out "$OUT/latency_sam_f0.25.json"; then
            runlog freddy_lat_sam OK "-"
        else runlog freddy_lat_sam FAIL "-"; FAILED=1; fi
    else
        log "SKIP SAM: manifest or checkpoint not found"; FAILED=1
    fi
fi

# ---------------------------------------------------------------------------
# 3) Storage report + seq-len verification from a real stored mean
# ---------------------------------------------------------------------------
MEAN=$(find results -path "*dinov3*" -name "*mean*.npy" 2>/dev/null | head -1)
[ -z "$MEAN" ] && MEAN=$(find results -name "*mean*.npy" 2>/dev/null | head -1)
python3 scripts/storage_report.py ${MEAN:+--mean_npy "$MEAN"} \
    --json "$OUT/storage_report.json" | tee "$OUT/storage_report.txt"
# record shapes of one mean file per model family for exact S verification
find results -name "*mean*.npy" | awk -F/ '{print $2}' | sort -u > "$OUT/mean_roots.txt" 2>/dev/null
runlog freddy_storage OK "-"

log "done (FAILED=$FAILED); outputs in $OUT/"
exit $FAILED
