#!/usr/bin/env bash
# Competitor algorithms on the SAME thesis datasets as T-VELCRO.
#   ViT side: ToMe (token merging) on the 7 ImageNet domains at matched FLOPs.
#   SAM side: MobileSAM + SlimSAM on the 5 segmentation domains, Hungarian
#             fidelity vs SAM ViT-H, GT coverage, measured latency.
# Run from repo root. ETA on A10: ToMe ~1 h; SAM competitors ~2-4 h at N=50.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
OUT_ROOT="${OUT_ROOT:-./results/baselines}"
VIT_GROUPS="${VIT_GROUPS:-$DATA_ROOT/vit_groups}"
VIT_DOMAINS="${VIT_DOMAINS:-dogs cats cars snakes fish boats birds}"
SAM_DOMAINS="${SAM_DOMAINS:-kvasir isic busi pcb deepglobe}"
SAM_VARIANTS="${SAM_VARIANTS:-mobile_sam slimsam_50}"
MAX_IMAGES="${MAX_IMAGES:-50}"

export PYTHONPATH=.

echo "=== ToMe on ViT thesis domains ==="
python baselines/tome_comparison.py \
    --data_dir "$VIT_GROUPS" \
    --domains $VIT_DOMAINS \
    --out "$OUT_ROOT/tome_comparison.json"

echo "=== SAM competitors on segmentation thesis domains ==="
python baselines/sam_competitors.py \
    --variants $SAM_VARIANTS \
    --domains $SAM_DOMAINS \
    --data_root "$DATA_ROOT" \
    --max_images "$MAX_IMAGES" \
    --out "$OUT_ROOT/sam_competitors"

echo "Done. Results under $OUT_ROOT/"
