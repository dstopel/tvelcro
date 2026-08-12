#!/bin/bash
set -euo pipefail

# ==============================================================================
# DINOv2 ViT-Giant — S-VELCRO Sweep (NEW GROUPS ONLY)
#
# Adds snakes, fish, boats, birds to the existing results database
# that already contains dogs, cats, cars.
#
# Profile + freeze + eval for new groups only, then re-plot ALL groups.
# ==============================================================================

BASE="results_vit_giant_svelcro"
BINS=200
BATCH_SIZE=4
MAX_IMAGES=300
IMG_BASE="image_for_exp"
SYNSET_WORDS="image_for_exp/synset_words.txt"

mkdir -p "${BASE}"
LOG="${BASE}/run_newgroups_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "Log: ${LOG}"
echo "Started: $(date)"
echo

# ==============================================================================
# Phase 1: Profile new groups (adds to existing profile dir)
# ==============================================================================
echo "========================================"
echo "Phase 1: PROFILE (new groups)"
echo "========================================"
python vit_giant_svelcro.py \
  --mode profile \
  --image_dirs ${IMG_BASE}/snakes_profile ${IMG_BASE}/fish_profile ${IMG_BASE}/boats_profile ${IMG_BASE}/birds_profile \
  --group_names snakes,fish,boats,birds \
  --max_images ${MAX_IMAGES} \
  --batch_size ${BATCH_SIZE} \
  --bins ${BINS} \
  --synset_words ${SYNSET_WORDS} \
  --out_dir ${BASE}/profile

# ==============================================================================
# Phase 2: Freeze sweep (new groups only)
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 2: FREEZE SWEEP (new groups)"
echo "========================================"
for FRAC in 0.0 0.09 0.17 0.25 0.34 0.42 0.50 0.59 0.67 0.75 0.84; do
  for GROUP in snakes fish boats birds; do
    echo "--- freeze FRAC=${FRAC} GROUP=${GROUP} ---"
    python vit_giant_svelcro.py \
      --mode freeze \
      --profile_dir ${BASE}/profile \
      --profile_group ${GROUP} \
      --freeze_fraction ${FRAC} \
      --out_dir ${BASE}/frac${FRAC}/freeze_${GROUP}
  done
done

# ==============================================================================
# Phase 3: Eval sweep — each group evaluated ONLY on its own images
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 3: EVAL SWEEP (new groups)"
echo "========================================"
for FRAC in 0.0 0.09 0.17 0.25 0.34 0.42 0.50 0.59 0.67 0.75 0.84; do
  for GROUP in snakes fish boats birds; do
    echo "--- eval FRAC=${FRAC} GROUP=${GROUP} ---"
    python vit_giant_svelcro.py \
      --mode eval \
      --image_dirs ${IMG_BASE}/${GROUP}_eval \
      --group_names ${GROUP} \
      --freeze_manifest ${BASE}/frac${FRAC}/freeze_${GROUP}/freeze_manifest.json \
      --profile_dir ${BASE}/profile \
      --max_images ${MAX_IMAGES} \
      --batch_size ${BATCH_SIZE} \
      --bins ${BINS} \
      --synset_words ${SYNSET_WORDS} \
      --out_dir ${BASE}/frac${FRAC}/eval_${GROUP}
  done
done

# ==============================================================================
# Phase 4: Plot accuracy vs FLOP reduction (ALL groups)
# ==============================================================================
echo ""
echo "========================================"
echo "Phase 4: PLOTS (all groups)"
echo "========================================"
python plot_accuracy_vs_flops.py --results_dir ${BASE}

echo ""
echo "========================================"
echo "ALL DONE: $(date)"
echo "Results: ${BASE}/"
echo "========================================"
