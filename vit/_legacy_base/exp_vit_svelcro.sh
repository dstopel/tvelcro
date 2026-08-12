#!/bin/bash
set -euo pipefail

# ==============================================================================
# ViT Value Locality Experiment — S-VELCRO Channel-Rank Method
#
# Key difference from vit/exp_vit.sh:
#   - Freeze uses ABSOLUTE channel rank (sum of variance per head)
#   - No reference group needed for freezing
#   - Parameter: --freeze_fraction (fraction of heads to freeze per block)
#
# Model: vit_base_patch16_224
# Groups: dogs, cats, cars (specialized)
# Data: ILSVRC2012 organized under image_dirs/
#
# Output:
#   results_vit_svelcro/
#     profile/
#     frac0.25/  frac0.33/  frac0.50/
#       freeze_dogs/  freeze_cats/  freeze_cars/
#       eval_dogs/    eval_cats/    eval_cars/
#       images_dogs/  images_cats/  images_cars/
# ==============================================================================

BASE="results_vit_svelcro"
BINS=200
BATCHES=20
IMG_BASE="image_for_exp"
GROUPS="dogs cats cars"
SYNSET_WORDS="image_for_exp/synset_words.txt"

mkdir -p "${BASE}"
LOG="${BASE}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "Log: ${LOG}"
echo "Started: $(date)"
echo

# ==============================================================================
# Phase 1: Profile (all groups — NO general reference needed for S-VELCRO freeze,
#          but we still profile general for eval comparison plots)
# ==============================================================================
python vit_value_locality_svelcro.py \
  --mode profile \
  --image_dirs ${IMG_BASE}/dogs_profile ${IMG_BASE}/cats_profile ${IMG_BASE}/cars_profile ${IMG_BASE}/general_profile \
  --group_names dogs,cats,cars,general \
  --max_batches ${BATCHES} \
  --bins ${BINS} \
  --synset_words ${SYNSET_WORDS} \
  --out_dir ${BASE}/profile

# ==============================================================================
# Phase 2: Freeze (sweep freeze fractions — S-VELCRO channel rank)
# ==============================================================================
for FRAC in 0.25 0.33 0.50; do
  for GROUP in dogs cats cars; do
    python vit_value_locality_svelcro.py \
      --mode freeze \
      --profile_dir ${BASE}/profile \
      --profile_group ${GROUP} \
      --freeze_fraction ${FRAC} \
      --out_dir ${BASE}/frac${FRAC}/freeze_${GROUP}
  done
done

# ==============================================================================
# Phase 3: Eval (mean-head network + accuracy)
# ==============================================================================
for FRAC in 0.25 0.33 0.50; do
  for GROUP in dogs cats cars; do
    python vit_value_locality_svelcro.py \
      --mode eval \
      --image_dirs ${IMG_BASE}/dogs_eval ${IMG_BASE}/cats_eval ${IMG_BASE}/cars_eval ${IMG_BASE}/general_eval \
      --group_names dogs,cats,cars,general \
      --freeze_manifest ${BASE}/frac${FRAC}/freeze_${GROUP}/freeze_manifest.json \
      --profile_dir ${BASE}/profile \
      --max_batches ${BATCHES} \
      --bins ${BINS} \
      --synset_words ${SYNSET_WORDS} \
      --out_dir ${BASE}/frac${FRAC}/eval_${GROUP}
  done
done

# ==============================================================================
# Phase 4: Classification Comparison Images (standard vs mean-head)
# ==============================================================================
for FRAC in 0.25 0.33 0.50; do
  for GROUP in dogs cats cars; do
    python generate_vit_comparison.py \
      --freeze_manifest ${BASE}/frac${FRAC}/freeze_${GROUP}/freeze_manifest.json \
      --image_dir ${IMG_BASE}/${GROUP}_eval \
      --group_name ${GROUP} \
      --num_images 150 \
      --synset_words ${SYNSET_WORDS} \
      --out_dir ${BASE}/frac${FRAC}/images_${GROUP}
  done
done
