#!/bin/bash
set -euo pipefail

# ==============================================================================
# T-VELCRO ViT sweep: profile -> freeze -> eval at explicit, distinct T_k
# operating points (no colliding freeze fractions), both models.
#
# Run from repo root:  bash experiments/vit/exp_sweep.sh
# ==============================================================================

# --- config (override via env) ------------------------------------------------
MODELS="${MODELS:-vit_base dinov2_giant}"
DOMAINS="${DOMAINS:-snakes fish boats birds dogs cats cars}"
IMG_BASE="${IMG_BASE:-image_for_exp}"
SYNSET_WORDS="${SYNSET_WORDS:-${IMG_BASE}/synset_words.txt}"
RESULTS="${RESULTS:-results/vit_sweep}"
SEED="${SEED:-0}"
MAX_IMAGES="${MAX_IMAGES:-300}"
RANK="${RANK:-variance}"
TKS_BASE="${TKS_BASE:-0 2 3 4 6 8}"        # vit_base: H=12
TKS_GIANT="${TKS_GIANT:-0 4 6 8 12 16}"    # dinov2_giant: H=24
BATCH_BASE="${BATCH_BASE:-32}"
BATCH_GIANT="${BATCH_GIANT:-4}"
# ------------------------------------------------------------------------------

export PYTHONPATH=.
mkdir -p "${RESULTS}"
LOG="${RESULTS}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "Log: ${LOG}"

bs()  { if [ "$1" = "dinov2_giant" ]; then echo "${BATCH_GIANT}"; else echo "${BATCH_BASE}"; fi; }
tks() { if [ "$1" = "dinov2_giant" ]; then echo "${TKS_GIANT}"; else echo "${TKS_BASE}"; fi; }

for MODEL in ${MODELS}; do
  ROOT="${RESULTS}/${MODEL}/seed${SEED}"

  PROF_DIRS=""
  GROUP_CSV=""
  for G in ${DOMAINS}; do
    PROF_DIRS="${PROF_DIRS} ${IMG_BASE}/${G}_profile"
    GROUP_CSV="${GROUP_CSV:+${GROUP_CSV},}${G}"
  done

  echo "=== [${MODEL}] Phase 1: PROFILE (${GROUP_CSV}) ==="
  if [ -f "${ROOT}/profile/meta/run_info.json" ]; then
    echo "  profile exists, skipping"
  else
    python3 vit/vit_velcro.py \
      --model "${MODEL}" --mode profile \
      --image_dirs ${PROF_DIRS} \
      --group_names "${GROUP_CSV}" \
      --max_images "${MAX_IMAGES}" --seed "${SEED}" \
      --batch_size "$(bs "${MODEL}")" \
      --synset_words "${SYNSET_WORDS}" \
      --out_dir "${ROOT}/profile"
  fi

  for TK in $(tks "${MODEL}"); do
    for G in ${DOMAINS}; do
      echo "=== [${MODEL}] tk${TK} ${G}: FREEZE + EVAL ==="
      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode freeze \
        --profile_dir "${ROOT}/profile" --profile_group "${G}" \
        --rank "${RANK}" --heads_per_block "${TK}" --seed "${SEED}" \
        --out_dir "${ROOT}/tk${TK}/freeze_${G}"

      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode eval \
        --image_dirs "${IMG_BASE}/${G}_eval" --group_names "${G}" \
        --freeze_manifest "${ROOT}/tk${TK}/freeze_${G}/freeze_manifest.json" \
        --max_images "${MAX_IMAGES}" --seed "${SEED}" \
        --batch_size "$(bs "${MODEL}")" \
        --synset_words "${SYNSET_WORDS}" \
        --out_dir "${ROOT}/tk${TK}/eval_${G}"
    done
  done

  echo "=== [${MODEL}] PLOTS ==="
  python3 vit/plot_results.py --results_dir "${RESULTS}/${MODEL}"
done

echo "DONE: ${RESULTS}/"
