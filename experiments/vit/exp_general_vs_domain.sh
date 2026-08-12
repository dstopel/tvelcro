#!/bin/bash
set -euo pipefail

# ==============================================================================
# T-VELCRO general-vs-domain: profile the 'general' group, freeze from it,
# then evaluate that general manifest on each specialized domain.
# Output layout matches vit/plot_results.py (eval_<domain> per tk<K>).
#
# Run from repo root:  bash experiments/vit/exp_general_vs_domain.sh
# ==============================================================================

# --- config (override via env) ------------------------------------------------
MODELS="${MODELS:-vit_base dinov2_giant}"
DOMAINS="${DOMAINS:-snakes fish boats birds dogs cats cars}"
PROFILE_GROUP="${PROFILE_GROUP:-general}"
IMG_BASE="${IMG_BASE:-image_for_exp}"
SYNSET_WORDS="${SYNSET_WORDS:-${IMG_BASE}/synset_words.txt}"
RESULTS="${RESULTS:-results/vit_general_vs_domain}"
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

  echo "=== [${MODEL}] PROFILE '${PROFILE_GROUP}' ==="
  if [ -f "${ROOT}/profile/meta/run_info.json" ]; then
    echo "  profile exists, skipping"
  else
    python3 vit/vit_velcro.py \
      --model "${MODEL}" --mode profile \
      --image_dirs "${IMG_BASE}/${PROFILE_GROUP}_profile" \
      --group_names "${PROFILE_GROUP}" \
      --max_images "${MAX_IMAGES}" --seed "${SEED}" \
      --batch_size "$(bs "${MODEL}")" \
      --synset_words "${SYNSET_WORDS}" \
      --out_dir "${ROOT}/profile"
  fi

  for TK in $(tks "${MODEL}"); do
    echo "=== [${MODEL}] tk${TK}: FREEZE from '${PROFILE_GROUP}' ==="
    python3 vit/vit_velcro.py \
      --model "${MODEL}" --mode freeze \
      --profile_dir "${ROOT}/profile" --profile_group "${PROFILE_GROUP}" \
      --rank "${RANK}" --heads_per_block "${TK}" --seed "${SEED}" \
      --out_dir "${ROOT}/tk${TK}/freeze_${PROFILE_GROUP}"

    for G in ${DOMAINS}; do
      echo "=== [${MODEL}] tk${TK}: EVAL '${PROFILE_GROUP}' manifest on '${G}' ==="
      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode eval \
        --image_dirs "${IMG_BASE}/${G}_eval" --group_names "${G}" \
        --freeze_manifest "${ROOT}/tk${TK}/freeze_${PROFILE_GROUP}/freeze_manifest.json" \
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
