#!/bin/bash
set -euo pipefail

# ==============================================================================
# T-VELCRO profiling-set-size sweep: N in {25,50,100,200,300} profile images,
# fixed operating point, multiple seeds. Seed only changes which profile subset
# is drawn; the eval pass uses the full eval dir at every N.
#
# Layout: RESULTS/<model>/N<n>/seed<s>/{profile,tk<K>/{freeze_,eval_}<domain>}
# so `plot_results.py --results_dir RESULTS/<model>/N<n>` aggregates seeds
# (mean line + min-max band).
#
# Run from repo root:  bash experiments/vit/exp_n_sweep.sh
# ==============================================================================

# --- config (override via env) ------------------------------------------------
MODELS="${MODELS:-dinov3_7b vit_base}"
DOMAIN="${DOMAIN:-birds}"
NS="${NS:-25 50 100 200 300}"
SEEDS="${SEEDS:-0 1 2}"
IMG_BASE="${IMG_BASE:-image_for_exp}"
SYNSET_WORDS="${SYNSET_WORDS:-${IMG_BASE}/synset_words.txt}"
RESULTS="${RESULTS:-results/vit_n_sweep}"
RANK="${RANK:-variance}"
TK_BASE="${TK_BASE:-3}"       # vit_base fixed operating point (H=12)
TK_7B="${TK_7B:-8}"           # dinov3_7b fixed operating point (H=32)
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-300}"
BATCH_BASE="${BATCH_BASE:-32}"
BATCH_7B="${BATCH_7B:-2}"
# ------------------------------------------------------------------------------

export PYTHONPATH=.
mkdir -p "${RESULTS}"
LOG="${RESULTS}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
echo "Log: ${LOG}"

bs() { if [ "$1" = "dinov3_7b" ]; then echo "${BATCH_7B}"; else echo "${BATCH_BASE}"; fi; }
tk() { if [ "$1" = "dinov3_7b" ]; then echo "${TK_7B}"; else echo "${TK_BASE}"; fi; }

for MODEL in ${MODELS}; do
  TK="$(tk "${MODEL}")"
  for N in ${NS}; do
    for SEED in ${SEEDS}; do
      ROOT="${RESULTS}/${MODEL}/N${N}/seed${SEED}"
      echo "=== [${MODEL}] N=${N} seed=${SEED} tk${TK} (${DOMAIN}) ==="

      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode profile \
        --image_dirs "${IMG_BASE}/${DOMAIN}_profile" --group_names "${DOMAIN}" \
        --max_images "${N}" --seed "${SEED}" \
        --batch_size "$(bs "${MODEL}")" \
        --synset_words "${SYNSET_WORDS}" \
        --out_dir "${ROOT}/profile"

      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode freeze \
        --profile_dir "${ROOT}/profile" --profile_group "${DOMAIN}" \
        --rank "${RANK}" --heads_per_block "${TK}" --seed "${SEED}" \
        --out_dir "${ROOT}/tk${TK}/freeze_${DOMAIN}"

      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode eval \
        --image_dirs "${IMG_BASE}/${DOMAIN}_eval" --group_names "${DOMAIN}" \
        --freeze_manifest "${ROOT}/tk${TK}/freeze_${DOMAIN}/freeze_manifest.json" \
        --max_images "${EVAL_MAX_IMAGES}" --seed 0 \
        --batch_size "$(bs "${MODEL}")" \
        --synset_words "${SYNSET_WORDS}" \
        --out_dir "${ROOT}/tk${TK}/eval_${DOMAIN}"
    done

    echo "=== [${MODEL}] N=${N}: PLOTS (seed aggregation) ==="
    python3 vit/plot_results.py --results_dir "${RESULTS}/${MODEL}/N${N}"
  done
done

echo "DONE: ${RESULTS}/"
