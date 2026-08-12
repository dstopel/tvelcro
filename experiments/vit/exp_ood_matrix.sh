#!/bin/bash
set -euo pipefail

# ==============================================================================
# T-VELCRO cross-domain (OOD) matrix: every domain's freeze manifest evaluated
# on every domain's eval dir, both models, at one fixed operating point.
#
# Output: RESULTS/<model>/seed<S>/tk<K>/eval_<src>_on_<dst>/  plus an assembled
# ood_matrix_top1.csv (rows = profile/manifest domain, cols = eval domain).
#
# Run from repo root:  bash experiments/vit/exp_ood_matrix.sh
# ==============================================================================

# --- config (override via env) ------------------------------------------------
MODELS="${MODELS:-vit_base dinov3_7b}"
DOMAINS="${DOMAINS:-snakes fish boats birds dogs cats cars}"
IMG_BASE="${IMG_BASE:-image_for_exp}"
SYNSET_WORDS="${SYNSET_WORDS:-${IMG_BASE}/synset_words.txt}"
RESULTS="${RESULTS:-results/vit_ood_matrix}"
SEED="${SEED:-0}"
MAX_IMAGES="${MAX_IMAGES:-300}"
RANK="${RANK:-variance}"
TK_BASE="${TK_BASE:-3}"       # vit_base fixed operating point (H=12)
TK_7B="${TK_7B:-8}"           # dinov3_7b fixed operating point (H=32)
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
  ROOT="${RESULTS}/${MODEL}/seed${SEED}"

  PROF_DIRS=""
  GROUP_CSV=""
  for G in ${DOMAINS}; do
    PROF_DIRS="${PROF_DIRS} ${IMG_BASE}/${G}_profile"
    GROUP_CSV="${GROUP_CSV:+${GROUP_CSV},}${G}"
  done

  echo "=== [${MODEL}] PROFILE all domains ==="
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

  for SRC in ${DOMAINS}; do
    echo "=== [${MODEL}] tk${TK}: FREEZE from '${SRC}' ==="
    python3 vit/vit_velcro.py \
      --model "${MODEL}" --mode freeze \
      --profile_dir "${ROOT}/profile" --profile_group "${SRC}" \
      --rank "${RANK}" --heads_per_block "${TK}" --seed "${SEED}" \
      --out_dir "${ROOT}/tk${TK}/freeze_${SRC}"

    for DSTG in ${DOMAINS}; do
      echo "=== [${MODEL}] tk${TK}: EVAL '${SRC}' manifest on '${DSTG}' ==="
      python3 vit/vit_velcro.py \
        --model "${MODEL}" --mode eval \
        --image_dirs "${IMG_BASE}/${DSTG}_eval" --group_names "${DSTG}" \
        --freeze_manifest "${ROOT}/tk${TK}/freeze_${SRC}/freeze_manifest.json" \
        --max_images "${MAX_IMAGES}" --seed "${SEED}" \
        --batch_size "$(bs "${MODEL}")" \
        --synset_words "${SYNSET_WORDS}" \
        --out_dir "${ROOT}/tk${TK}/eval_${SRC}_on_${DSTG}"
    done
  done

  echo "=== [${MODEL}] ASSEMBLE MATRIX ==="
  TKDIR="${ROOT}/tk${TK}" DOMS="${DOMAINS}" python3 - <<'PYEOF'
import csv, json, os
from pathlib import Path

tkdir = Path(os.environ["TKDIR"])
domains = os.environ["DOMS"].split()
out = tkdir / "ood_matrix_top1.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["profile\\eval"] + domains)
    for src in domains:
        row = [src]
        for dst in domains:
            ri = tkdir / f"eval_{src}_on_{dst}" / "meta" / "run_info.json"
            if ri.exists():
                with open(ri) as g:
                    acc = json.load(g)["accuracy"]
                row.append(acc.get(dst, {}).get("top1", ""))
            else:
                row.append("")
        w.writerow(row)
print(f"  Saved {out}")
PYEOF
done

echo "DONE: ${RESULTS}/"
