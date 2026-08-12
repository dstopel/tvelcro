#!/bin/bash
set -uo pipefail
cd /workspace/tvelcro
source .venv/bin/activate
source /workspace/.secrets.env
export PYTHONPATH=. DATA_ROOT=./data
mkdir -p results/_logs results/_failures
RUNLOG=results/RUNLOG.md
run() {
  local name=$1; shift
  local t0=$(date +%s)
  echo "=== $name start $(date -u +%FT%TZ) ==="
  if "$@" > results/_logs/$name.log 2>&1; then st=OK
  else st=FAIL; tail -100 results/_logs/$name.log > results/_failures/$name.log; fi
  echo "| $(date -u +%FT%TZ) | $name | $st | $(( ($(date +%s)-t0)/60 ))m |" >> $RUNLOG
  echo "=== $name $st ==="
}
run smoke_sam_surgery python sam/surgery.py --selftest
run smoke_vit_surgery python vit/surgery.py --selftest
run sam_main_sweep bash experiments/sam/exp_main_sweep.sh
run sam_structured_latency bash experiments/sam/exp_structured_latency.sh
run sam_general_vs_domain bash experiments/sam/exp_general_vs_domain.sh
run sam_random_baseline bash experiments/sam/exp_random_baseline.sh
run sam_profile_size bash experiments/sam/exp_profile_size.sh
run sam_competitors python baselines/sam_competitors.py --variants mobile_sam slimsam_50 --max_images 50
run sam_n_sweep bash experiments/sam/exp_n_sweep.sh
# ViT stage: retry gated ImageNet once; skip cleanly if still locked
if [ ! -d /workspace/imagenet_val/n01440764 ]; then
  python /workspace/get_imagenet.py > /workspace/inet.log 2>&1 || true
fi
if [ -d /workspace/imagenet_val/n01440764 ]; then
  run vit_groups python vit/create_groups.py --src /workspace/imagenet_val --dst data/vit_groups --domains dogs cats cars snakes fish boats birds general --num_profile 300 --num_eval 300 --seed 0
  run vit_sweep bash experiments/vit/exp_sweep.sh
  run vit_latency bash experiments/vit/exp_latency.sh
  run vit_general_vs_domain bash experiments/vit/exp_general_vs_domain.sh
  run tome python baselines/tome_comparison.py --data_dir data/vit_groups --out results/baselines/tome_comparison.json
  run vit_ood bash experiments/vit/exp_ood_matrix.sh
  run vit_n_sweep bash experiments/vit/exp_n_sweep.sh
else
  echo "| $(date -u +%FT%TZ) | vit_stage | SKIPPED imagenet gated | - |" >> $RUNLOG
fi
# collect results
git checkout -b results-$(date +%Y%m%d-%H%M) || true
git add -f results/ && git commit -m "pod results $(date -u +%F)" && git push -u origin HEAD || echo PUSH_FAILED >> $RUNLOG
# self-stop the pod
curl -s -X POST "https://rest.runpod.io/v1/pods/t4qfb3imjuk578/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >> /workspace/stop.log 2>&1
