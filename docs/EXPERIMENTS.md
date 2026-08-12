# EXPERIMENTS.md, the autonomous run-book

> **This file is an executable mission for an AI agent (Claude Code).**
> Given this file and the operator config below, the agent connects to the
> RunPod pod, initializes it, runs every experiment the DATE 2027 paper
> needs, verifies each output, collects the results, and stops the pod.
> A human can follow the same steps manually.

**Mission success =** every P0 experiment in Section 4 has a passing
success check, results are pushed to a `results-<date>` branch of
`dstopel/tvelcro`, a final report table is produced, and the pod is stopped.

---

## 0. Operator config (fill in before handing to the agent)

```
RUNPOD_API_KEY = <set as env var or in runpodctl config; never commit>
POD_ID         = <RunPod pod id, e.g. abc123xyz>          # or leave empty to pick the only running pod
GPU_EXPECTED   = A10 or better, 24 GB VRAM minimum
GITHUB_TOKEN   = <fine-grained token with repo read on dstopel/tvelcro>
KAGGLE_USERNAME / KAGGLE_KEY = <for the 5 segmentation datasets>
IMAGENET_DIR   = <path on the pod to an ImageNet val-by-class tree>
```

## 1. Ground rules for the agent

1. **One GPU job at a time.** Run experiments serially inside `tmux`
   (`tmux new -s exp`), never in parallel; SAM eval already uses most of a
   24 GB card.
2. **Verify after every step** using the success checks in Section 4 before
   moving on. Do not mark an experiment done because the script exited 0.
3. **Failure policy:** on failure, retry once; if the failure is CUDA OOM,
   retry with `POINTS_PER_SIDE=16` (SAM) or halved `BATCH_SIZE` (ViT), which
   is the known-good fallback from the thesis runs. After two failures,
   capture the last 100 log lines into `results/_failures/<name>.log` and
   continue with the next experiment. Never loop indefinitely.
4. **Never delete or overwrite** anything under `data/` or an existing
   `results/` subdirectory; scripts are resumable and skip finished work.
5. **Cost control:** the pod is billed per hour. If all remaining work is
   blocked (e.g. missing ImageNet), collect what exists, report, and stop
   the pod anyway.
6. **Secrets** stay in environment variables; never echo them, never commit
   them.

## 2. Connect to the pod

```bash
# on the local machine
runpodctl config --apiKey "$RUNPOD_API_KEY"      # once
runpodctl get pod                                 # find POD_ID and state
runpodctl start pod $POD_ID                       # if stopped
# obtain ssh: RunPod console shows the ssh command per pod, typically:
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

Checks after connecting: `nvidia-smi` shows the expected GPU;
`df -h /workspace` has at least 80 GB free.

## 3. Initialize the pod (idempotent, safe to re-run)

```bash
cd /workspace
[ -d tvelcro ] || git clone https://$GITHUB_TOKEN@github.com/dstopel/tvelcro.git
cd tvelcro && git pull
export KAGGLE_USERNAME=... KAGGLE_KEY=... IMAGENET_DIR=/workspace/imagenet_val
bash setup_pod.sh                     # venv, deps, SAM + MobileSAM ckpts, COCO, kaggle sets
source .venv/bin/activate
export PYTHONPATH=.
```

**Smoke tests (must all pass before any long run):**

```bash
python sam/surgery.py --selftest      # expects: All SAM surgery self-tests passed
python vit/surgery.py --selftest      # expects: All ViT surgery self-tests passed
python vit/create_groups.py --src "$IMAGENET_DIR" --dst data/vit_groups \
    --domains dogs cats cars snakes fish boats birds general \
    --num_profile 300 --num_eval 300 --seed 0
python - <<'EOF'                      # one-batch model-load sanity
import json,subprocess
for cmd in (
  "python vit/vit_velcro.py --mode profile --model vit_base --image_dirs data/vit_groups/dogs_profile --group_names smoke --max_images 4 --out_dir results/_smoke/base",
  "python vit/vit_velcro.py --mode profile --model dinov2_giant --image_dirs data/vit_groups/dogs_profile --group_names smoke --max_images 2 --out_dir results/_smoke/giant",
):
    subprocess.run(cmd, shell=True, check=True)
print("SMOKE OK")
EOF
```

If ImageNet is absent: SAM experiments (Section 4, items 1, 2, 4, 5, 6, 8)
still run; skip ViT items and say so in the final report.

## 4. Execution plan

Run in this order. P0 items are paper-critical; P1 strengthen the paper.
Total GPU time, order of magnitude: 25 to 40 h on an A10.

| # | Pri | Command (from repo root, in tmux) | ETA | Success check | Feeds |
|---|-----|-----------------------------------|-----|---------------|-------|
| 1 | P0 | `bash experiments/sam/exp_main_sweep.sh` | 6-9 h | `results/` sweep dir has `eval_results.json` for all 5 domains x 3 fracs; each parses and has `overall_mean_iou > 0.8` | main SAM table; produces the profiles reused by 2, 4, 5, 8 |
| 2 | P0 | `bash experiments/sam/exp_structured_latency.sh` (DOMAIN=kvasir FRAC=0.25; then MLP TODO note) | <1 h | JSON has baseline/hooked/sliced medians; `sliced < baseline` | measured latency + energy table, the core DATE requirement |
| 3 | P0 | `bash experiments/vit/exp_sweep.sh` | 1-2 h | run_info.json per model x domain x T_k; baseline top-1 within 2 pp of thesis values | main ViT table + accuracy-vs-FLOPs figure |
| 4 | P0 | `bash experiments/sam/exp_general_vs_domain.sh` | 3-5 h | comparison summary JSON exists; domain-profile IoU >= general-profile IoU on majority of domains | domain-specialization ablation |
| 5 | P0 | `bash experiments/sam/exp_random_baseline.sh` | 1-2 h | summary JSON has variance/random/entropy rows x seeds | random-control rows |
| 6 | P0 | `bash experiments/baselines/exp_competitors.sh` | 3-5 h | both JSONs exist; every requested variant has either metrics or a recorded skip reason | baseline comparison table |
| 7 | P0 | `bash experiments/vit/exp_latency.sh` | <1 h | JSON: sliced faster than baseline at batch 1 and 8 | ViT latency rows |
| 8 | P1 | `bash experiments/sam/exp_n_sweep.sh` | 3-4 h | per-N eval JSONs + Spearman table from analyze_profile | N-sensitivity figure |
| 9 | P1 | `bash experiments/vit/exp_n_sweep.sh` | 1-2 h | per-N x seed run_info.json tree | N-sensitivity (ViT) |
| 10 | P1 | `bash experiments/vit/exp_general_vs_domain.sh` | ~1 h | per-domain eval dirs complete | ablation (ViT side) |
| 11 | P1 | `bash experiments/vit/exp_ood_matrix.sh` | 1-2 h | `ood_matrix_top1.csv` is a full matrix, no empty cells | OOD heatmap |
| 12 | P1 | `bash experiments/sam/exp_profile_size.sh` | min | JSON with FP32/FP16/frozen-only sizes per profile | serving table |
| 13 | P1 | ViT random control (no shell): `python vit/vit_velcro.py --mode freeze --rank random --seed 0 ...` per model at one T_k, then eval | ~1 h | manifests record `rank: random` | control row (ViT) |

After each item: append one line to `results/RUNLOG.md`
(`<timestamp> <name> <status> <wall time> <one key number>`).

## 5. Collect results

```bash
cd /workspace/tvelcro
git checkout -b results-$(date +%Y%m%d)
git add -f results/            # results/ is gitignored; -f is intentional here
git commit -m "pod results $(date +%F)"
git push -u origin HEAD
```

Fallback if push fails: `tar czf /workspace/results_$(date +%F).tgz results/`
and transfer with `runpodctl send` or scp, then say where the tarball is.

## 6. Shut down and report

```bash
runpodctl stop pod $POD_ID     # from the local machine; billed until stopped
```

Final report (in the conversation, and saved to `results/FINAL_REPORT.md`):
one table row per experiment (status, wall time, headline number), the
success-check outcomes, any failures with their log paths, measured latency
and energy highlights, and which paper tables are now fully backed by data.

---

## Appendix: what each experiment is for

- **exp_main_sweep (SAM)**: accuracy versus freeze ratio per domain
  (kvasir, isic, busi, pcb, deepglobe); the headline SAM table.
- **exp_sweep (ViT)**: top-1 versus explicit per-block head counts for
  vit_base and dinov2_giant on the 7 ImageNet domains (dogs, cats, cars,
  snakes, fish, boats, birds); distinct T_k points, no fraction collisions.
- **exp_structured_latency (SAM) / exp_latency (ViT)**: baseline vs hooked vs
  sliced (surgery) wall-clock and NVML energy; converts analytical FLOPs
  into measured speedup, the core DATE reviewer requirement.
- **exp_general_vs_domain**: freeze from a general-data profile at equal
  budget; the gap to domain profiles is the evidence that domain
  specialization, not just variance ranking, is the contribution.
- **exp_random_baseline**: same freeze counts chosen at random (3 seeds) and
  by entropy; the gap to variance ranking is the evidence that value
  locality is the signal.
- **exp_competitors**: ToMe on the 7 ImageNet domains at matched FLOPs;
  MobileSAM and SlimSAM on the 5 segmentation domains under the identical
  Hungarian-fidelity + GT-coverage + latency protocol. EfficientSAM stays a
  quoted row (no native automatic mask generation).
- **exp_n_sweep**: profile quality versus N in {25, 50, 100, 200, 300} with
  seeds; backs the "small unlabeled calibration set" claim and the Spearman
  rank-stability figure.
- **exp_ood_matrix (ViT)**: profile from domain i evaluated on domain j;
  quantifies out-of-domain behavior and motivates per-domain profiles.
- **exp_profile_size (SAM)**: on-disk profile size (FP32, FP16, frozen-only)
  and hook attach/detach time; basis of the one-binary-many-profiles serving
  argument.
