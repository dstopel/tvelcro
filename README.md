# T-VELCRO

T-VELCRO is a training-free, domain-specialized, reversible compression method
for ViTs and SAM based on value locality: units whose activations are nearly
constant within a target domain are profiled once and frozen to their cached
values at inference. Compression is a per-domain profile (a mask plus cached
values), not a new model, so it costs zero training, specializes to each
deployment domain, and is removed bit-exactly by dropping the profile.
Target venue: DATE 2027, with measured latency and energy plus matched-FLOP
baseline comparisons.

## Repo map

| Path | Contents |
|---|---|
| common/ | shared timing and energy harness (measure_latency, EnergyMeter) |
| sam/ | SAM ViT-H pipeline: sam_velcro.py (profile/freeze/eval), surgery.py, bench_latency.py |
| vit/ | ViT pipeline: vit_velcro.py --model {vit_base,dinov2_giant}, create_groups.py, surgery.py, bench_latency.py |
| experiments/sam/, experiments/vit/ | one shell script per paper experiment |
| baselines/ | ToMe measured baseline + quoted SAM baseline protocol |
| docs/ | EXPERIMENTS.md: per-experiment commands, outputs, ETAs |

Conventions: run everything from repo root with PYTHONPATH=. and data under
DATA_ROOT (default ./data).

## Quickstart (RunPod)

```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...   # optional: SAM domain datasets
export IMAGENET_DIR=/path/to/val_by_class   # required for the ViT path
bash setup_pod.sh
source .venv/bin/activate

# SAM path (data auto-downloaded)
PYTHONPATH=. bash experiments/sam/main_sweep.sh

# ViT path (build domain groups first)
PYTHONPATH=. python vit/create_groups.py --src $IMAGENET_DIR --dst ./data/vit_groups
PYTHONPATH=. bash experiments/vit/main_sweep.sh
```

Then run the remaining experiments in the order of the table below; details
and ETAs in docs/EXPERIMENTS.md.

## Experiment to paper mapping

| Experiment | DATE reviewer requirement it answers |
|---|---|
| experiments/{sam,vit}/main_sweep.sh | headline accuracy vs FLOP-reduction tradeoff |
| experiments/{sam,vit}/structured_latency.sh | measured latency and energy, not just analytical FLOPs |
| experiments/{sam,vit}/general_vs_domain.sh | domain specialization matters (domain vs general profile) |
| experiments/{sam,vit}/random_baseline.sh | random control: value locality beats random freezing |
| experiments/{sam,vit}/n_sweep.sh | sensitivity to profile-set size N |
| experiments/{sam,vit}/ood_matrix.sh | robustness: cross-domain (OOD) profile transfer matrix |
| experiments/{sam,vit}/profile_size.sh | profile artifact size; multi-tenant serving story |
| baselines/tome_comparison.py | matched-FLOP measured baseline (ToMe) on ViT |
| baselines/README.md quoted table | positioning vs PTQ4SAM / SlimSAM / MobileSAM / EfficientSAM |

## Status

| Item | Status |
|---|---|
| common/ harness, sam/ and vit/ pipelines, experiment scripts | code done |
| baselines/tome_comparison.py | code done |
| setup_pod.sh, docs | done |
| all measured results (sweeps, latency, energy, ToMe, OOD) | needs GPU run |
| quoted-baseline table values | needs literature pass |
