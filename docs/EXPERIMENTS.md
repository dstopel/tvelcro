# Experiments

All commands run from repo root with PYTHONPATH=. and data under DATA_ROOT
(default ./data). Every experiment exists for both models; substitute sam or
vit in the path. ETAs are order-of-magnitude on an A10-class GPU (SAM ViT-H
is the slow one; the ViT variants are roughly 5 to 10x faster). Outputs land
under results/ in a subdirectory named after the script; see each script
header for exact paths.

## main_sweep

Purpose: the headline result, accuracy (mIoU for SAM, top-1 for ViT) versus
freeze ratio / FLOP reduction, per domain.

```bash
PYTHONPATH=. bash experiments/sam/main_sweep.sh
PYTHONPATH=. bash experiments/vit/main_sweep.sh
```

Outputs: results/{sam,vit}/main_sweep/ (per-domain profiles plus a sweep
summary JSON/CSV).
ETA: hours for SAM (profiling plus eval across domains and ratios), about an
hour for ViT.
Feeds: main results table and the accuracy-vs-FLOPs figure.

## general_vs_domain

Purpose: ablation showing a domain-specific profile beats a general
(mixed-domain) profile at equal FLOP reduction.

```bash
PYTHONPATH=. bash experiments/sam/general_vs_domain.sh
PYTHONPATH=. bash experiments/vit/general_vs_domain.sh
```

Outputs: results/{sam,vit}/general_vs_domain/.
ETA: hours (SAM), tens of minutes (ViT).
Feeds: domain-specialization ablation table.

## n_sweep

Purpose: sensitivity of profile quality to the number of profiling images N;
shows small N suffices (cheap deployment story).

```bash
PYTHONPATH=. bash experiments/sam/n_sweep.sh
PYTHONPATH=. bash experiments/vit/n_sweep.sh
```

Outputs: results/{sam,vit}/n_sweep/.
ETA: hours (SAM), about an hour (ViT).
Feeds: N-sensitivity figure (accuracy vs N).

## random_baseline

Purpose: control, freezing the same unit count at random instead of by value
locality; the gap is the evidence that locality is the signal.

```bash
PYTHONPATH=. bash experiments/sam/random_baseline.sh
PYTHONPATH=. bash experiments/vit/random_baseline.sh
```

Outputs: results/{sam,vit}/random_baseline/.
ETA: about an hour each (few ratios, several seeds).
Feeds: random-control rows in the main table.

## profile_size

Purpose: measure the on-disk size of a per-domain profile (mask plus cached
values), the basis of the multi-tenant serving argument (one backbone, many
tiny domain profiles, bit-exact reversibility).

```bash
PYTHONPATH=. bash experiments/sam/profile_size.sh
PYTHONPATH=. bash experiments/vit/profile_size.sh
```

Outputs: results/{sam,vit}/profile_size/.
ETA: minutes (no eval, just profile accounting).
Feeds: profile-size table and the serving/discussion section.

## structured_latency

Purpose: measured wall-clock latency and energy of structured removal
(surgery.py) via common.timing.measure_latency and EnergyMeter; converts
analytical FLOP reductions into measured speedup and Joules.

```bash
PYTHONPATH=. bash experiments/sam/structured_latency.sh
PYTHONPATH=. bash experiments/vit/structured_latency.sh
```

Outputs: results/{sam,vit}/structured_latency/.
ETA: tens of minutes (bench loops at several operating points).
Feeds: measured latency and energy table, the core DATE requirement.

## ood_matrix

Purpose: full cross-domain matrix, profile from domain i evaluated on domain
j; quantifies what breaks out of domain and motivates per-domain profiles.

```bash
PYTHONPATH=. bash experiments/sam/ood_matrix.sh
PYTHONPATH=. bash experiments/vit/ood_matrix.sh
```

Outputs: results/{sam,vit}/ood_matrix/.
ETA: hours (N x N domain pairs; SAM is the long pole).
Feeds: OOD heatmap figure.

## ToMe baseline (ViT only)

Purpose: measured matched-FLOP comparison against ToMe token merging; see
baselines/README.md for the protocol.

```bash
PYTHONPATH=. python baselines/tome_comparison.py \
    --data_dir ./data/vit_groups --domains snakes fish boats birds
```

Outputs: baselines/results/tome_comparison.json.
ETA: tens of minutes.
Feeds: baseline-comparison table (alongside the quoted SAM baselines).
