# Baselines

T-VELCRO is compared against prior compression work along two tracks: a
measured head-to-head where a fair one exists (ToMe on ViT), and quoted
numbers where the methods target a different axis and retraining them under
our protocol would be neither feasible nor informative (SAM baselines).

## (a) ToMe: measured head-to-head on ViT

ToMe (Bolya et al., ICLR 2023) merges r tokens per transformer block, so it
shrinks the sequence length S. T-VELCRO keeps S constant and freezes computed
units (attention heads, MLP slices). Neither requires training, which makes
this the one directly comparable baseline.

Matched-FLOP protocol:

1. Compute the analytical MAC reduction of ToMe over the 12 ViT-B/16 blocks
   as a function of r (patch embed and head excluded on both sides).
2. Pick the r values whose reduction matches the T-VELCRO operating points
   (25% and 41.7%).
3. Evaluate top-1 on the identical per-domain eval dirs produced by
   vit/create_groups.py, and measure latency with the identical
   common.timing.measure_latency harness.

Run from repo root:

```bash
PYTHONPATH=. python baselines/tome_comparison.py \
    --data_dir ./data/vit_groups --domains snakes fish boats birds
```

Output: baselines/results/tome_comparison.json (per r: analytical reduction,
latency, per-domain top-1).

## (b) SAM baselines: quoted numbers

These methods compress SAM along axes orthogonal to ours, and all except
quantization require training. We quote their published numbers rather than
re-run them. Fill this table from the cited papers:

| Method | Venue | Params | FLOPs / speedup | mIoU delta | Training cost | Reversible | Domain-specialized |
|---|---|---|---|---|---|---|---|
| PTQ4SAM (quantization) | CVPR 2024 | quote | quote | quote | calibration only | no (weights rewritten) | no |
| SlimSAM (prune + distill) | NeurIPS 2024 | quote | quote | quote | quote (GPU-hours, data) | no | no |
| MobileSAM (distillation) | arXiv 2023 | quote | quote | quote | quote | no | no |
| EfficientSAM (distillation) | CVPR 2024 | quote | quote | quote | quote | no | no |
| T-VELCRO (ours) | | 0 new | measured (experiments/sam) | measured | none (profiling only) | yes (masks, not weights) | yes (per-domain profiles) |

## Why these are orthogonal axes

Quantization changes numeric precision; pruning and distillation produce a
new, smaller, permanently altered model at significant training cost.
T-VELCRO changes which units are computed, per domain, with no training and
full reversibility: dropping the profile restores the original model
bit-exactly. The techniques compose rather than compete; a quantized or
distilled SAM can still be value-locality frozen. The paper argument is
therefore not "T-VELCRO beats distillation at equal FLOPs" but "T-VELCRO adds
a training-free, domain-aware, reversible operating point that no method in
this table offers", with ToMe as the fair matched-FLOP measured comparison on
the ViT side.
