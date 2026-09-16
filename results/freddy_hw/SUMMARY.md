# Hardware evidence summary (NVIDIA L4, torch 2.13.0+cu126, batch 1, medians)

| Model | Frac | Speedup | Peak GPU mem | Net disk (FP16) | Energy/fwd |
|---|---|---|---|---|---|
| ViT-Base (FP32) | 0.167 | 1.04x | 347->338 MB | 4 MB | ~0% (noise) |
| ViT-Base (FP32) | 0.25 | **1.10x** | 347->331 MB | 13 MB | -4% |
| ViT-Base (FP32) | 0.333 | 1.08x | 347->323 MB | 17 MB | -13% |
| ViT-Base (FP32) | 0.50 | 1.17x (1.23x @b8) | 347->310 MB | 25 MB | -13% |
| DINOv3-7B (FP16) | 0.125 | 1.07x | 12.85->12.29 GB | 632 MB | -7% |
| DINOv3-7B (FP16) | 0.25 | **1.11x** | 12.85->11.64 GB | 1.27 GB | -11% |
| DINOv3-7B (FP16) | 0.375 | 1.17x | 12.85->10.99 GB | 1.90 GB | -13% |
| DINOv3-7B (FP16) | 0.50 | 1.20x | 12.85->10.36 GB | 2.53 GB | -17% |
| SAM ViT-H encoder (FP32) | 0.25 | **1.13x** | 5.74->4.88 GB | 87 MB | -9% |
| SAM full mask-gen (FP32) | 0.25 | 1.09x | (pipeline) | - | -5.5% |

Notes:
- Net disk = removed weights minus the stored mean constant tensors (Freddy note 4),
  both FP16; ViT-Base/SAM peak-mem deltas are FP32 (2x the FP16 disk figure).
- GPU pinned at its 72 W cap in every config, so energy saving tracks latency saving.
- ViT-Base J/fwd is ~0.5 J; the 0.167 row's delta is within NVML sampling noise.
- All speedups >5 sigma vs run-to-run std. DINOv2-Giant absent: no profiles on volume.
- Raw JSONs beside this file; provenance in gpu_info.csv / sw_versions.json.

# FP16 addendum (2026-09-05, same L4, f=0.25)

| Model | Speedup (b1) | Energy/fwd | Peak mem | Accuracy (fp16 base -> fp16 T-VELCRO) |
|---|---|---|---|---|
| ViT-Base fp16 | 1.41x (1.29x @b8) | 0.16->0.09 J (-44%) | 189->179 MB | dogs 82.7->77.7, cats 81.7->79.0, cars 85.7->83.7 (= fp32 deltas) |
| SAM ViT-H fp16 | 1.18x enc, 1.08x AMG | 20.3->17.3 J enc (-15%), 66->62 J/img | 2.87->2.45 GB | Hungarian mIoU 0.961 (fp32: 0.9614), 3/614 misses domain-relevant |

- fp16 = half weights; SAM mask generation additionally under fp16 autocast.
- fp16 amplifies the T-VELCRO speedup vs fp32 (ViT-Base 1.10->1.41, SAM enc 1.13->1.18).
- hooked variant slower than baseline (3.49 vs 2.58 ms) - hook is the fidelity mode.
- SAM eval: 100 kvasir imgs, diagnostic_level 0 (default 3 crashes under fp16).

# Pruning ladder, both models (2026-09-16, L4, FP16, batch 1, encoder)

| Model | 12.5% | 25% | 37.5% | 50% |
|---|---|---|---|---|
| SAM ViT-H speedup | 1.01x | 1.18x | 1.24x | 1.38x |
| SAM ViT-H energy | -2.0% | -15.1% | -20.2% | -27.8% |
| SAM full mask gen | 1.02x | 1.08x | 1.12x | 1.20x |
| SAM peak mem (MB) | 2672 | 2454 | 2199 | 1963 (base 2866) |
| DINOv3-7B speedup | 1.07x | 1.11x | 1.17x | 1.20x |
| DINOv3-7B energy | -6.6% | -11.0% | -12.8% | -16.8% |
| DINOv3-7B peak mem (GB) | 12.29 | 11.64 | 10.99 | 10.36 (base 12.85) |

SAM profile: kvasir, f=0.5 manifest generated 2026-09-16 from the stored profile.
SAM 12.5% is within noise of baseline (2 of 16 heads per block).
