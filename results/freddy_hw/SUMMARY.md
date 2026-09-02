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
