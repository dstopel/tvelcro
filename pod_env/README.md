# Persistent pod environment (lives on the tvelcro-data network volume at /workspace/env)

Deploy pods from the RunPod template "tvelcro-ml" with the tvelcro-data volume attached:
bootstrap runs automatically at pod start, and login lands in zsh + oh-my-zsh with the
mlenv venv (torch cu126, vLLM, transformers, datasets, jupyter) active.
Aliases: `ml` (vLLM learning env), `tv` (experiments env), `gpu` (utilization).
HF and pip caches persist on the volume (HF_HOME, PIP_CACHE_DIR).
To rebuild a fresh volume: copy this directory to /workspace/env and run bootstrap.sh once.

## Templates

- `tvelcro-ml-cu13` (id `jrqhr9ll86`) - CUDA 13.0 image (torch 2.9.1, nvcc 13),
  container disk 60 GB. Use this one: it matches the CUDA 13 host driver that
  current vLLM wheels require, and its nvcc matches the driver for writing
  custom CUDA kernels.
- `tvelcro-ml` (id `8f543v0hbb`) - older CUDA 12.4 image, kept for reference.

Neither template can pin the CUDA driver version: that is a per-deployment
setting (`allowedCudaVersions`), which scripts/deploy_pod.sh sets to 13.0.
