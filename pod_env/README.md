# Persistent pod environment (lives on the tvelcro-data network volume at /workspace/env)

Deploy pods from the RunPod template "tvelcro-ml" with the tvelcro-data volume attached:
bootstrap runs automatically at pod start, and login lands in zsh + oh-my-zsh with the
mlenv venv (torch cu126, vLLM, transformers, datasets, jupyter) active.
Aliases: `ml` (vLLM learning env), `tv` (experiments env), `gpu` (utilization).
HF and pip caches persist on the volume (HF_HOME, PIP_CACHE_DIR).
To rebuild a fresh volume: copy this directory to /workspace/env and run bootstrap.sh once.
