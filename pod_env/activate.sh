# auto-sourced ML environment (persistent on the tvelcro-data volume)
export HF_HOME=/workspace/.hf_cache
export PIP_CACHE_DIR=/workspace/.pip_cache
export PYTHONPATH=.
[ -f /workspace/.secrets.env ] && source /workspace/.secrets.env
alias tv="cd /workspace/tvelcro && source .venv/bin/activate"
alias ml="source /workspace/mlenv/bin/activate"
alias gpu="nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader"
[ -d /workspace/mlenv ] && source /workspace/mlenv/bin/activate 2>/dev/null
cd /workspace 2>/dev/null
echo "[mlenv] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)  |  ml=vllm env, tv=experiments env"
