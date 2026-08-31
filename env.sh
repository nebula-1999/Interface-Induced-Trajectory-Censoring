# code agent 项目的环境变量。每条命令前 source 一次。
# 要点：所有 cache 显式指向数据盘——系统盘只有 30 GB，
# 默认路径 (~/.cache) 在系统盘，下几个模型就能把它打爆。

export PROJ=/root/autodl-tmp/code-agent
export VENV=/root/code-venv

export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1          # 不设会导致 hf-mirror 下载全部 401
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TRITON_CACHE_DIR=/root/autodl-tmp/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cache/inductor
export VLLM_CACHE_ROOT=/root/autodl-tmp/cache/vllm

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
cd "$PROJ" 2>/dev/null || true
