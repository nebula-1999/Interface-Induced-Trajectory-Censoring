#!/bin/bash
# 一条命令跑完工具调用诊断：起 vLLM（带 hermes parser）→ 探测 → 关掉。
# 约 3 分钟。不训练、不评测，只看模型对带 tools 的请求返回什么。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent
M="${1:-/root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 关键：--enable-auto-tool-choice + --tool-call-parser hermes
# 训练时 verl 起的 rollout server 是否带了这两个参数，正是嫌疑所在
setsid --fork python -m vllm.entrypoints.openai.api_server \
  --model "$M" --served-model-name "$M" --port 8000 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  < /dev/null > "$P/vllm_probe_tool.log" 2>&1

for _ in $(seq 60); do
  curl -s localhost:8000/v1/models > /dev/null 2>&1 && break
  sleep 5
done

python "$P/probe_toolcall.py" --model "$M" --port 8000 2>&1 | tee "$P/toolcall_diag.log"

ps -eo pid=,args= | awk '/vllm.entrypoints.openai.api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null
echo "toolcall diag done $(date +%H:%M:%S)" >> "$P/runall.done"
