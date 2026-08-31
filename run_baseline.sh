#!/bin/bash
# 起 vLLM → 跑分解评测 → 关掉。两个候选 base model 各跑一次。
#   ./run_baseline.sh /root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct
set -uo pipefail
MODEL="${1:?用法: ./run_baseline.sh <模型路径>}"
PORT="${PORT:-8000}"
source /root/autodl-tmp/env.sh
PROJ_DIR=/root/autodl-tmp/code-agent
export PYTHONPATH="$PROJ_DIR/flash_attn_shim:$PROJ_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

TAG=$(basename "$MODEL")
LOG=$PROJ_DIR/vllm_$TAG.log

# baseline 阶段 GPU 全归 vLLM（没有训练进程抢显存），可以给高一些
setsid --fork python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --port "$PORT" --gpu-memory-utilization 0.85 \
  --max-model-len 8192 --tensor-parallel-size 1 \
  < /dev/null > "$LOG" 2>&1

# out-dir 必须按模型隔离：run_baseline.py 的 dump 是 append 模式，
# 共用目录会让不同模型的 per-task 记录混进同一个 step_00000.jsonl
python "$PROJ_DIR/run_baseline.py" --model "$MODEL" --port "$PORT" --tag "$TAG" \
  --out-dir "/root/autodl-tmp/runs/baseline-$TAG" \
  2>&1 | tee "$PROJ_DIR/baseline_$TAG.log"

# 按 PID 关，不用 pkill -f：那会匹配到发起检查的命令自己
ps -eo pid=,args= | awk '/vllm.entrypoints.openai.api_server/ && !/awk/ {print $1}' \
  | xargs -r kill 2>/dev/null
echo "baseline $TAG done $(date +%H:%M:%S)" >> "$PROJ_DIR/runall.done"
