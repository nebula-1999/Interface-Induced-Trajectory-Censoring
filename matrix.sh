#!/bin/bash
# 2×2 矩阵：协议(fc/react) × 强度(optional/mandatory)，三档规模。
# 同一批 KodCode 题、同一沙箱、同样 4 轮 —— 只有这两个变量在动。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$P" || exit 1

cell () {                       # 单格：跑 + 校验，通过才写 done
  local NAME=$1 M=$2 N=$3 PROTO=$4 STR=$5
  local TAG="${NAME}_${PROTO}_${STR}"
  local JL="$P/traj_$TAG.jsonl"
  python "$P/probe_react_full.py" --model "$M" --port 8000 --n "$N" \
    --protocol "$PROTO" --strength "$STR" --out "$JL" \
    2>&1 | grep -v "Warning: You are sending" | tee "$P/mx_$TAG.log"
  local rc=${PIPESTATUS[0]}
  local lines=0; [ -f "$JL" ] && lines=$(wc -l < "$JL")
  local errs; errs=$(grep -oE "请求错误: [0-9]+" "$P/mx_$TAG.log" | grep -oE "[0-9]+" | head -1)
  errs=${errs:-999}
  # 三重校验：python 退出码、JSONL 行数、请求错误数
  if [ "$rc" -eq 0 ] && [ "$lines" -eq "$N" ] && [ "$errs" -eq 0 ]; then
    echo "OK  $TAG  (rc=$rc lines=$lines errs=$errs) $(date +%H:%M:%S)" >> "$P/runall.done"
  else
    echo "BAD $TAG  (rc=$rc lines=$lines/$N errs=$errs) 结果不可用 $(date +%H:%M:%S)" >> "$P/runall.done"
  fi
}

serve () {
  local M=$1 MAXLEN=${2:-8192} UTIL=${3:-0.85}
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    < /dev/null > "$P/vllm_mx.log" 2>&1
  for _ in $(seq 120); do
    curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || return 1
    sleep 5
  done
  return 1
}
stop () { ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 25; }

run_model () {
  local NAME=$1 M=$2 N=$3 MAXLEN=${4:-8192} UTIL=${5:-0.85}
  serve "$M" "$MAXLEN" "$UTIL" || { echo "BAD $NAME vLLM 未起" >> "$P/runall.done"; return; }
  for PROTO in fc react; do
    for STR in optional mandatory; do
      cell "$NAME" "$M" "$N" "$PROTO" "$STR"
    done
  done
  stop
}

run_model 1.5B /root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct 200
run_model 7B   /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct   100
run_model 32B  /root/autodl-tmp/models/Qwen2.5-Coder-32B-Instruct  50 4096 0.95

tar czf matrix_results.tgz mx_*.log traj_*.jsonl ablation_*.log poscontrol_*.log \
    n200_*.log diag*.log runall.done 2>/dev/null
echo "MATRIX DONE $(date +%H:%M:%S)" >> "$P/runall.done"
sync
sleep 2400
shutdown -h now
