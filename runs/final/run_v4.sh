#!/bin/bash
# 两个对照，各自拆掉一个混淆变量：
#   fc_rich : 把 FC 的 schema 描述补到与 ReAct 模板同等详尽 → 检验「schema 混淆」
#   fc_cot  : 给 FC 加上与 ReAct 同等的 Thought 脚手架    → 拆开「协议」与「思维链」
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for _ in $(seq 900); do [ -f V3_READY ] && break; sleep 30; done
rm -f V4_READY
N=100; ADP="--parser-adapter cross_family"
wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2; shift 2
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v4_$TAG.log" 2>&1
  for _ in $(seq 200); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; return 1; }
run(){ local TAG=$1 OUT=$2; shift 2
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成，跳过"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --port 8000 --n $N $ADP --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "$TAG rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0) $(date +%H:%M:%S)" >> v4_progress.txt
  return 0; }
QW=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
LL=/root/autodl-tmp/models/Llama-3.1-8B-Instruct

if serve "$LL" ll_v4 --enable-auto-tool-choice --tool-call-parser llama3_json; then
  run Llama8B_fc_rich traj_v4_Llama8B_fc_rich.jsonl --model "$LL" --protocol fc \
      --strength optional --fc-schema rich
  run Llama8B_fc_cot  traj_v4_Llama8B_fc_cot.jsonl  --model "$LL" --protocol fc \
      --strength optional --sys-file sys_fc_cot.txt
fi
kill_vllm
# Qwen 的 FC 变体已砍：hermes 对 Qwen2.5-Coder 是已知盲区（vLLM #32926），
# 在失灵的 parser 上叠加变量得不到可解释的数字。
touch V4_READY
echo "V4_READY $(date +%H:%M:%S)" >> runall.done
