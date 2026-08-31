#!/bin/bash
# 复盘后的重跑队列。顺序按「判决价值」排，中途断电也先拿到最关键的答案。
#
# 为什么几乎全部重跑：fix_audit.py 把 max_tokens 从 1024 抬到 2048。
# 任何跨协议比较都必须共用同一上限，否则 FC 因 JSON 转义更费 token 而被系统性压低。
# 旧数据（traj_x2_*）一律标记为 v2，不与新数据混用。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
rm -f V3_READY
N=100
ADP="--parser-adapter cross_family"   # 全家族统一口径，消除 Qwen legacy 的不可比

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2; shift 2
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v3_$TAG.log" 2>&1
  for _ in $(seq 200); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback" "vllm_v3_$TAG.log" | head -3; return 1; }
preflight(){ local c=$(curl -s -o /tmp/pf3.json -w "%{http_code}" localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "{\"model\":\"$1\",\"max_tokens\":16,\"temperature\":0,
      \"messages\":[{\"role\":\"user\",\"content\":\"run the tests\"}],\"tool_choice\":\"auto\",
      \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run_tests\",\"description\":\"run unit tests\",
      \"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}]}")
  echo "  FC preflight HTTP $c"; [ "$c" = "200" ]; }

run(){  # run <tag> <out> <probe 其余参数...>
  local TAG=$1 OUT=$2; shift 2
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成，跳过"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --port 8000 --n $N $ADP --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  local rc=${PIPESTATUS[0]}
  echo "$TAG rc=$rc lines=$(wc -l < "$OUT" 2>/dev/null || echo 0) $(date +%H:%M:%S)" >> v3_progress.txt
  [ "$rc" = "2" ] && echo "  ⚠️ $TAG 有请求错误，该臂不可用" >> v3_progress.txt
  # 必须显式 return：否则上一行的 [ ] 判断为假时函数返回 1，
  # && 链会静默跳过后续实验臂（Llama8B_fc_legacy 已因此被跳过一次）。
  return "$rc"
}

QW=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
LL=/root/autodl-tmp/models/Llama-3.1-8B-Instruct
MI=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3

### 优先级 1：解决那 26 条 —— Llama8B 的 FC/ReAct 在新上限下重跑
if serve "$LL" ll_fc --enable-auto-tool-choice --tool-call-parser llama3_json && preflight "$LL"; then
  run Llama8B_fc        traj_v3_Llama8B_fc.jsonl        --model "$LL" --protocol fc --strength optional
  run Llama8B_fc_legacy traj_v3_Llama8B_fc_legacy.jsonl --model "$LL" --protocol fc \
      --strength optional --sys-file sys_legacy_train.txt
fi
kill_vllm
if serve "$LL" ll_react; then
  run Llama8B_react traj_v3_Llama8B_react.jsonl --model "$LL" --protocol react --strength optional
fi
kill_vllm

### 优先级 2：分离协议效应与训练 SYSTEM 尾句效应（Qwen 是训练用的模型）
if serve "$QW" qw_fc --enable-auto-tool-choice --tool-call-parser hermes && preflight "$QW"; then
  run Qwen7B_fc_nosuffix traj_v3_Qwen7B_fc_nosuffix.jsonl --model "$QW" --protocol fc --strength optional
  run Qwen7B_fc_legacy   traj_v3_Qwen7B_fc_legacy.jsonl   --model "$QW" --protocol fc \
      --strength optional --sys-file sys_legacy_train.txt
fi
kill_vllm
if serve "$QW" qw_react; then
  run Qwen7B_react traj_v3_Qwen7B_react.jsonl --model "$QW" --protocol react --strength optional
fi
kill_vllm

### 优先级 3：Mistral 重跑（旧 fc 臂有 3 个 request_error，脚本自判不可用）
if serve "$MI" mi_fc --enable-auto-tool-choice --tool-call-parser mistral && preflight "$MI"; then
  run Mistral7B_fc traj_v3_Mistral7B_fc.jsonl --model "$MI" --protocol fc --strength optional
fi
kill_vllm
if serve "$MI" mi_react; then
  run Mistral7B_react traj_v3_Mistral7B_react.jsonl --model "$MI" --protocol react --strength optional
fi
kill_vllm

touch V3_READY
echo "V3_READY $(date +%H:%M:%S)" >> runall.done
