#!/bin/bash
# A3：官方完整配置重跑 FC 臂（此前所有 FC 臂都缺 chat-template 等必需参数）
# A4：Qwen 五规模 + 意图检测器 —— 量化「服务端解析出的调用」与「模型真实调用意图」的缺口
#
# A4 统一 max-model-len 4096 / util 0.95：历史上只有 32B 是 4096、其余 8192，
# 那是个隐藏的跨规模不一致，这里消除掉。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for _ in $(seq 1200); do [ -f V4_READY ] && break; sleep 30; done
rm -f V5_READY
N=100; ADP="--parser-adapter cross_family"
T=/root/autodl-tmp/code-agent/templates

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2 LEN=$3 UTIL=$4; shift 4
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization "$UTIL" --max-model-len "$LEN" "$@" \
    < /dev/null > "vllm_v5_$TAG.log" 2>&1
  for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback|Assertion" "vllm_v5_$TAG.log" | head -4; return 1; }
preflight(){ local c=$(curl -s -o /tmp/pf5.json -w "%{http_code}" localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "{\"model\":\"$1\",\"max_tokens\":16,\"temperature\":0,
      \"messages\":[{\"role\":\"user\",\"content\":\"run the tests\"}],\"tool_choice\":\"auto\",
      \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run_tests\",\"description\":\"run unit tests\",
      \"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}]}")
  echo "  FC preflight HTTP $c"; [ "$c" = "200" ] || head -c 300 /tmp/pf5.json; [ "$c" = "200" ]; }
smoke(){ local M=$1 TAG=$2
  echo "  --- 冒烟 n=3 ($TAG) ---"
  python probe_react_full.py --model "$M" --port 8000 --n 3 $ADP --protocol fc --strength optional \
    --out "traj_v5_smoke_$TAG.jsonl" 2>&1 | grep -E "服务端解析出调用|未被解析但确在尝试|L1 严格|请求错误"; }
run(){ local TAG=$1 OUT=$2; shift 2
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成，跳过"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --port 8000 --n $N $ADP --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "$TAG rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0) $(date +%H:%M:%S)" >> v5_progress.txt
  return 0; }

MD=/root/autodl-tmp/models

############ A3：官方完整配置 ############
echo "########## A3-1 Llama-3.1-8B 官方配置（加 chat-template）##########"
if serve "$MD/Llama-3.1-8B-Instruct" ll_official 8192 0.85 \
     --enable-auto-tool-choice --tool-call-parser llama3_json \
     --chat-template "$T/tool_chat_template_llama3.1_json.jinja" && preflight "$MD/Llama-3.1-8B-Instruct"; then
  smoke "$MD/Llama-3.1-8B-Instruct" ll_official
  run Llama8B_fc_official traj_v5_Llama8B_fc_official.jsonl \
    --model "$MD/Llama-3.1-8B-Instruct" --protocol fc --strength optional
fi
kill_vllm

echo "########## A3-2 Mistral-7B-v0.3 官方配置（hf format + parallel 模板）##########"
if serve "$MD/Mistral-7B-Instruct-v0.3" mi_official 8192 0.85 \
     --tokenizer-mode hf --config-format hf --load-format hf \
     --enable-auto-tool-choice --tool-call-parser mistral \
     --chat-template "$T/tool_chat_template_mistral_parallel.jinja" && preflight "$MD/Mistral-7B-Instruct-v0.3"; then
  smoke "$MD/Mistral-7B-Instruct-v0.3" mi_official
  run Mistral7B_fc_official traj_v5_Mistral7B_fc_official.jsonl \
    --model "$MD/Mistral-7B-Instruct-v0.3" --protocol fc --strength optional
fi
kill_vllm

############ A4：Qwen 五规模意图缺口 ############
for S in 1.5B 3B 7B 14B 32B; do
  M="$MD/Qwen2.5-Coder-${S}-Instruct"
  [ -f "$M/config.json" ] || { echo "[Qwen$S] 缺模型，跳过"; continue; }
  echo "########## A4 Qwen2.5-Coder-$S 意图缺口 ##########"
  if serve "$M" "qw_$S" 4096 0.95 --enable-auto-tool-choice --tool-call-parser hermes && preflight "$M"; then
    run "Qwen${S}_fc_intent" "traj_v5_Qwen${S}_fc_intent.jsonl" \
      --model "$M" --protocol fc --strength optional
  fi
  kill_vllm
done

touch V5_READY
echo "V5_READY $(date +%H:%M:%S)" >> runall.done
