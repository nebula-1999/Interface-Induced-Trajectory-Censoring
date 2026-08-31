#!/bin/bash
# 跨家族 n=100 正式跑（修正版）
#  - FC arm 必须用 --enable-auto-tool-choice + 家族对应 parser 重启服务
#  - FC arm 开跑前 preflight：带 tools 的请求必须 HTTP 200，否则记 SKIP 不产假数据
#  - 已完成的 arm（行数 >= N）自动跳过
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$P" || exit 1
rm -f XFAM2_READY
N=100

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }

serve(){  # serve <model> <logtag> [extra args...]
  local M=$1 TAG=$2; shift 2
  wait_port_free || { echo "[$TAG] 端口未释放"; return 1; }
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_x2_$TAG.log" 2>&1
  for _ in $(seq 200); do
    curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break
    sleep 5
  done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback" "vllm_x2_$TAG.log" | head -3; return 1
}

fc_preflight(){  # 验证服务真的接受 tools + tool_choice:auto
  local M=$1
  local code
  code=$(curl -s -o /tmp/pf.json -w "%{http_code}" localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "{
      \"model\": \"$M\", \"max_tokens\": 16, \"temperature\": 0,
      \"messages\": [{\"role\":\"user\",\"content\":\"run the tests\"}],
      \"tool_choice\": \"auto\",
      \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"run_tests\",
        \"description\":\"run unit tests on code\",
        \"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}]}")
  if [ "$code" != "200" ]; then
    echo "  !! FC preflight 失败 HTTP $code: $(head -c 200 /tmp/pf.json)"
    return 1
  fi
  echo "  FC preflight OK (HTTP 200)"
  return 0
}

arm(){  # arm <model> <tag> <protocol>
  local M=$1 TAG=$2 PROTO=$3
  local OUT="traj_x2_${TAG}_${PROTO}.jsonl"
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then
    echo "===== $TAG / $PROTO : 已完成，跳过 ====="; return; fi
  echo "===== $TAG / $PROTO / n=$N  $(date +%H:%M:%S) ====="
  if [ "$PROTO" = "fc" ]; then
    fc_preflight "$M" || { echo "$TAG $PROTO SKIP_PREFLIGHT" >> xfam2_progress.txt; return; }
  fi
  python probe_react_full.py --model "$M" --port 8000 --n "$N" \
    --protocol "$PROTO" --strength optional --out "$OUT" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "$TAG $PROTO done $(wc -l < "$OUT") $(date +%H:%M:%S)" >> xfam2_progress.txt
}

run_model(){  # run_model <model> <tag> <fc_parser|none> <proto...>
  local M=$1 TAG=$2 PARSER=$3; shift 3
  if [ ! -f "$M/config.json" ]; then echo "[$TAG] 模型缺失，跳过"; return; fi
  # ReAct arm：普通服务
  for PROTO in "$@"; do [ "$PROTO" = "react" ] || continue
    serve "$M" "${TAG}_react" || { kill_vllm; return; }
    arm "$M" "$TAG" react
    kill_vllm
  done
  # FC arm：带 auto tool choice + 家族 parser 的服务
  for PROTO in "$@"; do [ "$PROTO" = "fc" ] || continue
    [ "$PARSER" = "none" ] && { echo "[$TAG] 无 FC parser，跳过 fc arm"; continue; }
    serve "$M" "${TAG}_fc" --enable-auto-tool-choice --tool-call-parser "$PARSER" \
      || { echo "[$TAG] FC 服务起不来（parser=$PARSER）"; kill_vllm; continue; }
    echo "  服务 flag: $(grep -oE 'enable_auto_tool_choice=[A-Za-z]+' vllm_x2_${TAG}_fc.log | head -1) parser=$PARSER"
    arm "$M" "$TAG" fc
    kill_vllm
  done
}

MD=/root/autodl-tmp/models
run_model $MD/Llama-3.1-8B-Instruct        Llama8B   llama3_json  react fc
run_model $MD/Mistral-7B-Instruct-v0.3     Mistral7B mistral      react fc
run_model $MD/deepseek-coder-6.7b-instruct DS6.7B    none         react
run_model $MD/deepseek-coder-1.3b-instruct DS1.3B    none         react
run_model $MD/Llama-3.2-3B-Instruct        Llama3B   llama3_json  react fc
run_model $MD/Llama-3.2-1B-Instruct        Llama1B   llama3_json  react fc

touch XFAM2_READY
echo "XFAM2_READY $(date +%H:%M:%S)" >> runall.done
