#!/bin/bash
# 单句 prompt 消融：判定历史 0/1000 是"FC 协议压制"还是"训练 SYSTEM 尾句压制"
#   arm A  Qwen7B react           （基线）
#   arm B  Qwen7B fc  FC_OPTIONAL （无尾句）
#   arm C  Qwen7B fc  legacy      （含尾句，= 0/1000 那批的 prompt）
#   arm D  Llama8B fc legacy      （跨家族确认；其 FC_OPTIONAL 版已得 97%）
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for _ in $(seq 900); do [ -f XFAM2_READY ] && break; sleep 30; done
rm -f QWENFC2_READY
N=100
wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2; shift 2
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_ab_$TAG.log" 2>&1
  for _ in $(seq 200); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback" "vllm_ab_$TAG.log" | head -3; return 1; }
preflight(){ local M=$1
  local c=$(curl -s -o /tmp/pf2.json -w "%{http_code}" localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "{\"model\":\"$M\",\"max_tokens\":16,\"temperature\":0,
      \"messages\":[{\"role\":\"user\",\"content\":\"run the tests\"}],\"tool_choice\":\"auto\",
      \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run_tests\",\"description\":\"run unit tests\",
      \"parameters\":{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"]}}}]}")
  echo "  FC preflight HTTP $c"; [ "$c" = "200" ]; }

QW=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
LL=/root/autodl-tmp/models/Llama-3.1-8B-Instruct

echo "########## A: Qwen7B react ##########"
if serve "$QW" qw_react; then
  python probe_react_full.py --model "$QW" --port 8000 --n $N --protocol react --strength optional \
    --out traj_ab_Qwen7B_react.jsonl 2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "A done $(date +%H:%M:%S)" >> ab_progress.txt; fi
kill_vllm

echo "########## Qwen7B FC 服务 (hermes) ##########"
if serve "$QW" qw_fc --enable-auto-tool-choice --tool-call-parser hermes && preflight "$QW"; then
  echo "########## B: Qwen7B fc  无尾句 ##########"
  python probe_react_full.py --model "$QW" --port 8000 --n $N --protocol fc --strength optional \
    --out traj_ab_Qwen7B_fc_nosuffix.jsonl 2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "B done $(date +%H:%M:%S)" >> ab_progress.txt
  echo "########## C: Qwen7B fc  含尾句(训练版) ##########"
  python probe_react_full.py --model "$QW" --port 8000 --n $N --protocol fc --strength optional \
    --sys-file sys_legacy_train.txt \
    --out traj_ab_Qwen7B_fc_legacy.jsonl 2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "C done $(date +%H:%M:%S)" >> ab_progress.txt
fi
kill_vllm

echo "########## D: Llama8B fc 含尾句(跨家族确认) ##########"
if serve "$LL" ll_fc --enable-auto-tool-choice --tool-call-parser llama3_json && preflight "$LL"; then
  python probe_react_full.py --model "$LL" --port 8000 --n $N --protocol fc --strength optional \
    --sys-file sys_legacy_train.txt \
    --out traj_ab_Llama8B_fc_legacy.jsonl 2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "D done $(date +%H:%M:%S)" >> ab_progress.txt
fi
kill_vllm
touch QWENFC2_READY
echo "QWENFC2_READY $(date +%H:%M:%S)" >> runall.done
