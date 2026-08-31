#!/bin/bash
# A7  DeepSeek 1.3B / 6.7B 在统一配置下补 ReAct（现有数据停在 v2：1024 上限、
#     无意图检测器、无函数名校验，跨家族表目前不可比）。
# A6  Llama-3.2 1B / 3B：给 Llama 家族也加上规模轴（ReAct + FC 各一）。
# 统一配置：max-model-len 8192、MAX_TOKENS 2048、cross_family adapter、意图检测器。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
for _ in $(seq 60); do { [ -f A14_READY ] || [ -f A14_INVALID ]; } && break; sleep 15; done
rm -f V11_READY V11_INVALID v11_manifest.txt
N=100; ADP="--parser-adapter cross_family"; MD=/root/autodl-tmp/models

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2; shift 2
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v11_$TAG.log" 2>&1
  for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback|denied" "vllm_v11_$TAG.log" | head -4; return 1; }
run(){ local TAG=$1 OUT=$2; shift 2
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --port 8000 --n $N $ADP --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  local rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "$TAG rc=$rc lines=$lines $(date +%H:%M:%S)" >> v11_manifest.txt; }

########## A7：DeepSeek 统一配置 ##########
for S in 1.3b 6.7b; do
  M="$MD/deepseek-coder-${S}-instruct"
  [ -f "$M/config.json" ] || { echo "DS$S MISSING_MODEL" >> v11_manifest.txt; continue; }
  echo "########## A7  DeepSeek-$S  ReAct（统一配置）##########"
  if serve "$M" "ds$S"; then
    run "DS${S}_react" "traj_v11_DS${S}_react.jsonl" --model "$M" --protocol react --strength optional
  else echo "DS$S SERVE_FAIL" >> v11_manifest.txt; fi
  kill_vllm
done

########## A6：Llama-3.2 规模轴 ##########
for S in 1B 3B; do
  M="$MD/Llama-3.2-${S}-Instruct"
  [ -f "$M/config.json" ] || { echo "Llama32_$S MISSING_MODEL" >> v11_manifest.txt; continue; }
  echo "########## A6  Llama-3.2-$S  ReAct ##########"
  if serve "$M" "l32${S}r"; then
    run "Llama32_${S}_react" "traj_v11_Llama32_${S}_react.jsonl" --model "$M" --protocol react --strength optional
  else echo "Llama32_$S SERVE_FAIL_react" >> v11_manifest.txt; fi
  kill_vllm
  echo "########## A6  Llama-3.2-$S  FC（官方模板 + strict）##########"
  if serve "$M" "l32${S}f" --enable-auto-tool-choice --tool-call-parser llama3_json \
       --chat-template "$P/templates/tool_chat_template_llama3.2_json.jinja"; then
    run "Llama32_${S}_fc_strict" "traj_v11_Llama32_${S}_fc_strict.jsonl" \
      --model "$M" --protocol fc --strength optional --fc-schema strict
  else echo "Llama32_$S SERVE_FAIL_fc" >> v11_manifest.txt; fi
  kill_vllm
done

ok=1
for f in traj_v11_DS1.3b_react.jsonl traj_v11_DS6.7b_react.jsonl \
         traj_v11_Llama32_1B_react.jsonl traj_v11_Llama32_3B_react.jsonl \
         traj_v11_Llama32_1B_fc_strict.jsonl traj_v11_Llama32_3B_fc_strict.jsonl; do
  [ "$(wc -l < "$f" 2>/dev/null || echo 0)" = "$N" ] || ok=0
done
[ "$ok" = 1 ] && touch V11_READY || touch V11_INVALID
echo "########## v11 manifest ##########"; cat v11_manifest.txt
