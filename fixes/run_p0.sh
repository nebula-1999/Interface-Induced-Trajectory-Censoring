#!/bin/bash
# P0：随机取样 + n=300 重跑核心臂，一次性消除两条局限
#   · clean[:100] 非随机  →  固定种子 20260901 随机抽 300
#   · n=100 功效不足      →  n=300，尤其冲 §5.5 那个 p=0.093
#
# 不需要腾磁盘：用现有模型跑新题目，轨迹每份几百 KB。
# 32B/14B 必须保留 —— 规模曲线要它们。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f P0_READY P0_INVALID p0_manifest.txt
sha256sum probe_react_full.py >> p0_manifest.txt
IDS=$(python -c "import json;print(','.join(map(str,json.load(open('analysis/p0_random_300.json'))['ids'])))")
N=300; ADP="--parser-adapter cross_family"; MD=/root/autodl-tmp/models; T=$P/templates; PL=$P/plugin

port_busy(){ awk '$2 ~ /:1F40$/ && $4=="0A" {f=1} END {exit !f}' /proc/net/tcp; }
gpu_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hard_clean(){
  for pat in api_server EngineCore launch_ppo "ray::"; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; done
  for _ in $(seq 60); do port_busy || { [ "$(gpu_mb)" -lt 2000 ] && return 0; }; sleep 3; done
  echo "!! 清理超时"; return 1; }
serve(){ local M=$1 TAG=$2 LEN=$3 UTIL=$4; shift 4
  hard_clean || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization "$UTIL" --max-model-len "$LEN" "$@" \
    < /dev/null > "vllm_p0_$TAG.log" 2>&1
  for _ in $(seq 300); do
    got=$(curl -s --max-time 5 localhost:8000/v1/models 2>/dev/null | head -c 4000)
    case "$got" in *"$M"*) return 0;; esac
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起或模型名不符"; return 1; }
run(){ local TAG=$1 OUT=$2 M=$3; shift 3
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成，跳过"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --model "$M" --port 8000 --only-ids "$IDS" $ADP \
    --temperature 0.0 --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]" | tail -12
  local rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "$TAG rc=$rc lines=$lines/$N $(date +%H:%M:%S)" >> p0_manifest.txt; }

########## A 规模曲线：Qwen 五档（fig 1 的 n=300 随机版）##########
for S in 1.5B 3B 7B 14B 32B; do
  M="$MD/Qwen2.5-Coder-${S}-Instruct"
  [ -f "$M/config.json" ] || { echo "Qwen$S MISSING" >> p0_manifest.txt; continue; }
  if serve "$M" "qw$S" 4096 0.95 --enable-auto-tool-choice --tool-call-parser hermes; then
    run "Qwen${S}_fc_intent" "traj_p0_Qwen${S}_fc_intent.jsonl" "$M" --protocol fc --strength optional
  fi
done
hard_clean || true

########## B 主表配对：Llama-8B ##########
LL=$MD/Llama-3.1-8B-Instruct
serve "$LL" ll_react 8192 0.85 && run Llama8B_react traj_p0_Llama8B_react.jsonl "$LL" --protocol react --strength optional
serve "$LL" ll_fc 8192 0.85 --enable-auto-tool-choice --tool-call-parser llama3_json \
  --chat-template "$T/tool_chat_template_llama3.1_json.jinja" \
  && run Llama8B_fc_strict traj_p0_Llama8B_fc_strict.jsonl "$LL" --protocol fc --strength optional --fc-schema strict

########## C 修复闭环：Qwen-7B（冲 p=0.093）##########
QW=$MD/Qwen2.5-Coder-7B-Instruct
serve "$QW" qw_react 8192 0.85 && run Qwen7B_react traj_p0_Qwen7B_react.jsonl "$QW" --protocol react --strength optional
serve "$QW" qw_plug 8192 0.85 --enable-auto-tool-choice \
  --tool-parser-plugin "$PL/qwen2_5_coder_tool_parser.py" --tool-call-parser qwen2_5_coder \
  --chat-template "$PL/tool_chat_template_qwen2_5_coder.jinja" \
  && run Qwen7B_fc_plugin traj_p0_Qwen7B_fc_plugin.jsonl "$QW" --protocol fc --strength optional
hard_clean || true

ok=1
for f in traj_p0_Qwen1.5B_fc_intent.jsonl traj_p0_Qwen3B_fc_intent.jsonl traj_p0_Qwen7B_fc_intent.jsonl \
         traj_p0_Qwen14B_fc_intent.jsonl traj_p0_Qwen32B_fc_intent.jsonl \
         traj_p0_Llama8B_react.jsonl traj_p0_Llama8B_fc_strict.jsonl \
         traj_p0_Qwen7B_react.jsonl traj_p0_Qwen7B_fc_plugin.jsonl; do
  [ "$(wc -l < "$f" 2>/dev/null || echo 0)" = "$N" ] || ok=0
done
[ "$ok" = 1 ] && touch P0_READY || touch P0_INVALID
echo "########## p0 manifest ##########"; cat p0_manifest.txt
