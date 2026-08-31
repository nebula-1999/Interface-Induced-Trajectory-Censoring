#!/bin/bash
# 主 ReAct 臂以真实 2048 重跑。v13 阶段 1 只修了 DS×2 与 Llama-3.2×2，
# 而论文引用的三组核心配对（Llama-3.1-8B / Qwen-7B / Mistral-7B）的 ReAct 臂
# 仍是 1024，对应 FC 臂却是 2048 —— "统一上限"对主结论从未成立。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
# 等 v13 的严格验证跑完（权威标记），不看宽松的 V13_READY
for _ in $(seq 720); do { [ -f V13_VERIFIED_OK ] || [ -f V13_VERIFIED_FAIL ]; } && break; sleep 20; done
rm -f V14_READY V14_INVALID v14_manifest.txt
N=100; ADP="--parser-adapter cross_family"; MD=/root/autodl-tmp/models
sha256sum probe_react_full.py | cut -c1-16 | xargs -I{} echo "probe_sha={}" >> v14_manifest.txt

port_busy(){ awk '$2 ~ /:1F40$/ && $4=="0A" {f=1} END {exit !f}' /proc/net/tcp; }
gpu_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hard_clean(){
  for pat in api_server EngineCore VLLM::Engine; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; done
  sleep 3
  for pat in api_server EngineCore VLLM::Engine; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; done
  for _ in $(seq 60); do
    port_busy || { [ "$(gpu_mb)" -lt 2000 ] && return 0; }; sleep 3; done
  echo "  !! 清理超时"; return 1; }
serve(){ local M=$1 TAG=$2; shift 2
  hard_clean || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v14_$TAG.log" 2>&1
  for _ in $(seq 240); do
    got=$(curl -s --max-time 5 localhost:8000/v1/models 2>/dev/null | head -c 4000)
    case "$got" in *"$M"*) return 0;; esac
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起或模型名不符"; return 1; }
run(){ local TAG=$1 OUT=$2 M=$3
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  rm -f "$OUT"
  python probe_react_full.py --model "$M" --port 8000 --n $N $ADP \
    --protocol react --strength optional --temperature 0.0 --out "$OUT" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]" | tail -12
  local rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "$TAG rc=$rc lines=$lines $(date +%H:%M:%S)" >> v14_manifest.txt; }

for spec in "Llama-3.1-8B-Instruct:Llama8B" "Qwen2.5-Coder-7B-Instruct:Qwen7B" \
            "Mistral-7B-Instruct-v0.3:Mistral7B"; do
  MDIR=${spec%%:*}; TAG=${spec##*:}; M="$MD/$MDIR"
  [ -f "$M/config.json" ] || { echo "$TAG MISSING" >> v14_manifest.txt; continue; }
  serve "$M" "$TAG" && run "${TAG}_react2048" "traj_v14_${TAG}_react.jsonl" "$M" \
    || echo "$TAG SERVE_FAIL" >> v14_manifest.txt
done
hard_clean || true
touch V14_DONE
echo "V14_DONE $(date +%H:%M:%S)" >> runall.done
