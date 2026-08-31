#!/bin/bash
# 阶段 1：以真实 2048 上限重跑 v11 的四个 ReAct 臂（此前 gen 默认 1024，
#         而 FC 走 MAX_TOKENS=2048，且 provenance 两边都错记 2048）。
# 阶段 2：A9 方差估计（Llama-8B ReAct vs FC官方+strict，temp 0.6 × 3 seed）。
#
# 修掉的四处（Codex）：
#   1) ReAct 真实上限 1024 → 已在 probe 修为 MAX_TOKENS
#   2) ss/lsof/fuser 均不存在，端口检查是假的 → 改用 /proc/net/tcp + 显存双条件
#   3) 成功门槛只看行数 → 改为 rc + 行数 + request_error + provenance 校验
#   4) 运行中改脚本导致 provenance schema 不一致 → 每臂记 script_sha256
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f V13_READY V13_INVALID v13_manifest.txt
N=100; ADP="--parser-adapter cross_family"; MD=/root/autodl-tmp/models; T=$P/templates
SHA=$(sha256sum probe_react_full.py | cut -c1-16)
echo "probe_sha=$SHA" >> v13_manifest.txt

# 端口 8000 = 0x1F40；LISTEN 状态 = 0A
port_busy(){ awk '$2 ~ /:1F40$/ && $4=="0A" {f=1} END {exit !f}' /proc/net/tcp; }
gpu_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hard_clean(){
  for pat in api_server EngineCore VLLM::Engine; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill 2>/dev/null
  done
  sleep 3
  for pat in api_server EngineCore VLLM::Engine; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null
  done
  for _ in $(seq 60); do
    port_busy || { [ "$(gpu_mb)" -lt 2000 ] && return 0; }
    sleep 3
  done
  echo "  !! 清理超时 port_busy=$(port_busy && echo yes || echo no) gpu=$(gpu_mb)MiB"; return 1; }

serve(){ local M=$1 TAG=$2; shift 2
  hard_clean || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v13_$TAG.log" 2>&1
  for _ in $(seq 240); do
    got=$(curl -s --max-time 5 localhost:8000/v1/models 2>/dev/null | head -c 4000)
    case "$got" in *"$M"*) return 0;; esac      # 必须确认返回的模型名就是本臂的
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5
  done
  echo "[$TAG] vLLM 未起或模型名不符"; grep -iE "error|Traceback" "vllm_v13_$TAG.log" | head -4; return 1; }

check(){ # check <out> <期望模型> <期望协议> <期望温度>
  python - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
f, want_m, want_p, want_t = sys.argv[1:5]
try:
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
except Exception as e:
    print(f"BAD json_error={e}"); raise SystemExit
if not R: print("BAD empty"); raise SystemExit
r0 = R[0]
errs = sum(1 for r in R for t in r["turns"] if (t.get("raw_output") or "").startswith("__ERROR__"))
bad = []
if want_m not in str(r0.get("model", "")): bad.append(f"model={r0.get('model')}")
if r0.get("protocol") != want_p: bad.append(f"protocol={r0.get('protocol')}")
if abs(float(r0.get("temperature", 0)) - float(want_t)) > 1e-9: bad.append(f"temp={r0.get('temperature')}")
if int(r0.get("max_tokens", 0)) != 2048: bad.append(f"max_tokens={r0.get('max_tokens')}")
print(("OK" if not bad and errs == 0 else "BAD") +
      f" n={len(R)} n_err={errs} sha={r0.get('script_sha256')} " + " ".join(bad))
PY
}

run(){ # run <tag> <out> <模型> <协议> <温度> <其余参数...>
  local TAG=$1 OUT=$2 M=$3 PROTO=$4 TEMP=$5; shift 5
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  rm -f "$OUT"
  python probe_react_full.py --model "$M" --port 8000 --n $N $ADP \
    --protocol "$PROTO" --temperature "$TEMP" --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]" | tail -12
  local rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  local v=$(check "$OUT" "$M" "$PROTO" "$TEMP")
  echo "$TAG rc=$rc lines=$lines $v $(date +%H:%M:%S)" >> v13_manifest.txt
  echo "  校验: $v"; }

########## 阶段 1：ReAct 臂以真实 2048 重跑 ##########
for spec in "deepseek-coder-1.3b-instruct:DS1.3b" "deepseek-coder-6.7b-instruct:DS6.7b" \
            "Llama-3.2-1B-Instruct:Llama32_1B" "Llama-3.2-3B-Instruct:Llama32_3B"; do
  MDIR=${spec%%:*}; TAG=${spec##*:}; M="$MD/$MDIR"
  [ -f "$M/config.json" ] || { echo "$TAG MISSING" >> v13_manifest.txt; continue; }
  serve "$M" "$TAG" && run "${TAG}_react2048" "traj_v13_${TAG}_react.jsonl" "$M" react 0.0 --strength optional
done
hard_clean || true

########## 阶段 2：A9 方差（temp 0.6 × 3 seed）##########
LL=$MD/Llama-3.1-8B-Instruct
if serve "$LL" a9react; then
  for SD in 1 2 3; do
    run "a9_react_s$SD" "traj_v13_Llama8B_react_t06_s$SD.jsonl" "$LL" react 0.6 --strength optional --seed $SD
  done
fi
if serve "$LL" a9fc --enable-auto-tool-choice --tool-call-parser llama3_json \
     --chat-template "$T/tool_chat_template_llama3.1_json.jinja"; then
  for SD in 1 2 3; do
    run "a9_fc_s$SD" "traj_v13_Llama8B_fcstrict_t06_s$SD.jsonl" "$LL" fc 0.6 \
      --strength optional --fc-schema strict --seed $SD
  done
fi
hard_clean || true

grep -q "BAD" v13_manifest.txt && touch V13_INVALID || touch V13_READY
echo "########## v13 manifest ##########"; cat v13_manifest.txt
