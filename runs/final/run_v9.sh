#!/bin/bash
# Mistral 的 strict 臂。Llama 上 strict:true 把 23 条参数污染清零，
# Mistral 的 42% 硬 400 源于重复 [TOOL_CALLS]（生成阶段问题），
# strict 走受约束解码，很可能同样被压掉。若成立，「官方推荐配置反而更差」需改写。
#
# 两臂各自与已有的非 strict 臂单变量对齐：
#   A  仅 parser + strict     ← 对照 v3 Mistral7B_fc      (错误 2, 解析 2, 最终 31)
#   B  官方模板 + strict      ← 对照 v5 Mistral7B_fc_official (错误 42, 解析 5, 最终 15)
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
for _ in $(seq 240); do { [ -f V8_READY ] || [ -f V8_INVALID ]; } && break; sleep 30; done
rm -f V9_READY V9_INVALID v9_manifest.txt
N=100; ADP="--parser-adapter cross_family"
T=$P/templates
MI=/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local TAG=$1; shift
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$MI" --served-model-name "$MI" --port 8000 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v9_$TAG.log" 2>&1
  for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback" "vllm_v9_$TAG.log" | head -4; return 1; }

nerr_of(){ python - <<PY
import json
try: R=[json.loads(l) for l in open("$1",encoding="utf-8") if l.strip()]
except FileNotFoundError: print(-1); raise SystemExit
print(sum(1 for r in R for t in r["turns"] if (t.get("raw_output") or "").startswith("__ERROR__")))
PY
}

smoke(){ local TAG=$1
  local O="traj_v9_smoke_$TAG.jsonl"; rm -f "$O"
  echo "  --- 冒烟 n=3 ($TAG) ---"
  local txt rc lines ne
  txt=$(python probe_react_full.py --model "$MI" --port 8000 --n 3 $ADP \
        --protocol fc --strength optional --fc-schema strict --out "$O" 2>&1); rc=$?
  echo "$txt" | grep -E "请求错误|服务端解析出调用|调用了不存在的工具" || true
  lines=$(wc -l < "$O" 2>/dev/null || echo 0); ne=$(nerr_of "$O")
  echo "    退出码=$rc 行数=$lines n_err=$ne"
  # 探针在 n_err>0 时返回 rc=2 —— 而请求错误率正是本臂要估计的量，
  # 因此 rc=2 必须放行，否则冒烟里出现一个 400 就会中止，恰好测不到目标。
  # 只有「真正的测量故障」才拦：其他退出码，或行数不足。
  if { [ "$rc" != "0" ] && [ "$rc" != "2" ]; } || [ "$lines" != "3" ]; then
    echo "  ✗ 冒烟未通过（rc=$rc 非 0/2，或行数≠3）→ 中止该臂"
    echo "$TAG SMOKE_FAIL rc=$rc lines=$lines" >> v9_manifest.txt; return 1; fi
  echo "  ✓ 冒烟可用（rc=$rc, n_err=$ne 记录在案，不作为门槛）"; return 0; }

run(){ local TAG=$1 OUT=$2
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --model "$MI" --port 8000 --n $N $ADP \
    --protocol fc --strength optional --fc-schema strict --out "$OUT" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  local rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0) ne=$(nerr_of "$OUT")
  echo "$TAG rc=$rc lines=$lines n_err=$ne $(date +%H:%M:%S)" >> v9_manifest.txt
  return 0; }

echo "########## A  仅 parser + strict ##########"
if serve A --enable-auto-tool-choice --tool-call-parser mistral; then
  smoke A && run Mistral7B_fc_strict traj_v9_Mistral7B_fc_strict.jsonl
else echo "A SERVE_FAIL" >> v9_manifest.txt; fi
kill_vllm

echo "########## B  官方模板 + strict ##########"
if serve B --tokenizer-mode hf --config-format hf --load-format hf \
     --enable-auto-tool-choice --tool-call-parser mistral \
     --chat-template "$T/tool_chat_template_mistral_parallel.jinja"; then
  smoke B && run Mistral7B_fc_official_strict traj_v9_Mistral7B_fc_official_strict.jsonl
else echo "B SERVE_FAIL" >> v9_manifest.txt; fi
kill_vllm

# 两级判定：
#   可靠性测量有效 = 两臂都满 100 行且 rc ∈ {0,2}（错误率普查成立）
#   通过率可比     = 还要求 rc=0（无请求错误，缺失数据为零）
reliab=1; comparable=1
for e in Mistral7B_fc_strict:traj_v9_Mistral7B_fc_strict.jsonl \
         Mistral7B_fc_official_strict:traj_v9_Mistral7B_fc_official_strict.jsonl; do
  tag=${e%%:*}; out=${e##*:}
  n=$(wc -l < "$out" 2>/dev/null || echo 0)
  line=$(grep -m1 "^$tag rc=" v9_manifest.txt 2>/dev/null || true)
  rc=$(echo "$line" | grep -oE "rc=[0-9]+" | cut -d= -f2)
  [ "$n" = "$N" ] || { reliab=0; comparable=0; }
  case "${rc:-x}" in 0) ;; 2) comparable=0 ;; *) reliab=0; comparable=0 ;; esac
done
if [ "$reliab" = "1" ] && [ "$comparable" = "1" ]; then
  touch V9_READY; echo "V9_READY $(date +%H:%M:%S)" >> runall.done
elif [ "$reliab" = "1" ]; then
  touch V9_RELIABILITY_ONLY
  echo "V9_RELIABILITY_ONLY 可靠性(错误率)结果有效；通过率因请求错误不可比 $(date +%H:%M:%S)" >> runall.done
else
  touch V9_INVALID; echo "V9_INVALID $(date +%H:%M:%S)" >> runall.done
fi
echo "########## v9 manifest ##########"; cat v9_manifest.txt
