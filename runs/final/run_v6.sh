#!/bin/bash
# A6：Llama-8B + strict:true —— 第四个替代解释。vLLM 文档称 auto 模式下不带
#     strict 时"参数可能违反 schema"，Llama 那 22% 填错参数名正落在这句话里。
# A5：Qwen + hanXen 的 <tools> parser 插件 —— 换对 parser 能拿回多少。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for _ in $(seq 1200); do [ -f V5_READY ] && break; sleep 30; done
rm -f V6_READY
N=100; ADP="--parser-adapter cross_family"
T=$P/templates; PL=$P/plugin

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2 LEN=$3 UTIL=$4; shift 4
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization "$UTIL" --max-model-len "$LEN" "$@" \
    < /dev/null > "vllm_v6_$TAG.log" 2>&1
  for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback|not a valid" "vllm_v6_$TAG.log" | head -5; return 1; }

# 这次冒烟真的会拦人：请求错误>0 即判定测量故障，中止该臂。
# 注意门槛是「服务端故障」而非「解析率低」—— 解析率低正是被研究的现象本身。
smoke(){ local TAG=$1; shift
  echo "  --- 冒烟 n=3 ($TAG) ---"
  local out; out=$(python probe_react_full.py --port 8000 --n 3 $ADP \
      --out "traj_v6_smoke_$TAG.jsonl" "$@" 2>&1)
  echo "$out" | grep -E "请求错误|服务端解析出调用|未被解析但确在尝试|L1 严格"
  local nerr; nerr=$(echo "$out" | grep -oE "请求错误: [0-9]+" | grep -oE "[0-9]+" | head -1)
  if [ "${nerr:-0}" != "0" ]; then
    echo "  ✗ 冒烟有 $nerr 个请求错误 → 判定测量故障，中止该臂"
    echo "$TAG SMOKE_FAIL nerr=$nerr $(date +%H:%M:%S)" >> v6_progress.txt
    return 1
  fi
  echo "  ✓ 冒烟无请求错误"; return 0; }

run(){ local TAG=$1 OUT=$2; shift 2
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$N" ]; then echo "[$TAG] 已完成，跳过"; return 0; fi
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --port 8000 --n $N $ADP --out "$OUT" "$@" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  echo "$TAG rc=${PIPESTATUS[0]} lines=$(wc -l < "$OUT" 2>/dev/null || echo 0) $(date +%H:%M:%S)" >> v6_progress.txt
  return 0; }

MD=/root/autodl-tmp/models
LL=$MD/Llama-3.1-8B-Instruct

############ A6：Llama + strict（决定第 5 节是否成立）############
echo "########## A6  Llama-3.1-8B  官方配置 + strict:true ##########"
if serve "$LL" ll_strict 8192 0.85 --enable-auto-tool-choice --tool-call-parser llama3_json \
     --chat-template "$T/tool_chat_template_llama3.1_json.jinja"; then
  if smoke ll_strict --model "$LL" --protocol fc --strength optional --fc-schema strict; then
    run Llama8B_fc_strict traj_v6_Llama8B_fc_strict.jsonl \
      --model "$LL" --protocol fc --strength optional --fc-schema strict
  fi
fi
kill_vllm

############ A5：Qwen + <tools> parser 插件 ############
for S in 7B 1.5B 3B; do
  M="$MD/Qwen2.5-Coder-${S}-Instruct"
  [ -f "$M/config.json" ] || { echo "[Qwen$S] 缺模型"; continue; }
  echo "########## A5  Qwen2.5-Coder-$S  hanXen <tools> parser ##########"
  if serve "$M" "qwplug_$S" 4096 0.95 --enable-auto-tool-choice \
       --tool-parser-plugin "$PL/qwen2_5_coder_tool_parser.py" \
       --tool-call-parser qwen2_5_coder \
       --chat-template "$PL/tool_chat_template_qwen2_5_coder.jinja"; then
    if smoke "qwplug_$S" --model "$M" --protocol fc --strength optional; then
      run "Qwen${S}_fc_plugin" "traj_v6_Qwen${S}_fc_plugin.jsonl" \
        --model "$M" --protocol fc --strength optional
    fi
  fi
  kill_vllm
done

touch V6_READY
echo "V6_READY $(date +%H:%M:%S)" >> runall.done
