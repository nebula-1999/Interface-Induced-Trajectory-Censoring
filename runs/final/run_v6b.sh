#!/bin/bash
# v6 的 Qwen 插件三臂（重做版）。修三处：
#  1) 4096 → 8192：每轮最多生成 2048、最多 4 轮，加 prompt 与 Observation 后
#     4096 在第 2~3 轮必溢出。溢出会被记成 400，把「上下文不够」伪装成
#     「专用 parser 多轮失败」—— 而多轮能否成立正是本臂要测的东西。
#     （A4 用 4096 没暴露，是因为那里一次多轮都没走成，平均轮数 0.95。）
#  2) 冒烟同时检查 python 退出码、JSONL 行数、n_err，三者缺一不可。
#     只 grep 文本时，探针崩溃不打印汇总 → nerr 为空 → ${nerr:-0}=0 → 假通过。
#  3) 不再无条件 touch READY：逐臂写 manifest，全部满 100 行且 rc=0 才算成功，
#     否则写 V6B_INVALID。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f V6B_READY V6B_INVALID v6b_manifest.txt
N=100; ADP="--parser-adapter cross_family"
PL=$P/plugin
MD=/root/autodl-tmp/models

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
kill_vllm(){ ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; sleep 2
             ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; wait_port_free; }
serve(){ local M=$1 TAG=$2; shift 2
  wait_port_free || return 1
  setsid --fork python -m vllm.entrypoints.openai.api_server \
    --model "$M" --served-model-name "$M" --port 8000 \
    --gpu-memory-utilization 0.90 --max-model-len 8192 "$@" \
    < /dev/null > "vllm_v6b_$TAG.log" 2>&1
  for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && return 0
    ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
  echo "[$TAG] vLLM 未起"; grep -iE "error|Traceback|not a valid|plugin" "vllm_v6b_$TAG.log" | head -6; return 1; }

smoke(){  # 三重检查：退出码 + 行数 + n_err
  local TAG=$1 M=$2
  local OUT="traj_v6b_smoke_$TAG.jsonl"
  rm -f "$OUT"
  echo "  --- 冒烟 n=3 ($TAG) ---"
  local txt rc lines nerr
  txt=$(python probe_react_full.py --model "$M" --port 8000 --n 3 $ADP \
        --protocol fc --strength optional --out "$OUT" 2>&1); rc=$?
  echo "$txt" | grep -E "请求错误|服务端解析出调用|未被解析但确在尝试|调用了不存在的工具" || true
  lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  nerr=$(echo "$txt" | grep -oE "请求错误: [0-9]+" | grep -oE "[0-9]+" | head -1)
  echo "    退出码=$rc  行数=$lines  n_err=${nerr:-未打印}"
  if [ "$rc" != "0" ] || [ "$lines" != "3" ] || [ "${nerr:-x}" != "0" ]; then
    echo "  ✗ 冒烟未通过（三项须同时满足：rc=0 / 3 行 / n_err=0）→ 中止该臂"
    echo "$TAG SMOKE_FAIL rc=$rc lines=$lines nerr=${nerr:-none}" >> v6b_manifest.txt
    return 1
  fi
  echo "  ✓ 冒烟通过"; return 0; }

run(){ local TAG=$1 OUT=$2 M=$3
  echo "===== $TAG  $(date +%H:%M:%S) ====="
  python probe_react_full.py --model "$M" --port 8000 --n $N $ADP \
    --protocol fc --strength optional --out "$OUT" \
    2>&1 | grep -vE "Warning: You are sending|examples/s\]|it/s\]"
  local rc=${PIPESTATUS[0]}
  local lines=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "$TAG rc=$rc lines=$lines $(date +%H:%M:%S)" >> v6b_manifest.txt
  return 0; }

EXPECT=""
for S in 7B 1.5B 3B; do
  M="$MD/Qwen2.5-Coder-${S}-Instruct"
  TAG="Qwen${S}_fc_plugin"; OUT="traj_v6b_${TAG}.jsonl"
  EXPECT="$EXPECT $TAG:$OUT"
  [ -f "$M/config.json" ] || { echo "[$TAG] 缺模型"; echo "$TAG MISSING_MODEL" >> v6b_manifest.txt; continue; }
  echo "########## $TAG  hanXen <tools> parser（模板 few-shot 已改为 run_tests）##########"
  if serve "$M" "$TAG" --enable-auto-tool-choice \
       --tool-parser-plugin "$PL/qwen2_5_coder_tool_parser.py" \
       --tool-call-parser qwen2_5_coder \
       --chat-template "$PL/tool_chat_template_qwen2_5_coder.jinja"; then
    if smoke "$TAG" "$M"; then run "$TAG" "$OUT" "$M"; fi
  else
    echo "$TAG SERVE_FAIL" >> v6b_manifest.txt
  fi
  kill_vllm
done

# 只有全部预期产出都是 100 行且 rc=0 才算成功
ok=1
for e in $EXPECT; do
  tag=${e%%:*}; out=${e##*:}
  n=$(wc -l < "$out" 2>/dev/null || echo 0)
  grep -q "^$tag rc=0 lines=$N " v6b_manifest.txt 2>/dev/null || ok=0
  [ "$n" = "$N" ] || ok=0
done
if [ "$ok" = "1" ]; then touch V6B_READY; echo "V6B_READY $(date +%H:%M:%S)" >> runall.done
else touch V6B_INVALID; echo "V6B_INVALID $(date +%H:%M:%S)" >> runall.done; fi
echo "########## manifest ##########"; cat v6b_manifest.txt
