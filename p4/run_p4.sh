#!/bin/bash
# P4：跨家族的前瞻机制检验。臂表与预测见 p4/PREREGISTRATION.md（跑批前已提交）。
#
# 固定 parser，按运行前的模板分析预测其可观测性：
#   Qwen3 各档 + hermes          → 预测解析率 > 0 且随尺寸上升（阳性对照）
#   Granite-3.1-8B + hermes      → 预测 ≈ 0
#   Granite-3.1-8B + granite     → 预测 > 0（rescue 对照；没有它，0 无法解释）
#
# 继承 P2 踩过的坑：显式钉死解释器（非交互 shell 无 python）、BLAS 线程钉死、
# 用 /proc/net/tcp 判端口（本机无 ss/lsof）、下载完成标记（目录存在≠下载完成）、
# 产物验收看内容不看 rc、单实例锁。
set -u

P=/root/autodl-tmp/p2                 # 复用 P2 的探针与题集，保证判据同源
M=/root/autodl-tmp/models
E=/root/autodl-tmp/envs/p2/bin
PY="$E/python"
[ -x "$PY" ] || { echo "★ 找不到解释器 $PY"; exit 1; }
export PATH="$E:$PATH"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/autodl-tmp/hf
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT="${P4_OUT:-$P/runs_p4}"
PORT=8000
LOG=$P/p4_run.log
mkdir -p "$OUT" "$M"

LOCK=$P/run_p4.pid
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "★ 已有实例在跑（pid $(cat "$LOCK")）"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# tag|repo|模型目录|parser|额外 serve 参数
ARMS=(
  "qwen3-0.6B|Qwen/Qwen3-0.6B|Qwen3-0.6B|hermes|--default-chat-template-kwargs {\"enable_thinking\":false}"
  "qwen3-1.7B|Qwen/Qwen3-1.7B|Qwen3-1.7B|hermes|--default-chat-template-kwargs {\"enable_thinking\":false}"
  "qwen3-4B|Qwen/Qwen3-4B|Qwen3-4B|hermes|--default-chat-template-kwargs {\"enable_thinking\":false}"
  "qwen3-8B|Qwen/Qwen3-8B|Qwen3-8B|hermes|--default-chat-template-kwargs {\"enable_thinking\":false}"
  "granite-8B-hermes|ibm-granite/granite-3.1-8b-instruct|granite-3.1-8b-instruct|hermes|"
  "granite-8B-granite|ibm-granite/granite-3.1-8b-instruct|granite-3.1-8b-instruct|granite|"
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
port_free() { local h; h=$(printf '%04X' "$PORT"); ! awk -v p=":$h" '$2 ~ p"$" && $4=="0A"{f=1} END{exit !f}' /proc/net/tcp; }
wait_port() { for _ in $(seq 1 60); do port_free && return 0; sleep 2; done; say "  端口 $PORT 未释放"; return 1; }

serve() {  # $1=模型路径 $2=parser $3=额外参数
  wait_port || return 1
  say "  启动 vLLM: $(basename "$1")  parser=$2"
  # shellcheck disable=SC2086
  setsid --fork vllm serve "$1" --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser "$2" \
    --max-model-len 8192 --gpu-memory-utilization 0.85 $3 \
    </dev/null >>"$P/p4_vllm.log" 2>&1
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { say "  就绪（${i}0s 内）"; return 0; }
    if [ $i -gt 6 ] && ! pgrep -f "[v]llm serve" >/dev/null; then
      say "  ★ vLLM 已退出，根因："; grep -aE "Error|error:|Traceback" "$P/p4_vllm.log" | tail -4 | tee -a "$LOG"; return 1
    fi
    sleep 5
  done
  say "  ★ 启动超时"; return 1
}
teardown() { pkill -f "[v]llm serve" 2>/dev/null; wait_port; }

say "===== P4 开始 ====="
for entry in "${ARMS[@]}"; do
  IFS='|' read -r tag repo dir parser extra <<< "$entry"
  say "--- $tag ---"
  out="$OUT/traj_p4_${tag}.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 100 ]; then say "  已有合格产物，跳过"; continue; fi

  path="$M/$dir"
  if [ ! -f "$path/.p4_download_ok" ]; then
    say "  下载 $repo"
    if hf download "$repo" --local-dir "$path" >>"$LOG" 2>&1; then touch "$path/.p4_download_ok"
    else say "  ★ 下载失败，跳过"; continue; fi
  fi
  df -h /root/autodl-tmp | tail -1 | tee -a "$LOG"

  serve "$path" "$parser" "$extra" || { say "  ★ 服务启动失败，跳过"; teardown; continue; }
  say "  preflight"
  "$PY" "$P/preflight_toolcall.py" --port "$PORT" >>"$LOG" 2>&1
  say "  preflight rc=$?（rc≠0 不阻断：auto=0 正是被预测的结果之一）"

  say "  探针 n=100"
  ( cd "$P" && "$PY" "$P/probe_react_full.py" --model "$path" --port "$PORT" --n 100 \
      --protocol fc --strength optional --fc-schema terse \
      --parser-adapter cross_family --temperature 0.0 --seed 0 \
      --out "$out" >>"$LOG" 2>&1 )
  say "  探针 rc=$?"
  if [ -f "$out" ]; then
    nerr=$(grep -c "request_error" "$out" 2>/dev/null) || nerr=0
    say "  产物 $(wc -l < "$out") 行，请求错误 $nerr"
    [ "$nerr" -gt 5 ] && { say "  ★ 请求错误过多，本臂作废"; mv "$out" "$out.invalid"; }
  else
    say "  ★ 无产物"
  fi
  teardown
done
say "===== P4 结束 ====="
ls -la "$OUT" | tee -a "$LOG"
