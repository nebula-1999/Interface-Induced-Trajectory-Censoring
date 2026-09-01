#!/usr/bin/env bash
# P1：在标准 tool-use benchmark 上做五层测量。
#
# 核心设计：**不改 BFCL / tau-bench 的任何代码。**
# 在它们与 vLLM 之间插一个透明记录代理，把 base_url 指过去即可。
# 理由：五层测量需要的「模型产出了但服务端没解析走」那份文本，只在 parser
# 失败时留在 content 里（解析成功会搬进 tool_calls 并清空 content）。
# 在 HTTP 层记录就能拿到全部原料，不必 fork 别人的 harness。
#
# 坑（本项目实测，别踩第二遍）：
#   · ss/lsof/fuser 在 autodl 镜像里不存在 → 用 /proc/net/tcp 判端口
#   · run() 必须显式 return，否则 && 链断裂会静默跳过整条臂
#   · 每条臂先 preflight，配置静默失效烧几小时是本项目最贵的教训
#   · 代理必须透明转发，改一个字节测的就不是 benchmark 的真实行为了
set -u

P=${P1_DIR:-/root/autodl-tmp/p1}
M=${MODEL_DIR:-/root/autodl-tmp/models}
PORT=${VLLM_PORT:-8000}
PROXY=${PROXY_PORT:-8001}
LOG=$P/p1_run.log
mkdir -p "$P/logs"

# 用旧机上已有的权重，不下新模型（旧机数据盘只剩 4.6 GB）
# 格式： 标签  模型目录名  vLLM 的 tool-call-parser
ARMS=(
  "Qwen-7B    Qwen2.5-Coder-7B-Instruct   hermes"
  "Llama-8B   Meta-Llama-3.1-8B-Instruct  llama3_json"
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

port_free() {  # $1=端口
  local hex; hex=$(printf '%04X' "$1")
  ! awk -v p=":$hex" '$2 ~ p"$" && $4=="0A" {f=1} END{exit !f}' /proc/net/tcp
}
wait_free() { for _ in $(seq 1 60); do port_free "$1" && return 0; sleep 2; done
              say "  端口 $1 未释放"; return 1; }

serve() {  # $1=模型路径 $2=parser
  wait_free "$PORT" || return 1
  say "  vLLM 启动: $(basename "$1")  parser=$2"
  nohup vllm serve "$1" --port "$PORT" --served-model-name probe \
    --enable-auto-tool-choice --tool-call-parser "$2" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    >>"$P/vllm.log" 2>&1 &
  echo $! > "$P/vllm.pid"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { say "  就绪"; return 0; }
    sleep 5
  done
  say "  ★ 启动超时"; return 1
}

start_proxy() {  # $1=标签
  wait_free "$PROXY" || return 1
  nohup python3 "$P/toolcall_proxy.py" --listen "$PROXY" \
    --upstream "http://127.0.0.1:$PORT" \
    --log "$P/logs/$1.jsonl" --tag "$1" \
    >>"$P/proxy.log" 2>&1 &
  echo $! > "$P/proxy.pid"
  sleep 3
  curl -sf "http://127.0.0.1:$PROXY/v1/models" >/dev/null 2>&1 \
    && { say "  代理就绪 :$PROXY → :$PORT"; return 0; }
  say "  ★ 代理未就绪"; return 1
}

teardown() {
  for f in "$P/proxy.pid" "$P/vllm.pid"; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
  done
  wait_free "$PROXY"; wait_free "$PORT"
}

run_bench() {  # $1=标签
  local rc
  say "  preflight（直连 vLLM，不经代理——先确认服务本身是对的）"
  python3 "$P/preflight_toolcall.py" --port "$PORT" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "  ★ preflight rc=$rc，跳过本臂"; return "$rc"; fi

  say "  preflight（经代理——确认代理确实透明）"
  python3 "$P/preflight_toolcall.py" --port "$PROXY" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "  ★ 经代理的 preflight rc=$rc，代理不透明，停"; return "$rc"; fi

  # ---- BFCL ----
  if [ -d "$P/bfcl" ]; then
    say "  BFCL"
    ( cd "$P/bfcl" && OPENAI_BASE_URL="http://127.0.0.1:$PROXY/v1" OPENAI_API_KEY=dummy \
        bash run_bfcl.sh "$1" ) >>"$LOG" 2>&1
    say "  BFCL rc=$?"
  else
    say "  跳过 BFCL（$P/bfcl 不存在）"
  fi

  # ---- tau-bench ----
  if [ -d "$P/tau-bench" ]; then
    say "  tau-bench"
    ( cd "$P/tau-bench" && OPENAI_BASE_URL="http://127.0.0.1:$PROXY/v1" OPENAI_API_KEY=dummy \
        bash run_tau.sh "$1" ) >>"$LOG" 2>&1
    say "  tau-bench rc=$?"
  else
    say "  跳过 tau-bench（$P/tau-bench 不存在）"
  fi
  return 0
}

say "===== P1 开始 ====="
for entry in "${ARMS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  dir=$(echo "$entry" | awk '{print $2}')
  parser=$(echo "$entry" | awk '{print $3}')
  path="$M/$dir"
  say "--- $tag ---"
  if [ ! -d "$path" ]; then say "  ★ 权重不存在: $path，跳过"; continue; fi

  if serve "$path" "$parser"; then
    if start_proxy "$tag"; then
      run_bench "$tag"
    fi
  fi
  teardown
done

say "===== P1 结束 ====="
say "分析："
python3 "$P/analyze_p1.py" "$P/logs/"*.jsonl 2>&1 | tee -a "$LOG"
