#!/usr/bin/env bash
# P1 冒烟：只验证「代理这一层能不能在真实 benchmark 上工作」。
#
# 为什么先冒烟：代理从没在真实 benchmark 上跑过。若 BFCL 的调用方式与预期
# 不同——自带重试、走 /v1/completions 而非 /v1/chat/completions、
# 或并发发请求——小规模几分钟就暴露，全量就是几小时。
#
# 本脚本**不产出论文数据**，只回答三个是非题：
#   1. benchmark 能否通过代理正常完成
#   2. 代理日志是否非空、且 n_tools_offered > 0
#   3. analyze_p1.py 能否出表
set -u
P=/root/autodl-tmp/p1
M=/root/autodl-tmp/models
PORT=8000; PROXY=8001
VLLM=/root/code-venv/bin
BENCH=/root/bench-venv/bin
# vLLM 会 fork 调 ninja 做 inductor 编译，那次查找走 PATH，绝对路径救不了
export PATH="$VLLM:$PATH"
TAG=smoke-Llama8B-FC
mkdir -p "$P/logs"
rm -f "$P/logs/$TAG.jsonl"

say(){ echo "[$(date +%H:%M:%S)] $*"; }
port_free(){ local h; h=$(printf '%04X' "$1")
  ! awk -v p=":$h" '$2 ~ p"$" && $4=="0A"{f=1} END{exit !f}' /proc/net/tcp; }
wait_free(){ for _ in $(seq 1 60); do port_free "$1" && return 0; sleep 2; done
  say "端口 $1 未释放"; return 1; }
cleanup(){ for f in "$P/proxy.pid" "$P/vllm.pid"; do
    [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  wait_free "$PROXY"; wait_free "$PORT"; }
trap cleanup EXIT

say "1/5 起 vLLM（Llama-3.1-8B + llama3_json）"
wait_free "$PORT" || exit 1
nohup "$VLLM/vllm" serve "$M/Llama-3.1-8B-Instruct" \
  --port "$PORT" --served-model-name meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --max-model-len 8192 --gpu-memory-utilization 0.90 \
  >"$P/vllm.log" 2>&1 &
echo $! > "$P/vllm.pid"
vpid=$(cat "$P/vllm.pid")
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  if ! kill -0 "$vpid" 2>/dev/null; then
    say "★ vLLM 进程已退出，不再等待。根因："
    grep -E "Error|FileNotFoundError|RuntimeError" "$P/vllm.log" | tail -5
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { say "★ vLLM 未就绪"; exit 1; }
say "    就绪"

say "2/5 起记录代理"
wait_free "$PROXY" || exit 1
nohup "$VLLM/python" "$P/toolcall_proxy.py" --listen "$PROXY" \
  --upstream "http://127.0.0.1:$PORT" --log "$P/logs/$TAG.jsonl" --tag "$TAG" \
  >"$P/proxy.log" 2>&1 &
echo $! > "$P/proxy.pid"
sleep 4
curl -sf "http://127.0.0.1:$PROXY/v1/models" >/dev/null 2>&1 || { say "★ 代理未就绪"; cat "$P/proxy.log"; exit 1; }
say "    就绪 :$PROXY → :$PORT"

say "3/5 双 preflight（直连 / 经代理，后者验代理透明）"
"$VLLM/python" "$P/preflight_toolcall.py" --port "$PORT"  || { say "★ 直连 preflight 失败"; exit 1; }
"$VLLM/python" "$P/preflight_toolcall.py" --port "$PROXY" || { say "★ 经代理 preflight 失败——代理不透明"; exit 1; }
say "    两道都过，代理透明"

say "4/5 BFCL 最小子集"
# ↓↓↓ 这一段按实际 CLI 填，装完后确认 ↓↓↓
mkdir -p "$P/bfcl-run"; cd "$P/bfcl-run"
# base_url 直接覆盖，指向记录代理而非 vLLM
export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:$PROXY/v1"
export OPENAI_API_KEY=dummy
# --skip-server-setup: BFCL 不自己起服务，用我们已经起好的
# --num-threads 1: 并发会让代理日志与评分难以按序对齐，且冒烟不赶时间
"$BENCH/bfcl" generate \
  --model meta-llama/Llama-3.1-8B-Instruct-FC \
  --test-category simple_python \
  --skip-server-setup --num-threads 1 --temperature 0.0
say "    BFCL rc=$?"

say "5/5 验收"
n=$(wc -l < "$P/logs/$TAG.jsonl" 2>/dev/null || echo 0)
say "    代理记录 $n 条"
[ "$n" -eq 0 ] && { say "★ 代理日志为空——benchmark 没走代理，检查 OPENAI_BASE_URL"; exit 1; }
"$VLLM/python" -c "
import json;R=[json.loads(l) for l in open('$P/logs/$TAG.jsonl') if l.strip()]
t=sum(1 for r in R if r.get('n_tools_offered'))
print(f'    带 tools 的调用 {t}/{len(R)}')
print('    ★ 为 0 说明 benchmark 没走 tools 分支，配置错了' if t==0 else '    OK')
"
"$VLLM/python" "$P/analyze_p1.py" "$P/logs/$TAG.jsonl"
