#!/usr/bin/env bash
# τ-bench 冒烟：3 个 task，验证管线通、且 agent 流量确实经过记录代理。
# 冒烟不过不要跑全量（115 task × 多轮，代价很大）。
#
# 分流方式（已验证 litellm 1.99.0 支持）：
#   agent      → hosted_vllm provider → HOSTED_VLLM_API_BASE → 记录代理 :8001 → vLLM :8000
#   user sim   → openai provider      → OPENAI_API_BASE      → vLLM :8010（跨臂固定）
set -uo pipefail
ARM="${1:-documented}"                     # documented | repaired
TAU=/root/tau-bench; VENV=/root/tau-venv; P1=/root/autodl-tmp/p1
AGENT=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
USERM=/root/autodl-tmp/models/Llama-3.1-8B-Instruct
# tau-bench 用模型名拼结果文件名，模型**路径**里的 / 会拼出非法路径并在写盘时报
# FileNotFoundError。给两个服务各起一个无斜杠的 served name，命令行也传该名字。
AGENT_NAME=tau-agent-qwen25-coder-7b
USER_NAME=tau-user-llama31-8b
OUT=/root/autodl-tmp/tau; mkdir -p "$OUT"
CV=/root/code-venv/bin                     # vLLM 在 code-venv 里
# PATH 必须含 venv 的 bin，不能只用绝对路径调 vllm：vLLM 启动时 FlashInfer 会
# **按可执行文件名**去找 ninja，那次查找走 PATH，绝对路径救不了。
# run_p1.sh 里有同样的注释，本脚本第一版漏了这一行，报
# FileNotFoundError: 'ninja' 而 vLLM 引擎初始化失败。
export PATH="$CV:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

pkill -f '[v]llm serve' 2>/dev/null; pkill -f '[t]oolcall_proxy' 2>/dev/null; sleep 8

up() { for i in $(seq 1 90); do curl -sf "$1/v1/models" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

echo "[tau] 起用户模拟器 :8010（跨臂固定，绝不随臂变化）"
setsid --fork "$CV/vllm" serve "$USERM" --port 8010 --served-model-name "$USER_NAME" \
  --max-model-len 16384 --gpu-memory-utilization 0.35 \
  </dev/null >>"$OUT/vllm_user.log" 2>&1
up http://127.0.0.1:8010 || { echo "[tau] ★ 用户模拟器未就绪"; tail -5 "$OUT/vllm_user.log"; exit 1; }

case "$ARM" in
  documented) EXTRA=(--tool-call-parser hermes) ;;
  repaired)   EXTRA=(--tool-parser-plugin "$P1/plugin/qwen2_5_coder_tool_parser.py"
                     --tool-call-parser qwen2_5_coder
                     --chat-template "$P1/plugin/tool_chat_template_qwen2_5_coder.jinja") ;;
  *) echo "未知臂 $ARM"; exit 1 ;;
esac
echo "[tau] 起 agent :8000  臂=$ARM"
setsid --fork "$CV/vllm" serve "$AGENT" --port 8000 --served-model-name "$AGENT_NAME" --enable-auto-tool-choice \
  --max-model-len 32768 --gpu-memory-utilization 0.45 "${EXTRA[@]}" \
  </dev/null >>"$OUT/vllm_agent_$ARM.log" 2>&1
up http://127.0.0.1:8000 || { echo "[tau] ★ agent 未就绪"; tail -5 "$OUT/vllm_agent_$ARM.log"; exit 1; }

echo "[tau] 起记录代理 :8001 → :8000"
LOGJ="$OUT/proxy_$ARM.jsonl"; rm -f "$LOGJ"
setsid --fork "$CV/python" "$P1/toolcall_proxy.py" --listen 8001 \
  --upstream http://127.0.0.1:8000 --log "$LOGJ" --tag "TAU-$ARM" \
  </dev/null >>"$OUT/proxy_$ARM.log" 2>&1
sleep 5
curl -sf http://127.0.0.1:8001/v1/models >/dev/null || { echo "[tau] ★ 代理未就绪"; exit 1; }

export HOSTED_VLLM_API_BASE=http://127.0.0.1:8001/v1  HOSTED_VLLM_API_KEY=EMPTY
export OPENAI_API_BASE=http://127.0.0.1:8010/v1       OPENAI_API_KEY=EMPTY
echo "[tau] 跑 3 个 task"
mkdir -p "$TAU/results"
cd "$TAU" && "$VENV/bin/python" run.py \
  --agent-strategy tool-calling --env retail \
  --model "$AGENT_NAME" --model-provider hosted_vllm \
  --user-model "$USER_NAME" --user-model-provider openai --user-strategy llm \
  --max-concurrency 1 --task-ids 0 1 2 2>&1 | tail -25

echo "[tau] ===== 冒烟验收 ====="
n=$(wc -l < "$LOGJ" 2>/dev/null || echo 0)
echo "  代理记录 $n 条  ← 为 0 说明 agent 没走代理，配置错了"
"$VENV/bin/python" - "$LOGJ" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()] if len(sys.argv)>1 else []
if rows:
    wt=sum(1 for r in rows if r.get("n_tools_offered"))
    pc=sum(1 for r in rows if r.get("parsed_tool_calls"))
    print(f"  带 tools 的请求 {wt}/{len(rows)}   ← 为 0 说明没走 function-calling 分支")
    print(f"  服务端解析出调用 {pc}/{len(rows)}")
PY
