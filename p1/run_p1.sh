#!/usr/bin/env bash
# P1: standard-benchmark replication of the five-layer interface funnel.
# Runs a paired BFCL-v4 subset through vLLM's real Chat Completions parser:
#   arm 1: documented hermes configuration
#   arm 2: community Qwen2.5-Coder parser + its paired chat template
set -Eeuo pipefail

P1_ROOT=${P1_ROOT:-/root/autodl-tmp/p1}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct}
CODE_VENV=${CODE_VENV:-/root/code-venv}
BFCL_SRC=${BFCL_SRC:-/root/p1-bfcl}
BFCL_VENV=${BFCL_VENV:-/root/p1-bfcl-venv}
BFCL_COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
RUN_ROOT=$P1_ROOT/bfcl_run
LOG_ROOT=$P1_ROOT/logs
SERVER_PORT=8000
PROXY_PORT=8001
THREADS=${P1_THREADS:-8}
N_SINGLE=${P1_N_SINGLE:-100}
N_MULTI=${P1_N_MULTI:-100}
SEED=${P1_SEED:-20260901}
MAIN_LOG=$P1_ROOT/p1_run.log

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
touch "$MAIN_LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }
die() { say "FATAL: $*"; touch "$P1_ROOT/P1_INVALID"; exit 1; }

server_pid=""
proxy_pid=""
cleanup() {
  if [[ -n "$proxy_pid" ]]; then kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi
  proxy_pid=""; server_pid=""
}
trap cleanup EXIT INT TERM

wait_url() {
  local url=$1 pid=$2
  for _ in $(seq 1 240); do
    curl -sf --max-time 3 "$url" >/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 3
  done
  return 1
}

port_open() {
  local port=$1 hex
  hex=$(printf '%04X' "$port")
  awk -v suffix=":$hex" '$2 ~ suffix"$" && $4=="0A" {found=1} END{exit !found}' /proc/net/tcp
}

wait_port_closed() {
  local port=$1
  for _ in $(seq 1 60); do port_open "$port" || return 0; sleep 1; done
  return 1
}

bootstrap_bfcl() {
  say "Bootstrap BFCL at pinned commit $BFCL_COMMIT"
  if [[ ! -d "$BFCL_SRC/.git" ]]; then
    mkdir -p "$BFCL_SRC"
    git -C "$BFCL_SRC" init -q
    git -C "$BFCL_SRC" remote add origin https://github.com/ShishirPatil/gorilla.git
    git -C "$BFCL_SRC" config core.sparseCheckout true
    echo '/berkeley-function-call-leaderboard/' > "$BFCL_SRC/.git/info/sparse-checkout"
  fi
  if ! git -C "$BFCL_SRC" cat-file -e "$BFCL_COMMIT^{commit}" 2>/dev/null; then
    git -C "$BFCL_SRC" fetch -q --depth=1 --filter=blob:none origin "$BFCL_COMMIT"
  fi
  git -C "$BFCL_SRC" checkout -q --detach "$BFCL_COMMIT"

  if [[ ! -x "$BFCL_VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$BFCL_VENV"
  fi
  "$BFCL_VENV/bin/python" -m pip install -q --upgrade pip
  "$BFCL_VENV/bin/python" -m pip install -q -e "$BFCL_SRC/berkeley-function-call-leaderboard" aiohttp

  local registered
  registered=$(PYTHONPATH="$P1_ROOT" P1_BFCL_MODEL_ID=P1-Smoke-FC \
    "$BFCL_VENV/bin/python" -c 'import bfcl_registration as r; print(r.REGISTERED_MODEL)')
  [[ "$registered" == "P1-Smoke-FC" ]] || die "custom BFCL handler did not register"
}

start_server() {
  local slug=$1 served=$2; shift 2
  cleanup
  wait_port_closed "$SERVER_PORT" || die "port $SERVER_PORT is already occupied by a non-P1 process"
  wait_port_closed "$PROXY_PORT" || die "port $PROXY_PORT is already occupied by a non-P1 process"
  local server_log=$LOG_ROOT/vllm_${slug}.log
  say "Start vLLM arm=$slug served_model=$served"
  "$CODE_VENV/bin/vllm" serve "$MODEL_PATH" \
    --port "$SERVER_PORT" --served-model-name "$served" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice "$@" >>"$server_log" 2>&1 &
  server_pid=$!
  wait_url "http://127.0.0.1:$SERVER_PORT/v1/models" "$server_pid" || {
    tail -80 "$server_log" | tee -a "$MAIN_LOG"; die "vLLM failed for $slug"; }
}

start_proxy() {
  local slug=$1 tag=$2 logfile=$3
  wait_port_closed "$PROXY_PORT" || die "proxy port $PROXY_PORT is occupied"
  "$BFCL_VENV/bin/python" "$P1_ROOT/toolcall_proxy.py" \
    --listen "$PROXY_PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
    --log "$logfile" --tag "$tag" >>"$LOG_ROOT/proxy_${slug}.log" 2>&1 &
  proxy_pid=$!
  wait_url "http://127.0.0.1:$PROXY_PORT/v1/models" "$proxy_pid" || die "proxy failed for $slug"
}

run_bfcl() {
  local registry=$1 project_root=$2 manifest=$3 threads=$4
  mkdir -p "$project_root"
  cp "$manifest" "$project_root/test_case_ids_to_generate.json"
  : > "$project_root/.env"
  BFCL_PROJECT_ROOT="$project_root" \
  OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY=EMPTY \
  PYTHONPATH="$P1_ROOT" P1_BFCL_MODEL_ID="$registry" P1_SERVED_MODEL="$P1_SERVED_MODEL" \
  P1_SEED=0 P1_MAX_TOKENS=2048 \
    "$BFCL_VENV/bin/python" -m bfcl_eval.__main__ generate \
      --model "$registry" --run-ids --num-threads "$threads" \
      --temperature 0 --include-input-log >>"$MAIN_LOG" 2>&1
  BFCL_PROJECT_ROOT="$project_root" \
  PYTHONPATH="$P1_ROOT" P1_BFCL_MODEL_ID="$registry" P1_SERVED_MODEL="$P1_SERVED_MODEL" \
    "$BFCL_VENV/bin/python" -m bfcl_eval.__main__ evaluate \
      --model "$registry" --test-category simple_python,multi_turn_base \
      --partial-eval >>"$MAIN_LOG" 2>&1
}

validate_project() {
  local project_root=$1 registry=$2 expected=$3
  local model_dir=${registry//\//_}
  local result_n score_n proxy_n
  read -r result_n score_n < <("$BFCL_VENV/bin/python" - "$project_root" "$model_dir" <<'PY'
import sys
from pathlib import Path
root, model = Path(sys.argv[1]), sys.argv[2]
results = sum(sum(bool(line.strip()) for line in p.open())
              for p in (root / "result" / model).rglob("BFCL_v4_*_result.json"))
scores = sum(max(0, sum(bool(line.strip()) for line in p.open()) - 1)
             for p in (root / "score" / model).rglob("BFCL_v4_*_score.json"))
print(results, scores)
PY
  )
  [[ "$result_n" -eq "$expected" ]] || die "$registry result count $result_n != $expected"
  [[ "$score_n" -eq "$expected" ]] || die "$registry score count $score_n != $expected"
}

validate_proxy() {
  local logfile=$1 expected_cases=$2
  "$BFCL_VENV/bin/python" - "$logfile" "$expected_cases" <<'PY'
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
cases = {r.get("case_id") for r in rows if r.get("case_id")}
errors = [r for r in rows if r.get("status", 200) >= 400 or r.get("_parse_error")]
if len(cases) != int(sys.argv[2]) or errors:
    raise SystemExit(f"proxy validation failed: cases={len(cases)} expected={sys.argv[2]} errors={len(errors)}")
print(f"proxy validation: cases={len(cases)} requests={len(rows)} errors=0")
PY
}

run_arm() {
  local slug=$1 registry=$2 served=$3; shift 3
  local full_log=$LOG_ROOT/bfcl_${slug}.jsonl
  local smoke_log=$LOG_ROOT/bfcl_${slug}_smoke.jsonl
  local arm_marker=$RUN_ROOT/${slug}.READY
  if [[ -f "$arm_marker" ]]; then say "Skip completed arm $slug"; return 0; fi

  export P1_SERVED_MODEL=$served
  start_server "$slug" "$served" "$@"

  rm -f "$smoke_log"
  start_proxy "${slug}_smoke" "SMOKE-${registry}" "$smoke_log"
  run_bfcl "$registry" "$RUN_ROOT/smoke_${slug}" "$RUN_ROOT/smoke_manifest.json" 1
  validate_project "$RUN_ROOT/smoke_${slug}" "$registry" 2
  validate_proxy "$smoke_log" 2 || die "$slug smoke proxy validation failed"
  kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true; proxy_pid=""
  wait_port_closed "$PROXY_PORT" || die "proxy port did not close after smoke"
  say "Smoke passed for $slug"

  start_proxy "$slug" "$registry" "$full_log"
  run_bfcl "$registry" "$RUN_ROOT" "$RUN_ROOT/test_case_ids_to_generate.json" "$THREADS"
  validate_project "$RUN_ROOT" "$registry" $((N_SINGLE + N_MULTI))
  validate_proxy "$full_log" $((N_SINGLE + N_MULTI)) || die "$slug full proxy validation failed"
  touch "$arm_marker"
  say "Completed arm $slug"
  cleanup
}

say "===== P1 begin ====="
[[ -f "$MODEL_PATH/config.json" ]] || die "model missing: $MODEL_PATH"
[[ -x "$CODE_VENV/bin/vllm" ]] || die "vLLM executable missing: $CODE_VENV/bin/vllm"
vllm_version=$($CODE_VENV/bin/python -c 'import importlib.metadata as m; print(m.version("vllm"))')
[[ "$vllm_version" == "0.27.1" ]] || die "vLLM must be 0.27.1, got $vllm_version"
[[ -f "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" ]] || die "community parser missing"
[[ -f "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja" ]] || die "community template missing"

bootstrap_bfcl
BFCL_DATA=$BFCL_SRC/berkeley-function-call-leaderboard/bfcl_eval/data
"$BFCL_VENV/bin/python" "$P1_ROOT/make_manifest.py" --data-dir "$BFCL_DATA" \
  --out "$RUN_ROOT/test_case_ids_to_generate.json" --n-single "$N_SINGLE" \
  --n-multi "$N_MULTI" --seed "$SEED" | tee -a "$MAIN_LOG"
"$BFCL_VENV/bin/python" "$P1_ROOT/make_manifest.py" --data-dir "$BFCL_DATA" \
  --out "$RUN_ROOT/smoke_manifest.json" --n-single 1 --n-multi 1 --seed "$SEED" | tee -a "$MAIN_LOG"

{
  echo "bfcl_commit=$BFCL_COMMIT"
  echo "vllm_version=$vllm_version"
  echo "model_path=$MODEL_PATH"
  echo "manifest_seed=$SEED inference_seed=0 n_single=$N_SINGLE n_multi=$N_MULTI threads=$THREADS"
  sha256sum "$RUN_ROOT/test_case_ids_to_generate.json" "$P1_ROOT/bfcl_registration.py" \
    "$P1_ROOT/toolcall_proxy.py" "$P1_ROOT/analyze_p1.py" \
    "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" \
    "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja"
} > "$RUN_ROOT/provenance.txt"

run_arm hermes P1-Qwen2.5-Coder-7B-Hermes-FC p1-qwen25-coder7b-hermes \
  --tool-call-parser hermes
run_arm repaired P1-Qwen2.5-Coder-7B-Repaired-FC p1-qwen25-coder7b-repaired \
  --tool-parser-plugin "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" \
  --tool-call-parser qwen2_5_coder \
  --chat-template "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja"

"$BFCL_VENV/bin/python" "$P1_ROOT/analyze_p1.py" "$LOG_ROOT/bfcl_hermes.jsonl" \
  "$LOG_ROOT/bfcl_repaired.jsonl" --result-root "$RUN_ROOT/result" \
  --score-root "$RUN_ROOT/score" | tee "$RUN_ROOT/P1_SUMMARY.txt"
touch "$P1_ROOT/P1_READY"
say "===== P1 complete ====="
