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
SMOKE_PER_CATEGORY=${P1_SMOKE_PER_CATEGORY:-2}
SMOKE_EXPECTED=$((SMOKE_PER_CATEGORY * 2))
MAX_MODEL_LEN=${P1_MAX_MODEL_LEN:-16384}
SEED=${P1_SEED:-20260901}
MAIN_LOG=$P1_ROOT/p1_run.log

# FlashInfer invokes `ninja` by executable name while vLLM starts.  Detached
# AutoDL shells do not inherit the interactive environment's venv PATH, so
# keep the runtime tool beside vLLM and make that location explicit.
export PATH="$CODE_VENV/bin:$PATH"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
touch "$MAIN_LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }
die() { say "FATAL: $*"; touch "$P1_ROOT/P1_INVALID"; exit 1; }

# README 承诺「失败即 fail closed 并产生 P1_INVALID」，但 die() 只在显式调用时
# 才打标记。实测 bootstrap 阶段 git fetch 连不上 github 时，set -e 让脚本直接退出，
# 既没有 P1_INVALID 也没有 FATAL——外面看起来像「跑完了」。补一个 ERR trap。
on_err() {
  local rc=$? line=${BASH_LINENO[0]:-?}
  say "FATAL: unexpected failure at line $line (exit $rc)"
  touch "$P1_ROOT/P1_INVALID"
}
trap on_err ERR

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
    # 这台机器直连 github 极慢（curl 拉 info/refs 单次就要 32s），git 默认在
    # 128s 上超时，症状是 bootstrap 静默失败。放宽低速阈值、重试，再退到镜像。
    # 镜像只影响**下载途径**，commit 哈希不变，checkout 后仍然逐字节可验。
    local urls=("https://github.com/ShishirPatil/gorilla.git"
                "${P1_GIT_MIRROR:-https://ghproxy.com/https://github.com/ShishirPatil/gorilla.git}")
    local ok=0 u
    for u in "${urls[@]}"; do
      for attempt in 1 2; do
        say "Fetch $BFCL_COMMIT from $u (attempt $attempt)"
        if git -C "$BFCL_SRC" \
             -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=600 \
             fetch -q --depth=1 --filter=blob:none "$u" "$BFCL_COMMIT"; then
          ok=1; break
        fi
        say "  fetch failed, retrying"
      done
      [[ $ok -eq 1 ]] && break
    done
    [[ $ok -eq 1 ]] || die "cannot fetch $BFCL_COMMIT from any source (set P1_GIT_MIRROR)"
  fi
  git -C "$BFCL_SRC" checkout -q --detach "$BFCL_COMMIT"

  if [[ ! -x "$BFCL_VENV/bin/python" ]]; then
    # `python3` is NOT on PATH in a non-interactive shell on this box (PATH is
    # /usr/local/sbin:...:/snap/bin, and there is no /usr/bin/python3).  Since
    # run_p1.sh is launched detached via setsid, relying on PATH would abort
    # bootstrap under `set -e` before a single case runs.  Resolve explicitly.
    local py3=${P1_PYTHON3:-}
    if [[ -z "$py3" ]]; then
      for cand in /root/miniconda3/bin/python3 /usr/bin/python3 "$(command -v python3 || true)"; do
        [[ -n "$cand" && -x "$cand" ]] && { py3=$cand; break; }
      done
    fi
    [[ -n "$py3" ]] || die "no python3 found to create $BFCL_VENV (set P1_PYTHON3)"
    say "Create BFCL venv with $py3"
    "$py3" -m venv --system-site-packages "$BFCL_VENV"
  fi
  "$BFCL_VENV/bin/python" -m pip install -q --upgrade pip
  # qwen-agent imports soundfile eagerly from its utility module, but does not
  # declare it in its package dependencies.  BFCL's model registry imports the
  # Qwen handler even for our custom OpenAI-compatible arm, so bootstrap must
  # install this otherwise-optional dependency explicitly.
  "$BFCL_VENV/bin/python" -m pip install -q \
    -e "$BFCL_SRC/berkeley-function-call-leaderboard" aiohttp soundfile==0.13.1

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
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.90 \
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
  local target_manifest="$project_root/test_case_ids_to_generate.json"
  # The full run already keeps its manifest at this exact destination.  GNU
  # cp treats copying a file onto itself as an error, which trips `set -e`
  # immediately after a successful smoke test.
  if [[ "$(readlink -f "$manifest")" != "$(readlink -m "$target_manifest")" ]]; then
    cp "$manifest" "$target_manifest"
  fi
  : > "$project_root/.env"
  BFCL_PROJECT_ROOT="$project_root" \
  OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY=EMPTY \
  PYTHONPATH="$P1_ROOT" P1_BFCL_MODEL_ID="$registry" P1_SERVED_MODEL="$P1_SERVED_MODEL" \
  P1_SEED=0 P1_MAX_TOKENS=2048 \
    "$BFCL_VENV/bin/python" -m bfcl_eval.__main__ generate \
      --model "$registry" --run-ids --num-threads "$threads" \
      --temperature 0 --include-input-log >>"$MAIN_LOG" 2>&1
  BFCL_PROJECT_ROOT="$project_root" \
  OPENAI_BASE_URL="http://127.0.0.1:$PROXY_PORT/v1" OPENAI_API_KEY=EMPTY \
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
import json, sys
from pathlib import Path
root, model = Path(sys.argv[1]), sys.argv[2]
results = sum(sum(bool(line.strip()) for line in p.open())
              for p in (root / "result" / model).rglob("BFCL_v4_*_result.json"))
# BFCL score JSONL stores one aggregate header plus detail rows for failures;
# correct cases do not receive detail rows.  Counting lines-minus-header thus
# measures the number of failures, not the number of scored cases.  The
# aggregate header's total_count is the authoritative completeness field.
scores = 0
for p in (root / "score" / model).rglob("BFCL_v4_*_score.json"):
    with p.open() as fh:
        header = json.loads(next(line for line in fh if line.strip()))
    scores += int(header["total_count"])
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
  local full_project=$RUN_ROOT/full_${slug}
  local arm_marker=$RUN_ROOT/${slug}.READY
  if [[ -f "$arm_marker" ]]; then say "Skip completed arm $slug"; return 0; fi

  # An interrupted or intentionally superseded run must not be appended to the
  # next measurement.  Preserve previous BFCL outputs under a timestamped
  # archive, and start both HTTP capture logs from an empty file.
  local model_dir=${registry//\//_}
  local archive_root=$RUN_ROOT/archive/$(date +%Y%m%d_%H%M%S)_${slug}
  # Both project roots must be cleared, not just the full one.  BFCL's generate
  # step skips cases already present in the result file, so a leftover smoke
  # project makes it emit zero HTTP requests; the proxy log then stays empty and
  # the smoke proxy check fails with no indication that the cause is stale
  # output rather than a broken interface.  (Observed 22:04 on 2026-09-01.)
  local smoke_project=$RUN_ROOT/smoke_${slug}
  local tree proj
  for proj in "$full_project" "$smoke_project"; do
    for tree in result score; do
      if [[ -d "$proj/$tree/$model_dir" ]]; then
        mkdir -p "$archive_root/$(basename "$proj")/$tree"
        mv "$proj/$tree/$model_dir" "$archive_root/$(basename "$proj")/$tree/"
      fi
    done
  done
  local capture
  for capture in "$full_log" "$smoke_log"; do
    if [[ -f "$capture" ]]; then
      mkdir -p "$archive_root/logs"
      mv "$capture" "$archive_root/logs/"
    fi
  done

  export P1_SERVED_MODEL=$served
  start_server "$slug" "$served" "$@"

  start_proxy "${slug}_smoke" "SMOKE-${registry}" "$smoke_log"
  run_bfcl "$registry" "$RUN_ROOT/smoke_${slug}" "$RUN_ROOT/smoke_manifest.json" 1
  validate_project "$RUN_ROOT/smoke_${slug}" "$registry" "$SMOKE_EXPECTED"
  validate_proxy "$smoke_log" "$SMOKE_EXPECTED" || die "$slug smoke proxy validation failed"
  kill "$proxy_pid" 2>/dev/null || true; wait "$proxy_pid" 2>/dev/null || true; proxy_pid=""
  wait_port_closed "$PROXY_PORT" || die "proxy port did not close after smoke"
  say "Smoke passed for $slug"

  start_proxy "$slug" "$registry" "$full_log"
  run_bfcl "$registry" "$full_project" "$RUN_ROOT/test_case_ids_to_generate.json" "$THREADS"
  validate_project "$full_project" "$registry" $((N_SINGLE + N_MULTI))
  validate_proxy "$full_log" $((N_SINGLE + N_MULTI)) || die "$slug full proxy validation failed"
  touch "$arm_marker"
  say "Completed arm $slug"
  cleanup
}

say "===== P1 begin ====="
[[ -f "$MODEL_PATH/config.json" ]] || die "model missing: $MODEL_PATH"
[[ -x "$CODE_VENV/bin/vllm" ]] || die "vLLM executable missing: $CODE_VENV/bin/vllm"
if [[ ! -x "$CODE_VENV/bin/ninja" ]]; then
  say "Install pinned ninja runtime into $CODE_VENV"
  "$CODE_VENV/bin/python" -m pip install -q ninja==1.13.2
fi
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
  --out "$RUN_ROOT/smoke_manifest.json" --n-single "$SMOKE_PER_CATEGORY" \
  --n-multi "$SMOKE_PER_CATEGORY" --seed "$SEED" | tee -a "$MAIN_LOG"

{
  echo "bfcl_commit=$BFCL_COMMIT"
  echo "vllm_version=$vllm_version"
  echo "model_path=$MODEL_PATH"
  echo "manifest_seed=$SEED inference_seed=0 n_single=$N_SINGLE n_multi=$N_MULTI threads=$THREADS max_model_len=$MAX_MODEL_LEN"
  # run_p1.sh 与 validate_p1.py 原来不在这张清单里，于是干净重跑那次的
  # provenance.txt 恰恰没钉住产出它的脚本本身；恢复版里那两行是当时手工补的。
  # sitecustomize/register_qwen_coder 决定注册了哪些模型，同样属于判据的一部分。
  sha256sum "$RUN_ROOT/test_case_ids_to_generate.json" \
    "$P1_ROOT/run_p1.sh" "$P1_ROOT/bfcl_registration.py" \
    "$P1_ROOT/register_qwen_coder.py" "$P1_ROOT/sitecustomize.py" \
    "$P1_ROOT/toolcall_proxy.py" "$P1_ROOT/analyze_p1.py" "$P1_ROOT/validate_p1.py" \
    "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" \
    "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja"
} > "$RUN_ROOT/provenance.txt"

run_arm hermes P1-Qwen2.5-Coder-7B-Hermes-FC p1-qwen25-coder7b-hermes \
  --tool-call-parser hermes
run_arm repaired P1-Qwen2.5-Coder-7B-Repaired-FC p1-qwen25-coder7b-repaired \
  --tool-parser-plugin "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" \
  --tool-call-parser qwen2_5_coder \
  --chat-template "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja"

# ── 2×2 消融：上面两条是对角线（documented+hermes / dedicated+dedicated），
# 单独看无法把 0→196 在 parser 与 chat template 之间分摊。补上两个非对角格。
# 已完成的臂靠 .READY 自动跳过，所以重跑本脚本只会执行下面两条。
run_arm hermes_dedtpl P1-Qwen2.5-Coder-7B-HermesDedTpl-FC p1-qwen25-coder7b-hermesdedtpl \
  --tool-call-parser hermes \
  --chat-template "$P1_ROOT/plugin/tool_chat_template_qwen2_5_coder.jinja"
run_arm dedparser_doctpl P1-Qwen2.5-Coder-7B-DedParserDocTpl-FC p1-qwen25-coder7b-dedparserdoctpl \
  --tool-parser-plugin "$P1_ROOT/plugin/qwen2_5_coder_tool_parser.py" \
  --tool-call-parser qwen2_5_coder

"$BFCL_VENV/bin/python" "$P1_ROOT/analyze_p1.py" "$LOG_ROOT/bfcl_hermes.jsonl" \
  "$LOG_ROOT/bfcl_repaired.jsonl" \
  --result-root "$RUN_ROOT/full_hermes/result" \
  --result-root "$RUN_ROOT/full_repaired/result" \
  --score-root "$RUN_ROOT/full_hermes/score" \
  --score-root "$RUN_ROOT/full_repaired/score" | tee "$RUN_ROOT/P1_SUMMARY.txt"
touch "$P1_ROOT/P1_READY"
say "===== P1 complete ====="
