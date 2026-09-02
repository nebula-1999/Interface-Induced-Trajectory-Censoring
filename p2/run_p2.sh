#!/usr/bin/env bash
# P2：Qwen2.5-Instruct 尺寸梯子——§5.2 的阴性对照。
#
# 设计：与 Coder 梯子同题集、同协议、同温度、同判据，只换血统。
# Coder 与 Instruct 共用同一份 chat template（Coder 从 Instruct 继承，
# 这正是 §5.1 假阳性的成因），差别在 Instruct **在 tool-call token 上训练过**。
# 若 Instruct 各尺寸 parsed ≈ emitted，而 Coder 是 0 vs 一路升到 80，
# 错配就被锁死在「是否受过该格式训练」这一个变量上。
#
# 本脚本带着这个项目踩过的坑：
#   · ss/lsof/fuser 在本机不存在 → 用 /proc/net/tcp 判端口
#   · run() 必须 return "$rc"，否则 && 链断裂会静默跳过整条臂
#   · 每条臂先 preflight，配置坏了立刻停，不要烧几小时再发现
#   · 外层环境变量会被内层硬编码吃掉 → 关键路径一律显式传参
set -u

P=/root/autodl-tmp/p2
M=/root/autodl-tmp/models
E=/root/autodl-tmp/envs/p2/bin
# PATH 必须含 venv 的 bin：vLLM 会 fork 调 ninja 做 inductor 编译，
# 那次查找走 PATH，绝对路径救不了（P1 上实测踩过）
export PATH="$E:$PATH"
export HF_ENDPOINT=https://hf-mirror.com   # AutoDL 直连不了 HF，旧机同此
export HF_HUB_DISABLE_XET=1                # 不设会导致 hf-mirror 下载全部 401
export HF_HOME=/root/autodl-tmp/hf
export VLLM_ENFORCE_STRICT_TOOL_CALLING=true
# 输出目录可覆盖：补跑执行层时写到独立目录，既保住第一轮的产物，也让幂等
# 跳过不会把五条臂全跳掉（旧目录里已有合格产物）。
OUT="${P2_OUT:-$P/runs}"
PORT=8000
LOG=$P/p2_run.log
mkdir -p "$OUT" "$M"

# 单实例锁：4 个实例并发跑过一次，互相抢 8000 端口 + 抢同一个下载目录，
# 结果是 1.5B 那条臂 100/100 全 404、3B 被下到一半就拿去 serve。
# 目录存在 != 下载完成，所以并发的代价不是慢，是静默的废数据。
LOCK=$P/run_p2.pid
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] ★ 已有实例在跑（pid $(cat "$LOCK")），本次退出" | tee -a "$LOG"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

MODELS=(
  "Qwen2.5-1.5B-Instruct  Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen2.5-3B-Instruct    Qwen/Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct    Qwen/Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct   Qwen/Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct   Qwen/Qwen2.5-32B-Instruct"
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 端口是否空闲：8000 = hex 1F40，状态 0A = LISTEN
port_free() {
  local hex; hex=$(printf '%04X' "$PORT")
  ! awk -v p=":$hex" '$2 ~ p"$" && $4=="0A" {found=1} END{exit !found}' /proc/net/tcp
}
wait_port_free() {
  for _ in $(seq 1 60); do port_free && return 0; sleep 2; done
  say "  端口 $PORT 60 秒未释放"; return 1
}

serve() {  # $1=模型路径
  wait_port_free || return 1
  say "  启动 vLLM: $(basename "$1")"
  # 不加 --served-model-name：旧机 50 个臂都是这么起的，模型 id 即路径，
  # 与探针 --model 发出的字符串一致。加了会 404（实测踩过）
  nohup vllm serve "$1" \
    --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    >>"$P/vllm.log" 2>&1 &
  echo $! > "$P/vllm.pid"
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { say "  就绪（${i}0s 内）"; return 0; }
    # 用 pgrep 查真实服务进程，不用 launcher 的 $!——它 fork 完就退，会误判
    if [ $i -gt 6 ] && ! pgrep -f "[v]llm serve" >/dev/null; then
      say "  ★ vLLM 进程已退出，根因："
      grep -E "FileNotFoundError|RuntimeError|ERROR" "$P/vllm.log" | tail -5
      return 1
    fi
    sleep 5
  done
  say "  ★ 启动超时"; return 1
}

teardown() {
  [ -f "$P/vllm.pid" ] && kill "$(cat "$P/vllm.pid")" 2>/dev/null
  rm -f "$P/vllm.pid"; wait_port_free
}

run_arm() {  # $1=标签 $2=模型路径
  local rc
  say "  preflight"
  python3 "$P/preflight_toolcall.py" --port "$PORT" >>"$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "  ★ preflight 失败（rc=$rc）——配置有问题，不跑这条臂"
    return "$rc"
  fi
  say "  探针 n=100"
  python3 "$P/probe_react_full.py" \
    --model "$2" --port "$PORT" --n 100 \
    --protocol fc --strength optional --fc-schema terse \
    --parser-adapter cross_family --temperature 0.0 --seed 0 \
    --out "$OUT/traj_p2_${1}_fc.jsonl" >>"$LOG" 2>&1
  rc=$?
  say "  探针 rc=$rc"
  # 产物验收：preflight 用 /v1/models 自查名字，探针用 --model 传路径，
  # 两者解析方式不同，preflight 拦不住模型名错配。必须看产物。
  local out="$OUT/traj_p2_${1}_fc.jsonl"
  if [ ! -f "$out" ]; then
    # 没产物也是失败。旧版这里是 if [ -f ]，探针没写文件就整道防线静默跳过。
    say "  ★ 探针没产出 $out——本臂作废"
    return 1
  fi
  local nerr ntot
  # grep -c 无命中时打印 0 并 exit 1，写成 `|| echo 0` 会得到两行 "0"，
  # 后面的 [ "$nerr" -gt 5 ] 直接报 integer expression expected 并跳过整道防线。
  nerr=$(grep -c "request_error" "$out" 2>/dev/null) || nerr=0
  ntot=$(wc -l < "$out")
  say "  请求错误 $nerr / $ntot"
  if [ "$nerr" -gt 5 ]; then
    say "  ★ 请求错误过多，本臂作废并停止整轮——继续跑只会生成更多废数据"
    grep -o "__ERROR__[^\"]*" "$out" | head -2 | tee -a "$LOG"
    mv "$out" "$out.invalid"
    return 1
  fi
  return "$rc"
}

say "===== P2 开始 ====="
say "题集哈希（与旧机 P0 比对用）："
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import hashlib, json
from datasets import load_dataset
kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
clean = json.load(open("/root/autodl-tmp/p2/clean_ids.json"))["clean_index"][:100]
h = hashlib.sha256()
for i in clean:
    h.update((kc[i].get("question") or kc[i].get("prompt") or "").encode())
print(f"  clean[:100] prompt sha256 = {h.hexdigest()[:32]}")
print("  ★ 待核验：旧机跑 P0 时同样算一次，两边必须相同，否则题集已漂移")
PY

for entry in "${MODELS[@]}"; do
  tag=$(echo "$entry" | awk '{print $1}')
  repo=$(echo "$entry" | awk '{print $2}')
  path="$M/$tag"
  say "--- $tag ---"

  # 幂等：已有合格产物的臂直接跳过。补跑单条臂时（比如 3B 那次因半截模型被跳过）
  # 不加这个就会把 32B 那条跑了 1.5 小时的轨迹重新覆盖一遍。
  # P2_ONLY 可再收窄到指定 tag，逗号分隔。
  if [ -n "${P2_ONLY:-}" ] && ! echo ",$P2_ONLY," | grep -q ",$tag,"; then
    say "  不在 P2_ONLY 名单里，跳过"; continue
  fi
  existing="$OUT/traj_p2_${tag}_fc.jsonl"
  if [ -f "$existing" ]; then
    nline=$(wc -l < "$existing")
    nbad=$(grep -c "request_error" "$existing" 2>/dev/null) || nbad=0
    if [ "$nline" -ge 100 ] && [ "$nbad" -le 5 ]; then
      say "  已有合格产物（$nline 行，请求错误 $nbad），跳过；要重跑先删掉它"
      continue
    fi
    say "  已有产物但不合格（$nline 行，请求错误 $nbad），重跑"
  fi

  # 目录存在 != 下载完成：3B 被并发实例下到一半，[ ! -d ] 判它"已下载"，
  # vLLM 起来才报 Weight files referenced in index but missing，
  # 而且此后每次重跑都会跳过它。改用显式完成标记。
  if [ ! -f "$path/.p2_download_ok" ]; then
    [ -d "$path" ] && say "  $tag 目录存在但无完成标记（可能是半截下载），重下"
    say "  下载 $repo（huggingface-cli 已废弃，用 hf）"
    if hf download "$repo" --local-dir "$path" >>"$LOG" 2>&1; then
      touch "$path/.p2_download_ok"
    else
      say "  ★ 下载失败，跳过"; continue
    fi
  fi
  df -h /root/autodl-tmp | tail -1 | tee -a "$LOG"

  if serve "$path"; then
    run_arm "$tag" "$path"
    arm_rc=$?
    teardown
    if [ "$arm_rc" -ne 0 ]; then
      # 旧版这里不看返回码，run_arm 里写的"停止整轮"根本停不了任何东西。
      say "★ $tag 作废（rc=$arm_rc），停止整轮——继续跑只会生成更多废数据"
      break
    fi
  else
    say "  ★ 服务启动失败，跳过 $tag"
    teardown
  fi
done

say "===== P2 结束 ====="
say "结果："
ls -la "$OUT" | tee -a "$LOG"
