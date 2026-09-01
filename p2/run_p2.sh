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
OUT=$P/runs
PORT=8000
LOG=$P/p2_run.log
mkdir -p "$OUT" "$M"

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
  nohup vllm serve "$1" \
    --port "$PORT" --served-model-name probe \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    >>"$P/vllm.log" 2>&1 &
  echo $! > "$P/vllm.pid"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { say "  就绪"; return 0; }
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
  return "$rc"          # ← 必须显式返回，否则 && 链断裂会静默跳过后续臂
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

  if [ ! -d "$path" ]; then
    say "  下载 $repo"
    huggingface-cli download "$repo" --local-dir "$path" >>"$LOG" 2>&1 || {
      say "  ★ 下载失败，跳过"; continue; }
  fi
  df -h /root/autodl-tmp | tail -1 | tee -a "$LOG"

  if serve "$path"; then
    run_arm "$tag" "$path"
    teardown
  else
    say "  ★ 服务启动失败，跳过 $tag"
    teardown
  fi
done

say "===== P2 结束 ====="
say "结果："
ls -la "$OUT" | tee -a "$LOG"
