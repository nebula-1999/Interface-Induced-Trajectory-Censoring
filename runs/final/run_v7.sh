#!/bin/bash
# 因果链的直接演示：训练 rollout 里，FC/hermes 收不到 Observation 而 ReAct 收得到。
#
# 关键认识：tool_calls/mean 是**逐步**记录的 rollout 指标，第 1 步就能看到，
# 不需要训练收敛。所以 10 步足够证明机制。
# （要证明"训练结果因此不同"是另一回事，那需要 150 步 × 2，约 10 小时，另行决策。）
#
# 两臂唯一差异：是否启用 ReAct AgentLoop。
#   armA  去掉 agent_loop 两行 → 回落到 verl 默认工具 agent loop + multi_turn.format=hermes
#   armB  保留 ReAct AgentLoop
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f V7_READY
STEPS_N="${TRAIN_STEPS:-10}"

mk(){  # mk <输出脚本> <是否用 ReAct>
  local OUT=$1 USE_REACT=$2
  cp run_code_grpo.sh "$OUT"
  # 步数与频率：不评测、不存权重，只要 rollout 指标
  sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=$STEPS_N; TEST_FREQ=-1; SAVE_FREQ=-1/" "$OUT"
  sed -i "s/trainer.val_before_train=True/trainer.val_before_train=False/" "$OUT"
  if [ "$USE_REACT" = "no" ]; then
    sed -i "/agent_loop_config_path/d;/default_agent_loop/d" "$OUT"
  fi
  chmod +x "$OUT"
  bash -n "$OUT" || { echo "[$OUT] 语法错"; return 1; }
  echo "  生成 $OUT  (ReAct=$USE_REACT)"
  echo "    STEPS 行: $(grep -m1 'STEPS=' $OUT | sed 's/^ *//')"
  echo "    agent_loop 行数: $(grep -c 'agent_loop' $OUT)"
}

echo "########## 生成两个变体 ##########"
mk run_probe_fc.sh no  || exit 1
mk run_probe_react.sh yes || exit 1

runarm(){  # runarm <脚本> <标签> <输出目录>
  local SH=$1 TAG=$2 OUT=$3
  echo "########## $TAG  $(date +%H:%M:%S) ##########"
  rm -rf "$OUT"
  OUTDIR="$OUT" SEED=7 ALGO=grpo bash "$SH" smoke > "train_$TAG.log" 2>&1
  echo "$TAG rc=$? $(date +%H:%M:%S)" >> v7_progress.txt
  python - <<PY
import re
t = open("train_$TAG.log", encoding="utf-8", errors="replace").read()
for k in ["tool_calls/mean", "num_turns/mean", "global_seqlen/mean", "timing_s/step"]:
    v = [float(x) for x in re.findall(rf"{re.escape(k)}:([0-9.eE+-]+)", t)]
    print(f"  $TAG  {k:<20} n={len(v):>3}" + (f"  首={v[0]:.3f} 末={v[-1]:.3f} 均={sum(v)/len(v):.3f}" if v else "  (无数值)"))
PY
}

runarm run_probe_fc.sh    FC    /root/autodl-tmp/runs/probe-train-fc
runarm run_probe_react.sh REACT /root/autodl-tmp/runs/probe-train-react

echo "########## 对照结论 ##########"
python - <<'PY'
import re
def m(f, k):
    try: t = open(f, encoding="utf-8", errors="replace").read()
    except FileNotFoundError: return None
    v = [float(x) for x in re.findall(rf"{re.escape(k)}:([0-9.eE+-]+)", t)]
    return (sum(v)/len(v)) if v else None
for k in ["tool_calls/mean", "num_turns/mean"]:
    a, b = m("train_FC.log", k), m("train_REACT.log", k)
    fa = f"{a:.3f}" if a is not None else "未记录"
    fb = f"{b:.3f}" if b is not None else "未记录"
    print(f"  {k:<18}  FC={fa:<10}  ReAct={fb}")
PY
touch V7_READY
echo "V7_READY $(date +%H:%M:%S)" >> runall.done
