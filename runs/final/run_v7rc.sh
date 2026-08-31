#!/bin/bash
# ReAct 计数补测（3 步）：主 ReAct 臂的 Ray worker 在 sandbox 插桩前已启动，
# 故无计数，只有 .tool_called 二值凭据。本臂用同一插桩取得与 FC 臂可比的执行次数。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
for _ in $(seq 200); do { [ -f V7FC_READY ] || [ -f V7FC_INVALID ]; } && break; sleep 30; done
rm -f V7RC_READY V7RC_INVALID
STEPS_N=3

cp run_code_grpo.sh run_probe_react3.sh
sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=$STEPS_N; TEST_FREQ=-1; SAVE_FREQ=-1/" run_probe_react3.sh
sed -i "s/trainer.val_before_train=True/trainer.val_before_train=False/" run_probe_react3.sh
sed -i 's|OUT=/root/autodl-tmp/runs/code-grpo-smoke|OUT=/root/autodl-tmp/runs/probe-train-react3|' run_probe_react3.sh
chmod +x run_probe_react3.sh; bash -n run_probe_react3.sh || exit 1
echo "变体: agent_loop 行=$(grep -c agent_loop run_probe_react3.sh) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' run_probe_react3.sh | head -1)"

echo "########## ReAct 计数补测  $(date +%H:%M:%S) ##########"
SEED=7 ALGO=grpo bash run_probe_react3.sh smoke > train_REACT3.log 2>&1
RC=$?
cp -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_call_count.react3 2>/dev/null || : > /root/autodl-tmp/runs/.tool_call_count.react3
NSTEP=$(grep -cE "step:[0-9]+ -" train_REACT3.log 2>/dev/null || echo 0)
NEXEC=$(wc -l < /root/autodl-tmp/runs/.tool_call_count.react3 2>/dev/null || echo 0)
echo "REACT3 rc=$RC steps=$NSTEP exec=$NEXEC $(date +%H:%M:%S)" >> v7_progress.txt

python - <<'PY'
import sys, os
sys.path.insert(0, ".")
import metrics_lib
print("\n########## 训练 rollout 对照（同一 sandbox 插桩）##########")
metrics_lib.show("train_FC.log", "FC/hermes  (FC prompt + tool_agent)")
metrics_lib.show("train_REACT3.log", "ReAct      (ReAct prompt + react_agent)")
def cnt(f):
    try: return sum(1 for _ in open(f))
    except FileNotFoundError: return 0
fc = cnt("/root/autodl-tmp/runs/.tool_call_count.fc")
rc = cnt("/root/autodl-tmp/runs/.tool_call_count.react3")
fs, fn, _ = metrics_lib.parse("train_FC.log")
rs, rn, _ = metrics_lib.parse("train_REACT3.log")
print(f"\n  sandbox 工具执行次数：")
print(f"    FC/hermes  {fc:>6} 次 / {fn} 步 = {fc/fn if fn else 0:.1f} 次每步")
print(f"    ReAct      {rc:>6} 次 / {rn} 步 = {rc/rn if rn else 0:.1f} 次每步")
try: print(f"  FC 的 .tool_called 凭据: {open('/root/autodl-tmp/runs/.tool_called.fc').read().strip()}")
except FileNotFoundError: pass
print(f"  ReAct 的 .tool_called 凭据: {'exists' if os.path.exists('/root/autodl-tmp/runs/.tool_called') else 'missing'}")
PY

if [ "$RC" = "0" ] && [ "$NSTEP" = "$STEPS_N" ] && ! grep -qE "Traceback|AssertionError|CUDA out of memory" train_REACT3.log; then
  touch V7RC_READY; echo "V7RC_READY steps=$NSTEP exec=$NEXEC $(date +%H:%M:%S)" >> runall.done
else
  touch V7RC_INVALID; echo "V7RC_INVALID rc=$RC steps=$NSTEP/$STEPS_N $(date +%H:%M:%S)" >> runall.done
fi
