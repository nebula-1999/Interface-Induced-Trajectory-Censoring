#!/bin/bash
# ReAct3：与 FC3 同插桩的工具调用计数。FC3 已确认 rc=0、10/10 步、0 次调用。
set -uo pipefail
source /root/autodl-tmp/env.sh
cd /root/autodl-tmp/code-agent || exit 1
rm -f REACT3_READY REACT3_INVALID

# 先存 FC3 的计数与状态
cp -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_call_count.fc3 2>/dev/null || : > /root/autodl-tmp/runs/.tool_call_count.fc3

for pat in launch_ppo "ray::" raylet gcs_server; do
  ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null
done
for _ in $(seq 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${u:-9999}" -lt 2000 ] && break; sleep 5
done
echo "GPU 已清空: ${u}MiB"

cp run_code_grpo.sh run_probe_react3.sh
sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=3; TEST_FREQ=-1; SAVE_FREQ=-1/" run_probe_react3.sh
sed -i "s/trainer.val_before_train=True/trainer.val_before_train=False/" run_probe_react3.sh
sed -i "s|OUT=/root/autodl-tmp/runs/code-grpo-smoke|OUT=/root/autodl-tmp/runs/probe-react3|" run_probe_react3.sh
chmod +x run_probe_react3.sh; bash -n run_probe_react3.sh || exit 1
echo "变体: agent_loop行=$(grep -c agent_loop run_probe_react3.sh) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' run_probe_react3.sh | head -1)"

rm -f train_REACT3.log
SEED=7 ALGO=grpo bash run_probe_react3.sh smoke > train_REACT3.log 2>&1
RC=$?
NSTEP=$(grep -cE "step:[0-9]+ -" train_REACT3.log 2>/dev/null || echo 0)
cp -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_call_count.react3 2>/dev/null || : > /root/autodl-tmp/runs/.tool_call_count.react3
NEXEC=$(wc -l < /root/autodl-tmp/runs/.tool_call_count.react3 2>/dev/null || echo 0)
echo "react3 rc=$RC steps=$NSTEP/3 tool_calls=$NEXEC $(date +%H:%M:%S)" >> v7b_manifest.txt

python - <<'PY'
import sys, os; sys.path.insert(0, ".")
import metrics_lib
def cnt(f):
    try: return sum(1 for _ in open(f))
    except FileNotFoundError: return 0
print("\n########## 训练 rollout 工具使用对照 ##########")
for log, tag, lab in [("train_FC3.log", "fc3", "FC  (FC_MANDATORY + tool_agent + hermes)"),
                      ("train_REACT3.log", "react3", "ReAct (ReAct prompt + react_agent)")]:
    s, n, crashed = metrics_lib.show(log, lab)
    c = cnt(f"/root/autodl-tmp/runs/.tool_call_count.{tag}")
    print(f"     agent 侧工具调用次数 = {c}   每步 = {c/n if n else 0:.1f}")
print("\n  ※ 步数不同（FC 10 步 / ReAct 3 步），跨臂请用首值或每步均值")
PY
if [ "$RC" = "0" ] && [ "$NSTEP" = "3" ]; then touch REACT3_READY; else touch REACT3_INVALID; fi
cat v7b_manifest.txt
