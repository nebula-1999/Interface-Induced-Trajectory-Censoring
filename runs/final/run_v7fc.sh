#!/bin/bash
# FC 协议对照（第二版）。两处修正：
#  1) 数据换成 train_fc3.parquet —— 不仅 agent_name=tool_agent，system prompt
#     也换成 FC 版（原 train_fc.parquet 100% 仍带 ReAct 模板，那样跑出来的是
#     「ReAct prompt 配 hermes loop」的不兼容负对照，不是 FC 协议对照）。
#  2) READY 不再无条件写：要求 rc=0、跑满 10 步、无 Traceback。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
# V7_READY 只当作「ReAct 臂已结束」的时序信号；它由 run_v7.sh 无条件写出，
# 不代表该臂有效 —— ReAct 的有效性另行用 metrics_lib 独立核验。
for _ in $(seq 200); do [ -f V7_READY ] && break; sleep 30; done
rm -f V7FC_READY V7FC_INVALID
STEPS_N=10

cp run_code_grpo.sh run_probe_fc.sh
sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=$STEPS_N; TEST_FREQ=-1; SAVE_FREQ=-1/" run_probe_fc.sh
sed -i "s/trainer.val_before_train=True/trainer.val_before_train=False/" run_probe_fc.sh
sed -i "/agent_loop_config_path/d;/default_agent_loop/d" run_probe_fc.sh
sed -i 's|data.train_files="$DATA/train.parquet"|data.train_files="$DATA/train_fc3.parquet"|' run_probe_fc.sh
sed -i 's|data.val_files="$DATA/val.parquet"|data.val_files="$DATA/val_fc3.parquet"|' run_probe_fc.sh
sed -i 's|OUT=/root/autodl-tmp/runs/code-grpo-smoke|OUT=/root/autodl-tmp/runs/probe-train-fc|' run_probe_fc.sh
chmod +x run_probe_fc.sh; bash -n run_probe_fc.sh || exit 1
echo "变体: agent_loop 行=$(grep -c agent_loop run_probe_fc.sh) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' run_probe_fc.sh | head -1) OUT=$(grep -oE 'OUT=/root[^ ]*' run_probe_fc.sh | head -1)"

echo "########## FC 协议对照  $(date +%H:%M:%S) ##########"
SEED=7 ALGO=grpo bash run_probe_fc.sh smoke > train_FC.log 2>&1
RC=$?
cp -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_call_count.fc 2>/dev/null || : > /root/autodl-tmp/runs/.tool_call_count.fc
if [ -f /root/autodl-tmp/runs/.tool_called ]; then echo exists > /root/autodl-tmp/runs/.tool_called.fc
else echo missing > /root/autodl-tmp/runs/.tool_called.fc; fi
NSTEP=$(grep -cE "step:[0-9]+ -" train_FC.log 2>/dev/null || echo 0)
NEXEC=$(wc -l < /root/autodl-tmp/runs/.tool_call_count.fc 2>/dev/null || echo 0)
echo "FC rc=$RC steps=$NSTEP exec=$NEXEC flag=$(cat /root/autodl-tmp/runs/.tool_called.fc) $(date +%H:%M:%S)" >> v7_progress.txt

python metrics_lib_show.py train_FC.log "FC/hermes" 2>/dev/null || python - <<'PY'
import sys; sys.path.insert(0, "."); import metrics_lib
metrics_lib.show("train_FC.log", "FC/hermes")
PY

if [ "$RC" = "0" ] && [ "$NSTEP" = "$STEPS_N" ] && ! grep -qE "Traceback|AssertionError|CUDA out of memory" train_FC.log; then
  touch V7FC_READY; echo "V7FC_READY steps=$NSTEP exec=$NEXEC $(date +%H:%M:%S)" >> runall.done
else
  touch V7FC_INVALID; echo "V7FC_INVALID rc=$RC steps=$NSTEP/$STEPS_N $(date +%H:%M:%S)" >> runall.done
fi
