#!/bin/bash
# 明天用：串行跑 FC3 与 ReAct3，一臂一日志，臂间强制清空 GPU 与 Ray。
# 修的是昨晚的调度事故：kill 只匹配了 wrapper，没杀 launch_ppo / Ray 进程树，
# 旧 FC2 继续占显存 → FC3 OOM → 而 ReAct3 把 V7FC_INVALID 当成放行信号照跑，
# 两套 Ray 集群同时在 GPU 上，日志还被两次运行混写。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f V7B_READY V7B_INVALID v7b_manifest.txt

hard_clean(){   # 杀全树并等显存真正归零
  for pat in run_probe_fc run_probe_react3 launch_ppo "ray::" raylet gcs_server dashboard_agent; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill 2>/dev/null
  done
  sleep 5
  for pat in run_probe_fc run_probe_react3 launch_ppo "ray::" raylet gcs_server dashboard_agent; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null
  done
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    n=$(ps -eo args= | grep -cE "[l]aunch_ppo|[r]ay::" || true)
    [ "${used:-9999}" -lt 2000 ] && [ "$n" = "0" ] && return 0
    sleep 5
  done
  echo "!! GPU/Ray 未清空（used=${used}MiB procs=$n）"; return 1; }

arm(){  # arm <标签> <数据前缀|react> <步数> <日志名>
  local TAG=$1 DATA=$2 STEPS_N=$3 LOG=$4
  hard_clean || { echo "$TAG SKIP_DIRTY_GPU" >> v7b_manifest.txt; return 1; }
  local SH=run_probe_$TAG.sh
  cp run_code_grpo.sh "$SH"
  sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=$STEPS_N; TEST_FREQ=-1; SAVE_FREQ=-1/" "$SH"
  sed -i "s/trainer.val_before_train=True/trainer.val_before_train=False/" "$SH"
  sed -i "s|OUT=/root/autodl-tmp/runs/code-grpo-smoke|OUT=/root/autodl-tmp/runs/probe-$TAG|" "$SH"
  if [ "$DATA" != "react" ]; then
    sed -i "/agent_loop_config_path/d;/default_agent_loop/d" "$SH"
    sed -i "s|data.train_files=\"\$DATA/train.parquet\"|data.train_files=\"\$DATA/${DATA}.parquet\"|" "$SH"
    sed -i "s|data.val_files=\"\$DATA/val.parquet\"|data.val_files=\"\$DATA/${DATA/train/val}.parquet\"|" "$SH"
  fi
  chmod +x "$SH"; bash -n "$SH" || { echo "$TAG SYNTAX_FAIL" >> v7b_manifest.txt; return 1; }
  echo "[$TAG] agent_loop行=$(grep -c agent_loop $SH) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' $SH | head -1)"
  rm -f "$LOG"
  echo "########## $TAG  $(date +%H:%M:%S) ##########"
  SEED=7 ALGO=grpo bash "$SH" smoke > "$LOG" 2>&1
  local rc=$? n=$(grep -cE "step:[0-9]+ -" "$LOG" || echo 0)
  local ex=$(wc -l < /root/autodl-tmp/runs/.tool_call_count 2>/dev/null || echo 0)
  cp -f /root/autodl-tmp/runs/.tool_call_count "/root/autodl-tmp/runs/.tool_call_count.$TAG" 2>/dev/null || : > "/root/autodl-tmp/runs/.tool_call_count.$TAG"
  echo "$TAG rc=$rc steps=$n/$STEPS_N tool_calls=$ex $(date +%H:%M:%S)" >> v7b_manifest.txt
  hard_clean || true
  [ "$rc" = "0" ] && [ "$n" = "$STEPS_N" ]; }

FCOK=0; RCOK=0
arm fc3    train_fc3 10 train_FC3.log     && FCOK=1
# 只有 FC3 有效才继续 —— 昨晚 ReAct3 在 FC3 失败后照跑，是错的
if [ "$FCOK" = "1" ]; then
  arm react3 react    3  train_REACT3.log && RCOK=1
else
  echo "react3 SKIPPED_FC3_INVALID" >> v7b_manifest.txt
fi

python - <<'PY'
import sys, os; sys.path.insert(0, ".")
import metrics_lib
def cnt(f):
    try: return sum(1 for _ in open(f))
    except FileNotFoundError: return 0
print("\n########## 训练 rollout 对照 ##########")
for log, tag, lab in [("train_FC3.log","fc3","FC/hermes (FC_MANDATORY prompt + tool_agent)"),
                      ("train_REACT3.log","react3","ReAct (ReAct prompt + react_agent)")]:
    s, n, crashed = metrics_lib.show(log, lab)
    c = cnt(f"/root/autodl-tmp/runs/.tool_call_count.{tag}")
    print(f"     工具调用次数(agent 侧专用计数) = {c}   每步 = {c/n if n else 0:.1f}")
print("\n  ※ 步数不同，跨臂比较请用首值（第 1 步、预更新 rollout）或每步均值")
PY
if [ "$FCOK" = "1" ] && [ "$RCOK" = "1" ]; then touch V7B_READY; else touch V7B_INVALID; fi
echo "########## manifest ##########"; cat v7b_manifest.txt
