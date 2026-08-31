#!/bin/bash
# TRAIN-01B：ReAct 75 步 —— 1.5B 上唯一能把工具反馈送进训练的通道。
#
# 为什么不是 "repaired FC"（三层排查的结论，见 RESULTS §5.6 附注）：
#   1) multi_turn.format=hermes 选的是 verl 自己的 parser，挂 vLLM 插件无效
#   2) 自写 verl 版 parser 在 1.5B+mandatory 下能解析 52/100，
#      但**无一是 run_tests**——全在调用题目函数本身（与 Llama 同型的角色混淆）
#   3) 修角色混淆要 strict/受约束解码，而 verl 0.9.0 的 vLLM rollout 路径不支持
#   → 1.5B 上不存在可用的 FC 修复路径；ReAct 是唯一有工具反馈的条件
#
# 与三个历史 FC run 严格同配置（同数据题目、同超参、同评测 hook、同 seed），
# 唯一差异是 agent loop 与 system prompt（即协议本身）。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f TRAIN01B_READY TRAIN01B_INVALID train01b_manifest.txt
sha256sum probe_react_full.py sandbox.py code_tool.py react_agent_loop.py > train01b_pinned_hashes.txt

port_busy(){ awk '$2 ~ /:1F40$/ && $4=="0A" {f=1} END {exit !f}' /proc/net/tcp; }
gpu_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hard_clean(){
  for pat in api_server EngineCore launch_ppo "ray::" raylet gcs_server; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null
  done
  for _ in $(seq 60); do
    port_busy || { [ "$(gpu_mb)" -lt 2000 ] && return 0; }; sleep 3; done
  echo "!! 清理超时"; return 1; }
hard_clean || exit 1
rm -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_called

cp run_code_grpo.sh run_train01b_variant.sh
# 75 步；每 15 步评测一次（与历史 run 同频，曲线可直接并列）；不存权重省磁盘
sed -i "s/^  STEPS=150; TEST_FREQ=15; SAVE_FREQ=50.*/  STEPS=75; TEST_FREQ=15; SAVE_FREQ=-1/" run_train01b_variant.sh
sed -i 's|OUT=/root/autodl-tmp/runs/code-grpo-seed$SEED|OUT=/root/autodl-tmp/runs/train01b-react|' run_train01b_variant.sh
chmod +x run_train01b_variant.sh
bash -n run_train01b_variant.sh || { echo "语法错"; touch TRAIN01B_INVALID; exit 1; }
echo "变体: STEPS=$(grep -oE 'STEPS=[0-9]+' run_train01b_variant.sh | head -1) TEST_FREQ=$(grep -oE 'TEST_FREQ=[0-9-]+' run_train01b_variant.sh | head -1) agent_loop行=$(grep -c agent_loop run_train01b_variant.sh) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' run_train01b_variant.sh | head -1)"

echo "########## TRAIN-01B  ReAct 75 步  $(date +%H:%M:%S) ##########"
CODE_EVAL_OUT=/root/autodl-tmp/runs/code-eval-train01b SEED=42 ALGO=grpo \
  bash run_train01b_variant.sh full > train01b.log 2>&1
RC=$?
NSTEP=$(grep -cE "step:[0-9]+ -" train01b.log || echo 0)
NEXEC=$(wc -l < /root/autodl-tmp/runs/.tool_call_count 2>/dev/null || echo 0)
NEVAL=$(ls /root/autodl-tmp/runs/code-eval-train01b/step_*.jsonl 2>/dev/null | wc -l)
echo "train01b rc=$RC steps=$NSTEP/75 tool_calls=$NEXEC eval_ckpt=$NEVAL $(date +%H:%M:%S)" >> train01b_manifest.txt

python - <<'PY'
import sys, os, glob, json; sys.path.insert(0, ".")
import metrics_lib
print("\n########## TRAIN-01B rollout 指标 ##########")
s, n, crashed = metrics_lib.show("train01b.log", "ReAct 75 步")
c = sum(1 for _ in open("/root/autodl-tmp/runs/.tool_call_count")) if os.path.exists("/root/autodl-tmp/runs/.tool_call_count") else 0
print(f"     agent 侧工具调用总数 = {c}   每步 = {c/n if n else 0:.1f}")
print("\n########## 多轮救回随训练步的变化（本实验的核心问题）##########")
KD = {"HumanEval/32", "Mbpp/599"}
for f in sorted(glob.glob("/root/autodl-tmp/runs/code-eval-train01b/step_*.jsonl")):
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    m = [x for x in R if x["channel"] == "multi" and x["task_id"] not in KD]
    if not m: continue
    t1 = sum(1 for x in m if x["turns"] and x["turns"][0]["all_passed"])
    fin = sum(1 for x in m if x["turns"] and x["turns"][-1]["all_passed"])
    print(f"  {os.path.basename(f)}  n={len(m)}  turn1={t1}  final={fin}  救回={fin-t1}")
print("\n  对照（历史 FC run，150 步）: 救回全程 6–9/540 横盘，末-首 +1/-1/0")
PY

[ "$RC" = "0" ] && [ "$NSTEP" = "75" ] && touch TRAIN01B_READY || touch TRAIN01B_INVALID
cat train01b_manifest.txt
