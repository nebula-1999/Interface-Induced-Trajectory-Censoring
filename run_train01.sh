#!/bin/bash
# TRAIN-01：修好接口后的 150 步 GRPO —— 闭合 §1 的谜题。
#
# 目前 §5.6 只证明「坏的还是坏的」（FC 10 步 0 次调用）。
# 本臂把 §5.5 的专用适配器接进训练，跑满 150 步，看多轮救回
# 是否从 6–9/540 的横盘抬起来。这是审稿人预写好的批评，也是
# 唯一能把「机制演示」升级成「因果闭环」的实验。
#
# 关键设计：
#  · 与已有三个 150 步 run 同模型、同数据题目、同超参、同评测 hook
#  · 数据用 train_fc3.parquet（FC_MANDATORY prompt + agent_name=tool_agent）
#  · 服务端换成 hanXen 专用适配器（parser + 模板 + few-shot 已改 run_tests）
#  · 每步记录 agent 侧工具调用计数，与 §5.6 同一插桩
#  · 先 5 步冒烟验证「工具调用 > 0」，不通过绝不进正式 150 步
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f TRAIN01_READY TRAIN01_INVALID train01_manifest.txt
sha256sum probe_react_full.py code_tool.py react_agent_loop.py sandbox.py \
  | cut -c1-16 | tr '\n' ' ' | xargs -I{} echo "hashes={}" >> train01_manifest.txt

port_busy(){ awk '$2 ~ /:1F40$/ && $4=="0A" {f=1} END {exit !f}' /proc/net/tcp; }
gpu_mb(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hard_clean(){
  for pat in api_server EngineCore VLLM::Engine launch_ppo "ray::" raylet gcs_server; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill 2>/dev/null; done
  sleep 4
  for pat in api_server EngineCore VLLM::Engine launch_ppo "ray::" raylet gcs_server; do
    ps -eo pid=,args= | awk -v p="$pat" 'index($0,p)>0 && !/awk/ {print $1}' | xargs -r kill -9 2>/dev/null; done
  for _ in $(seq 60); do
    port_busy || { [ "$(gpu_mb)" -lt 2000 ] && return 0; }; sleep 3; done
  echo "!! GPU/Ray 未清空"; return 1; }

mk(){  # 生成训练变体：步数 / 输出目录 / 专用适配器
  local STEPS_N=$1 OUT=$2 SH=$3
  cp run_code_grpo.sh "$SH"
  sed -i "s/^  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1.*/  STEPS=$STEPS_N; TEST_FREQ=15; SAVE_FREQ=-1/" "$SH"
  sed -i "s|OUT=/root/autodl-tmp/runs/code-grpo-smoke|OUT=$OUT|" "$SH"
  sed -i "/agent_loop_config_path/d;/default_agent_loop/d" "$SH"          # 用 verl 内置 tool_agent
  sed -i 's|data.train_files="$DATA/train.parquet"|data.train_files="$DATA/train_fc3.parquet"|' "$SH"
  sed -i 's|data.val_files="$DATA/val.parquet"|data.val_files="$DATA/val_fc3.parquet"|' "$SH"
  # rollout 走专用适配器：verl 通过 vllm 的 server 参数透传
  sed -i 's|actor_rollout_ref.rollout.name=vllm \\|actor_rollout_ref.rollout.name=vllm \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_parser_plugin="'"$P"'/plugin/qwen2_5_coder_tool_parser.py" \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen2_5_coder \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template="'"$P"'/plugin/tool_chat_template_qwen2_5_coder.jinja" \\|' "$SH"
  chmod +x "$SH"; bash -n "$SH" || return 1
  echo "  变体 $SH: STEPS=$(grep -m1 'STEPS=' $SH | tr -d ' ') 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' $SH | head -1) agent_loop行=$(grep -c agent_loop $SH)"
  return 0; }

count_calls(){ wc -l < /root/autodl-tmp/runs/.tool_call_count 2>/dev/null || echo 0; }

############ 阶段 1：5 步冒烟，必须见到工具调用 > 0 ############
hard_clean || exit 1
mk 5 /root/autodl-tmp/runs/train01-smoke run_train01_smoke.sh || { echo "变体生成失败"; touch TRAIN01_INVALID; exit 1; }
rm -f /root/autodl-tmp/runs/.tool_call_count
echo "########## 冒烟 5 步  $(date +%H:%M:%S) ##########"
SEED=7 ALGO=grpo bash run_train01_smoke.sh smoke > train01_smoke.log 2>&1
SRC=$?; SN=$(grep -cE "step:[0-9]+ -" train01_smoke.log || echo 0); SC=$(count_calls)
echo "smoke rc=$SRC steps=$SN/5 tool_calls=$SC" >> train01_manifest.txt
echo "  冒烟结果: rc=$SRC 步数=$SN 工具调用=$SC"
hard_clean || true
if [ "$SRC" != "0" ] || [ "$SN" -lt 3 ] || [ "$SC" -lt 1 ]; then
  echo "✗ 冒烟未通过（需 rc=0、步数≥3、工具调用≥1）→ 不进正式 150 步"
  grep -iE "Traceback|Error|not a valid|unrecognized" train01_smoke.log | head -5
  touch TRAIN01_INVALID; exit 1
fi
echo "✓ 冒烟通过：适配器在训练 rollout 中确实产生了工具调用"

############ 阶段 2：正式 150 步 ############
mk 150 /root/autodl-tmp/runs/train01-fcfixed run_train01_full.sh || { touch TRAIN01_INVALID; exit 1; }
rm -f /root/autodl-tmp/runs/.tool_call_count
export CODE_EVAL_OUT=/root/autodl-tmp/runs/code-eval-fcfixed-seed7
echo "########## 正式 150 步  $(date +%H:%M:%S) ##########"
SEED=7 ALGO=grpo bash run_train01_full.sh full > train01_full.log 2>&1
RC=$?; N=$(grep -cE "step:[0-9]+ -" train01_full.log || echo 0); CALLS=$(count_calls)
cp -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.tool_call_count.train01 2>/dev/null || true
echo "full rc=$RC steps=$N/150 tool_calls=$CALLS $(date +%H:%M:%S)" >> train01_manifest.txt
hard_clean || true

python - <<'PY'
import sys, os, re, glob, json
sys.path.insert(0, ".")
import metrics_lib
print("\n########## TRAIN-01 结果 ##########")
metrics_lib.show("train01_full.log", "FC + 专用适配器 (150 步)")
try: print("  agent 侧工具调用总数:", sum(1 for _ in open("/root/autodl-tmp/runs/.tool_call_count.train01")))
except FileNotFoundError: print("  计数文件缺失")
print("\n########## 多轮救回是否抬起来（对照三个已有 run）##########")
KNOWN = {"HumanEval/32", "Mbpp/599"}
def curve(d):
    out=[]
    for f in sorted(glob.glob(f"{d}/step_*.jsonl")):
        R=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
        m=[r for r in R if r["channel"]=="multi" and r["task_id"] not in KNOWN]
        if not m: continue
        t1=sum(1 for r in m if r["turns"] and r["turns"][0].get("all_passed"))
        fin=sum(1 for r in m if any(t.get("all_passed") for t in r["turns"]))
        out.append((os.path.basename(f)[5:10], t1, fin, fin-t1, len(m)))
    return out
for d,lab in [("/root/autodl-tmp/runs/code-eval-fcfixed-seed7","FC-fixed(本臂)"),
              ("/root/autodl-tmp/runs/code-eval-grpo-seed1","GRPO seed1(已有)"),
              ("/root/autodl-tmp/runs/code-eval-rloo-seed42","RLOO(已有)")]:
    c=curve(d)
    if not c: print(f"  {lab:<20} 无评测输出"); continue
    print(f"  {lab:<20} 救回曲线: {[x[3] for x in c]}   首轮 {c[0][1]}→{c[-1][1]}   最终 {c[0][2]}→{c[-1][2]}")
print("\n  判读：若本臂救回曲线明显高于/上升于已有三个 run，则 §1 谜题闭合；")
print("        若仍横盘，说明修好接口不足以让 1.5B 学会多轮 —— 同样是有价值的负结果。")
PY
if [ "$RC" = "0" ] && [ "$N" = "150" ] && [ "$CALLS" -gt 0 ]; then
  touch TRAIN01_READY; echo "TRAIN01_READY steps=$N calls=$CALLS $(date +%H:%M:%S)" >> runall.done
else
  touch TRAIN01_INVALID; echo "TRAIN01_INVALID rc=$RC steps=$N/150 calls=$CALLS" >> runall.done
fi
echo "########## manifest ##########"; cat train01_manifest.txt
