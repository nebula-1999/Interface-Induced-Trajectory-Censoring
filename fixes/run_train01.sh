#!/bin/bash
# TRAIN-01：接口修好之后，RL 能不能学会多轮？（两位审稿人一致认定的阻断项）
#
# 现状：§5.6 只证明「坏的还是坏的」——FC/hermes 下 10 步 0 次工具调用。
# 缺的是「修好就能学会」——把 §5.5 的专用适配器接进训练、跑满 150 步，
# 看多轮救回是否从 6–9/540 的横盘抬起来。
#
# 与三个历史 run 严格同配置（同数据、同超参、同评测 hook），只改两处：
#   1) vLLM 起服务时挂 hanXen 的 <tools> parser 与改写过的模板
#   2) 数据用 train_fc3.parquet（agent_name=tool_agent + FC_MANDATORY prompt）
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
rm -f TRAIN01_READY TRAIN01_INVALID train01_manifest.txt
sha256sum probe_react_full.py sandbox.py code_tool.py > train01_pinned_hashes.txt

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

# 变体：ReAct loop 去掉（走 verl 内置 tool_agent），数据换 fc3，输出目录独立
cp run_code_grpo.sh run_train01_variant.sh
sed -i "/agent_loop_config_path/d;/default_agent_loop/d" run_train01_variant.sh
sed -i 's|data.train_files="$DATA/train.parquet"|data.train_files="$DATA/train_fc3.parquet"|' run_train01_variant.sh
sed -i 's|data.val_files="$DATA/val.parquet"|data.val_files="$DATA/val_fc3.parquet"|' run_train01_variant.sh
sed -i 's|OUT=/root/autodl-tmp/runs/code-grpo-seed$SEED|OUT=/root/autodl-tmp/runs/train01-fcfixed|' run_train01_variant.sh
# 关键：让 rollout 用专用适配器起服务
sed -i 's|actor_rollout_ref.rollout.name=vllm|actor_rollout_ref.rollout.name=vllm \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_parser_plugin="'"$P"'/plugin/qwen2_5_coder_tool_parser.py" \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen2_5_coder \\\n  +actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template="'"$P"'/plugin/tool_chat_template_qwen2_5_coder.jinja"|' run_train01_variant.sh
chmod +x run_train01_variant.sh
bash -n run_train01_variant.sh || { echo "语法错"; touch TRAIN01_INVALID; exit 1; }
echo "变体检查: agent_loop行=$(grep -c agent_loop run_train01_variant.sh) 数据=$(grep -oE 'train[a-z0-9_]*\.parquet' run_train01_variant.sh | head -1) parser=$(grep -c qwen2_5_coder run_train01_variant.sh)"

echo "########## TRAIN-01  150 步  $(date +%H:%M:%S) ##########"
CODE_EVAL_OUT=/root/autodl-tmp/runs/code-eval-train01 SEED=42 ALGO=grpo \
  bash run_train01_variant.sh full > train01.log 2>&1
RC=$?
NSTEP=$(grep -cE "step:[0-9]+ -" train01.log || echo 0)
NEXEC=$(wc -l < /root/autodl-tmp/runs/.tool_call_count 2>/dev/null || echo 0)
echo "train01 rc=$RC steps=$NSTEP/150 tool_calls=$NEXEC $(date +%H:%M:%S)" >> train01_manifest.txt

python - <<'PY'
import sys, os, re; sys.path.insert(0, ".")
import metrics_lib
print("\n########## TRAIN-01 rollout 指标 ##########")
metrics_lib.show("train01.log", "FC + 专用适配器（150 步）")
def cnt(f):
    try: return sum(1 for _ in open(f))
    except FileNotFoundError: return 0
c = cnt("/root/autodl-tmp/runs/.tool_call_count")
s, n, _ = metrics_lib.parse("train01.log")
print(f"     agent 侧工具调用总数 = {c}   每步 = {c/n if n else 0:.1f}")
print(f"     .tool_called 凭据: {'存在' if os.path.exists('/root/autodl-tmp/runs/.tool_called') else '缺失'}")
print("\n  对照（已有）: FC/hermes 10 步 0 次 · ReAct 3 步 788 次（每 rollout 2.052）")
PY

[ "$RC" = "0" ] && [ "$NSTEP" = "150" ] && touch TRAIN01_READY || touch TRAIN01_INVALID
cat train01_manifest.txt
