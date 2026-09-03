#!/bin/bash
# P3：7B 在 verl 训练 rollout 路径上的探针。**不训练、不存权重、不评测。**
#
# 复现的是 §5.7 那条 FC 结论对应的条件：multi_turn.format=hermes + 默认的
# ToolAgentLoop（**不是** react_agent——最后一次训练用的是 ReAct，那是另一个条件）。
#
# 判据：emitted>0 且 accepted=0 且 executed=0 → 因果链闭合。
# 但先判钩子：三个计数都可能天然为 0，钩子没装上产生的也是一串 0。
set -uo pipefail

PROJ_DIR=/root/autodl-tmp/code-agent
DATA=/root/autodl-tmp/train/code-data
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct}"
export P3_OUT="${P3_OUT:-/root/autodl-tmp/runs/p3_rollout_probe}"
export P3_PROJ_DIR="$PROJ_DIR"
OUT="${P3_CKPT_DIR:-/root/p3_artifacts/probe_ckpt}"
P3_BATCH="${P3_BATCH:-16}"
P3_N="${P3_N:-8}"
P3_MINI_BATCH="${P3_MINI_BATCH:-16}"
P3_REWARD_WORKERS="${P3_REWARD_WORKERS:-1}"
P3_AGENT_WORKERS="${P3_AGENT_WORKERS:-2}"
P3_RAY_CPUS="${P3_RAY_CPUS:-6}"

# p3 必须在最前：它的 sitecustomize 会装探针钩子，并接力加载 PROJ_DIR 那份
export PYTHONPATH="$PROJ_DIR/p3:$PROJ_DIR/flash_attn_shim:$PROJ_DIR:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Ray 的对象溢写默认落 /tmp。数据盘只剩 ~3.9 GB，绝不能让它写那儿。
export RAY_TMPDIR=/tmp/ray_p3
mkdir -p "$RAY_TMPDIR" "$P3_OUT" "$OUT"
rm -f "$P3_OUT"/events.*.jsonl "$P3_OUT/summary.json" "$P3_OUT/P3_INVALID"
rm -f /root/autodl-tmp/runs/.tool_called

# 非交互 shell 里没有 `python`——原训练脚本靠 source env.sh 激活 venv 才有。
# setsid 起的进程走的正是那个 PATH，所以必须显式钉死解释器（P1 上同类坑踩过）。
PY="${P3_PYTHON:-/root/code-venv/bin/python}"
[ -x "$PY" ] || { echo "[p3] ★ 找不到解释器 $PY"; exit 1; }
echo "[p3] 解释器: $PY"
# 数据自检：train.parquet 是 ReAct 版（agent_name=react_agent、Thought/Action 提示词），
# 用它会得到「Agent loop react_agent not registered」——第一次就是这么失败的。
# §5.7 那条 FC 结论对应 fc3：tool_agent + FC_MANDATORY 提示词。
"$PY" - <<'PYCHK' || exit 1
import pyarrow.parquet as pq, collections, sys
t = pq.read_table("/root/autodl-tmp/train/code-data/train_fc3.parquet")
an = collections.Counter(t.column("agent_name").to_pylist())
pr = t.column("prompt").to_pylist()[0]
txt = pr[0]["content"] if isinstance(pr, list) else str(pr)
ok = set(an) == {"tool_agent"} and "必须先调用 run_tests" in txt
print(f"[p3] 数据自检: agent_name={dict(an)}  FC_MANDATORY={'必须先调用 run_tests' in txt}")
sys.exit(0 if ok else 1)
PYCHK

echo "[p3] 磁盘余量（写权重会炸盘，本轮 save_freq=-1）："
df -h /root/autodl-tmp | tail -1

"$PY" "$PROJ_DIR/launch_ppo.py" \
  algorithm.adv_estimator=grpo \
  data.seed=0 \
  actor_rollout_ref.rollout.seed=0 \
  data.train_files="$DATA/train_fc3.parquet" \
  data.val_files="$DATA/val_fc3.parquet" \
  data.train_batch_size="$P3_BATCH" \
  data.dataloader_num_workers=1 \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$P3_MINI_BATCH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.n="$P3_N" \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1200 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJ_DIR/code_tool_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.num_workers="$P3_AGENT_WORKERS" \
  custom_reward_function.path="$PROJ_DIR/reward_code.py" \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path="$PROJ_DIR/reward_code.py" \
  reward.custom_reward_function.name=compute_score \
  reward.num_workers="$P3_REWARD_WORKERS" \
  ray_kwargs.ray_init.num_cpus="$P3_RAY_CPUS" \
  trainer.use_v1=False \
  trainer.logger='[console]' \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$OUT" \
  trainer.project_name=code-agent \
  trainer.experiment_name=p3-rollout-probe-7b \
  "$@"
rc=$?
echo "[p3] launch_ppo rc=$rc"
echo "$rc" > "$P3_OUT/launch_rc"
echo "[p3] ===== 收尾判定 ====="
"$PY" "$PROJ_DIR/p3/summarise_p3.py"
