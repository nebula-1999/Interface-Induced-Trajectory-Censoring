#!/bin/bash
# GRPO + multi-turn code agent。参数基线来自主论文那条线 2026-08-27 09:13
# 实际跑通的 overrides（Qwen3-4B / 2×A800），按 1.5B / 单卡调整。
#
# 用法：  ./run_code_grpo.sh [smoke|full]
#   smoke  只跑 20 步，用来验证栈通不通（开卡后第一件事）
#   full   150 步正式 run
set -euo pipefail

MODE="${1:-smoke}"
# seed 必须**同时**作用于数据采样与 rollout 采样。只换训练 seed 会让
# 误差棒只含数据顺序的方差、不含生成随机性——那正是 repro-variance 那条线
# 批评各家论文的地方，自己不能犯。已跑完的第一个 run 用的是 verl 默认 42。
SEED="${SEED:-42}"

# 中途叫停后续 seed 用的闸门。batch_seeds.sh 的 for 循环已在内存里改不动，
# 但每个 seed 都是新 bash 重新读本文件，所以在这里拦得住——
# 且完全不影响正在跑的那个 seed。
source /root/autodl-tmp/env.sh

PROJ_DIR=/root/autodl-tmp/code-agent
DATA=/root/autodl-tmp/train/code-data
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-Coder-1.5B-Instruct}"

# flash_attn_shim 必须在最前：verl 的 log-prob 计算硬依赖 flash_attn.bert_padding，
# 而 flash-attn 最高只有 torch 2.9 的预编译轮子，与 torch 2.13 ABI 不符。
# PROJ_DIR 本身供 verl import code_tool.CodeTool / sandbox / reward_code。
export PYTHONPATH="$PROJ_DIR/flash_attn_shim:$PROJ_DIR:${PYTHONPATH:-}"
# 每个 run 独立的评测输出目录，否则多 seed/多算法的 per-task 记录会 append 到一起
export CODE_EVAL_OUT="/root/autodl-tmp/runs/code-eval-${ALGO:-grpo}-seed${SEED}"
# ReAct AgentLoop 要能 import sandbox，且要能被 verl 的注册机制发现
export PYTHONPATH="$PROJ_DIR:${PYTHONPATH:-}"
# BLAS 线程必须钉死：这台 112 核，沙箱子进程一 import numpy 就会因为
# 线程池缓冲区超掉 RLIMIT_AS 而报 OpenBLAS 内存分配失败
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 清掉上一轮的工具执行凭据，否则 guard 会被旧文件骗过
rm -f /root/autodl-tmp/runs/.tool_called
rm -f /root/autodl-tmp/runs/.tool_call_count /root/autodl-tmp/runs/.sandbox_exec_count

if [ "$MODE" = "smoke" ]; then
  STEPS=20; TEST_FREQ=10; SAVE_FREQ=-1        # 冒烟不存权重
  export CODE_EVAL_LIMIT=24                   # 冒烟只评 24 题，跑满 542 题要 1.5h
  OUT=/root/autodl-tmp/runs/code-grpo-smoke
else
  STEPS=150; TEST_FREQ=15; SAVE_FREQ=50       # 150 步存 3 份，够断点续跑
  OUT=/root/autodl-tmp/runs/code-grpo-seed$SEED
fi
# smoke 与 full **必须分目录**：共用目录配上 resume_mode=auto，
# 冒烟会静默续跑正式 run 的 checkpoint，而日志上看不出来。

# 走 launch_ppo.py 而不是 python -m verl.trainer.main_ppo：
# 它会先 import code_patch 把分解评测挂到 RayPPOTrainer._validate 上，
# 权重一次都不落盘。直接 -m 的话钩子挂不上，曲线拿不到。
python "$PROJ_DIR/launch_ppo.py" \
  algorithm.adv_estimator=${ALGO:-grpo} \
  data.seed=$SEED \
  actor_rollout_ref.rollout.seed=$SEED \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/val.parquet" \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=6144 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1200 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJ_DIR/code_tool_config.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJ_DIR/react_agent.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
  custom_reward_function.path="$PROJ_DIR/reward_code.py" \
  custom_reward_function.name=compute_score \
  reward.custom_reward_function.path="$PROJ_DIR/reward_code.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.logger='[console]' \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=$STEPS \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.max_actor_ckpt_to_keep=1 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]' \
  trainer.resume_mode=auto \
  trainer.default_local_dir="$OUT" \
  trainer.project_name=code-agent \
  trainer.experiment_name="${ALGO:-grpo}-1.5b-$MODE-seed$SEED" \
  "${@:2}"
