#!/bin/bash
# P3 正式训练：broken-FC vs repaired-FC 的单变量对照。
#
# 与本目录 run_p3_probe.sh 的关系：本文件只做四件事——
#   1. 决定臂（multi_turn.format）
#   2. 决定步数与产物落点
#   3. 决定评测间隔（见下）
#   4. 把这些以**环境变量 + Hydra 覆盖**的形式传给 run_p3_probe.sh
#
# 其余全部配置（模型、数据、种子、LoRA、奖励、钩子）都沿用探针脚本里那一份，
# 一个字符都不改。两条臂的差异必须**只有** format，否则对照不成立。
#
#   usage: ARM=broken|repaired STEPS=150 [EVAL_FREQ=30] [DRY=1] bash p3/run_p3_arm.sh
set -uo pipefail

ARM="${ARM:?必须指定 ARM=broken 或 ARM=repaired}"
STEPS="${STEPS:-150}"
PROJ_DIR="${PROJ_DIR:-/root/autodl-tmp/code-agent}"

# ---- 评测间隔：P3 能不能回答主问题，全看这一个值 ----
# code_patch.py 把 CodeEvalHook 包在 RayPPOTrainer._validate 外面，而 _validate
# 只在 test_freq>0 时被调用。test_freq=-1 时钩子装好了但一次都不触发，于是拿不到
# code/turn1_pass 与 code/final_pass 的分解——也就拿不到「多轮救回数」。
#
# 而 P3 要回答的 sufficiency 问题（修好接口后多轮学习是否恢复）**只能**由救回数
# 回答：训练侧的 critic/score/mean 是聚合量，本文 §1 已证明它可以完全由首轮质量
# 提升驱动（历史 run 里 pass@1 涨 2.8 分而救回数纹丝不动，91–94% 的新通过题是
# 首轮过的）。拿聚合分判分支，等于用本文自证为混淆的量推翻本文的结论。
#
# val_before_train=True 让第 0 步就评一次：既给出训练前基线，也让「评测本身会不会
# 崩」在 15 分钟内暴露，而不是三小时后。
# **两条臂必须用同一个 EVAL_FREQ**，否则对照不成立。
EVAL_FREQ="${EVAL_FREQ:-30}"

case "$ARM" in
  broken)   FORMAT=hermes ;;
  repaired) FORMAT=qwen2_5_coder ;;
  *) echo "未知 ARM=$ARM（只能是 broken / repaired）" >&2; exit 1 ;;
esac

RUN_ROOT="/root/p3_formal/$ARM"
export P3_OUT="$RUN_ROOT/events"
export P3_CKPT_DIR="$RUN_ROOT/ckpt"

mkdir -p "$RUN_ROOT"
# 事件文件按 PID 追加，重跑会与上一次混在一起——清掉再开。
rm -rf "$P3_OUT" "$P3_CKPT_DIR"
mkdir -p "$P3_OUT" "$P3_CKPT_DIR"

echo "[p3-formal] 臂=$ARM  format=$FORMAT  步数=$STEPS  评测间隔=$EVAL_FREQ"
echo "[p3-formal] 产物目录=$RUN_ROOT"
df -h /root/autodl-tmp | tail -1

cd "$PROJ_DIR"

# 步数与评测通过 Hydra 覆盖传入；探针脚本尾部会把其余参数原样透传。
# 注意：探针脚本里的 trainer.total_training_steps=1 / test_freq=-1 出现在**前面**，
# 这里的覆盖在**后面**，Hydra 以最后一个为准。
bash p3/run_p3_probe.sh \
  actor_rollout_ref.rollout.multi_turn.format="$FORMAT" \
  trainer.total_training_steps="$STEPS" \
  trainer.test_freq="$EVAL_FREQ" \
  trainer.val_before_train=True \
  trainer.experiment_name="p3-$ARM-7b-lora"
