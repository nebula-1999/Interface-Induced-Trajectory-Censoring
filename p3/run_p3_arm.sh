#!/bin/bash
# P3 正式训练：broken-FC vs repaired-FC 的单变量对照。
#
# 与本目录 run_p3_probe.sh 的关系：本文件只做三件事——
#   1. 决定臂（multi_turn.format）
#   2. 决定步数与产物落点
#   3. 把这两条以**环境变量 + Hydra 覆盖**的形式传给 run_p3_probe.sh
#
# 其余全部配置（模型、数据、种子、LoRA、奖励、钩子）都沿用探针脚本里那一份，
# 一个字符都不改。两条臂的差异必须**只有** format，否则对照不成立。
#
#   usage: ARM=broken|repaired STEPS=150 [DRY=1] bash p3/run_p3_arm.sh
set -uo pipefail

ARM="${ARM:?必须指定 ARM=broken 或 ARM=repaired}"
STEPS="${STEPS:-150}"
PROJ_DIR="${PROJ_DIR:-/root/autodl-tmp/code-agent}"

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

echo "[p3-formal] 臂=$ARM  format=$FORMAT  步数=$STEPS"
echo "[p3-formal] 产物目录=$RUN_ROOT"
df -h /root/autodl-tmp | tail -1

cd "$PROJ_DIR"

# 步数通过 Hydra 覆盖传入；探针脚本尾部会把其余参数原样透传。
# 注意：探针脚本里的 trainer.total_training_steps=1 出现在**前面**，
# 这里的覆盖在**后面**，Hydra 以最后一个为准。
bash p3/run_p3_probe.sh \
  actor_rollout_ref.rollout.multi_turn.format="$FORMAT" \
  trainer.total_training_steps="$STEPS" \
  trainer.experiment_name="p3-$ARM-7b-lora"
