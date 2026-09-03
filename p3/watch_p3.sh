#!/bin/bash
# P3 两臂自动接力 + 结果回传。
#
# **运行位置：本机（Mac）。** 所有服务器操作走 ssh，所有 git 操作在本机仓库。
# 不要搬到服务器上跑——那里没有 ssh key 也没有 git 仓库。
#
# 设计前提：本脚本会被定时任务在**全新会话**里执行，看不到任何对话上下文。
# 因此所有路径、臂名、判据全部写死在这里，不依赖环境变量或调用方的记忆。
#
# 每轮做四件事，每件都可独立失败而不影响其余：
#   1. 探测两臂当前状态（未启动 / 运行中 / 收尾中 / 已结束）
#   2. 若 broken 已结束且 repaired 未启动 → 启动 repaired
#   3. 若任一条臂刚结束 → 把 summary 拉回本机仓库并提交推送
#   4. 把本轮状态追加到 p3/WATCH_LOG.md（给人看，也给下一轮自己看）
#
# 安全约束：
#   - 同一时刻只允许一条臂在跑（单卡，显存不够两条并行）
#   - 已启动过就不再重复启动（靠 .launched marker 判定，不靠进程名）
#   - 推送失败不删任何产物，下轮重试
set -uo pipefail

SSH_HOST="autodl-code"
PROJ_DIR="/root/autodl-tmp/code-agent"
FORMAL_DIR="/root/p3_formal"
LOCAL_REPO="/Users/wangwenbo/RL Project/code_agent"
STEPS=150
LOG="$LOCAL_REPO/p3/WATCH_LOG.md"

mkdir -p "$LOCAL_REPO/p3/results"

# 一次 ssh 拿回两条臂的关键状态，避免反复握手
remote_probe() {
  ssh -o BatchMode=yes -o ConnectTimeout=12 "$SSH_HOST" '
    for arm in broken repaired; do
      d=/root/p3_formal/$arm
      printf "%s|" "$arm"
      [ -f /root/p3_formal/$arm.launched ] && printf "launched" || printf "not_launched"
      printf "|"
      [ -f $d/events/summary.json ] && printf "has_summary" || printf "no_summary"
      printf "|"
      p=$(grep -oE "Training Progress: *[0-9]+%[^,]*" /root/p3_formal/${arm}_driver.log 2>/dev/null | tail -1)
      printf "%s" "${p:-none}"
      printf "\n"
    done
  ' 2>/dev/null
}

arm_field() {   # arm_field <探测输出> <臂名> <字段号 2=launched 3=summary>
  echo "$1" | grep "^$2|" | cut -d'|' -f"$3"
}
arm_prog() {
  echo "$1" | grep "^$2|" | cut -d'|' -f4
}

launch_arm() {
  local arm="$1"
  echo "[watch] 启动 $arm 臂（$STEPS 步）"
  ssh -o BatchMode=yes -o ConnectTimeout=12 "$SSH_HOST" \
    "mkdir -p $FORMAL_DIR && cd $PROJ_DIR && touch $FORMAL_DIR/${arm}.launched && \
     setsid nohup env ARM=$arm STEPS=$STEPS bash p3/run_p3_arm.sh \
       > $FORMAL_DIR/${arm}_driver.log 2>&1 < /dev/null &" 2>&1 | head -3
  sleep 5
}

sync_arm() {
  local arm="$1"
  local sum_dst="p3/results/${arm}_formal_summary.json"
  local log_tmp="/tmp/p3_${arm}_driver.log"
  local csv_dst="p3/results/${arm}_steps.csv"

  scp -q "$SSH_HOST:$FORMAL_DIR/$arm/events/summary.json" \
        "/tmp/p3_${arm}_formal.json" 2>/dev/null || {
    echo "[watch] $arm: summary 拉取失败，下轮重试"; return 1
  }
  # 驱动日志也要拉回来：判分支 A/B 用的是**每步趋势**，不是一个汇总数
  scp -q "$SSH_HOST:$FORMAL_DIR/${arm}_driver.log" "$log_tmp" 2>/dev/null || {
    echo "[watch] $arm: 驱动日志拉取失败（本轮仍提交 summary）"
    log_tmp=""
  }

  cd "$LOCAL_REPO" || return 1

  # 内容变了才提交——用内容比对而不是「文件在不在」，
  # 这样上一轮提交成功但推送失败的情况也能自动补齐
  local changed=0
  if ! cmp -s "/tmp/p3_${arm}_formal.json" "$sum_dst" 2>/dev/null; then
    cp "/tmp/p3_${arm}_formal.json" "$sum_dst"; changed=1
  fi

  if [ -n "$log_tmp" ]; then
    python3 p3/parse_steps.py "$log_tmp" --arm "$arm" \
            --csv-out "$csv_dst" > "/tmp/p3_${arm}_trend.json" 2>&1
    if ! cmp -s "$csv_dst" "/tmp/p3_${arm}_csv_prev" 2>/dev/null; then
      cp "$csv_dst" "/tmp/p3_${arm}_csv_prev"; changed=1
    fi
  fi

  if [ "$changed" -eq 0 ]; then
    echo "[watch] $arm: 结果无变化，跳过提交"; return 0
  fi

  git add "$sum_dst" "$csv_dst" 2>/dev/null
  git commit -q -m "P3: $arm 臂正式训练结果（$STEPS 步）

自动回传：整轮 summary + 每步指标 CSV。
判分支 A/B 看每步趋势，见 p3/PAPER_EDIT_MAP.md 第四节。" \
    && git push -q origin main \
    && echo "[watch] $arm: 已推送 summary + 每步 CSV" \
    || echo "[watch] $arm: 推送失败，产物保留在本地，下轮重试"
}

# ---------------------------------------------------------------- 主流程
PROBE=$(remote_probe)
if [ -z "$PROBE" ]; then
  echo "[watch] ✗ ssh 探测失败（机器关机或网络问题），本轮结束" | tee -a "$LOG"
  exit 0
fi

{
  echo "## $(date '+%Y-%m-%d %H:%M:%S')"

  B_LAUNCHED=$(arm_field "$PROBE" broken   2)
  B_SUMMARY=$( arm_field "$PROBE" broken   3)
  R_LAUNCHED=$(arm_field "$PROBE" repaired 2)
  R_SUMMARY=$( arm_field "$PROBE" repaired 3)

  echo "- broken:  launched=$B_LAUNCHED  summary=$B_SUMMARY  $(arm_prog "$PROBE" broken)"
  echo "- repaired:launched=$R_LAUNCHED  summary=$R_SUMMARY  $(arm_prog "$PROBE" repaired)"

  # ---- broken ----
  if [ "$B_LAUNCHED" != "launched" ]; then
    echo "- broken 未启动 → 启动"
    launch_arm broken
  elif [ "$B_SUMMARY" = "has_summary" ]; then
    echo "- broken 已完成"
    sync_arm broken
    if [ "$R_LAUNCHED" != "launched" ]; then
      echo "- → 启动 repaired 臂接力"
      launch_arm repaired
    fi
  else
    echo "- broken 运行中，本轮不动"
  fi

  # ---- repaired ----
  if [ "$R_SUMMARY" = "has_summary" ]; then
    echo "- repaired 已完成"
    sync_arm repaired
    echo "- ★★ 两臂全部完成，结果已在仓库，可以按分支 A/B 写论文了"
  elif [ "$R_LAUNCHED" = "launched" ]; then
    echo "- repaired 运行中，本轮不动"
  fi
  echo ""
} >> "$LOG" 2>&1

tail -14 "$LOG"
