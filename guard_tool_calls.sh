#!/bin/bash
# 训练启动后若干分钟内，工具必须被真正执行过一次，否则立刻中止训练。
#
# 起因：2026-08-28 那次 150 步 GRPO，工具一次都没被调用
# （vLLM 默认 tool parser 与 Qwen2.5-Coder-1.5B 不兼容），
# 模型全程收到空的/幻觉的多轮反馈，4 小时 ¥20 白跑，
# 而唯一的破绽是几千行日志里的 timing_s/agent_loop/tool_calls/mean: 0.0。
#
# 用法：训练启动后立刻 setsid --fork ./guard_tool_calls.sh
WAIT_MIN=${1:-15}
FLAG=/root/autodl-tmp/runs/.tool_called
LOG=/root/autodl-tmp/code-agent/guard.log
exec >>"$LOG" 2>&1
echo "=== guard 启动 $(date +%F\ %T)，${WAIT_MIN} 分钟内工具必须被调用 ==="

for _ in $(seq "$((WAIT_MIN * 2))"); do
  sleep 30
  if [ -f "$FLAG" ]; then
    echo "$(date +%T) ✅ 工具已被执行，guard 退出"
    exit 0
  fi
  # 训练自己先结束了就不用管
  ps -eo args= | grep -q "[l]aunch_ppo.py" || { echo "$(date +%T) 训练已结束，guard 退出"; exit 0; }
done

echo "$(date +%T) ❌ ${WAIT_MIN} 分钟内工具从未被执行 —— 中止训练"
echo "   多轮反馈无效，继续跑只会产出无法解释的结果。"
echo "   先跑 probe_toolcall.py 定位是模型没发 tool_call 还是 parser 没解析出来。"
ps -eo pid=,args= | awk '/launch_ppo.py|run_code_grpo/ && !/awk/ {print $1}' | xargs -r kill
echo "TOOL NEVER CALLED $(date +%H:%M:%S)" >> /root/autodl-tmp/code-agent/runall.done
