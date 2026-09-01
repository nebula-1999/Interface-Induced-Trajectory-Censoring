#!/usr/bin/env bash
# 把 P2 载荷推到新机并起跑。先确认公钥已装：ssh autodl-p2 true
set -euo pipefail
H=autodl-p2
R=/root/autodl-tmp/p2
D="$(cd "$(dirname "$0")/.." && pwd)"

echo "== 连通性 =="
ssh -o BatchMode=yes "$H" 'echo OK; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root/autodl-tmp | tail -1'

echo "== 推送载荷 =="
ssh "$H" "mkdir -p $R"
scp "$D/probe_react_full.py" "$D/sandbox.py" "$D/clean_ids.json" \
    "$D/preflight_toolcall.py" "$D/validate_arms.py" \
    "$D/p2/run_p2.sh" "$H:$R/"
ssh "$H" "mkdir -p $R/analysis && chmod +x $R/run_p2.sh"
scp "$D/analysis/intent.py" "$H:$R/analysis/"

echo "== 环境自检（不自动装，装错了比没装更难查）=="
ssh "$H" 'python3 -V; pip show vllm 2>/dev/null | head -2 || echo "★ vllm 未装 —— 必须装 0.27.1"; python3 -c "import datasets" 2>/dev/null && echo "datasets OK" || echo "★ 缺 datasets"'

cat <<'MSG'

载荷已就位。确认上面 vllm 是 0.27.1 之后，起跑：

    ssh autodl-p2 'cd /root/autodl-tmp/p2 && setsid --fork ./run_p2.sh </dev/null >p2_boot.log 2>&1'

盯进度：

    ssh autodl-p2 'tail -f /root/autodl-tmp/p2/p2_run.log'

MSG
