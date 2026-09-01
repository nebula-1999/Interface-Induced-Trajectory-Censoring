#!/usr/bin/env bash
# Deploy P1 to the old GPU. Override OLD_GPU_HOST if its SSH alias changes.
set -euo pipefail

H=${OLD_GPU_HOST:-autodl-code}
R=${P1_REMOTE_ROOT:-/root/autodl-tmp/p1}
D=$(cd "$(dirname "$0")/.." && pwd)

echo "== old GPU connectivity =="
ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" \
  'echo OK; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader; df -h / /root/autodl-tmp | tail -2'

echo "== deploy isolated P1 payload =="
ssh "$H" "mkdir -p '$R/plugin' '$R/logs'"
# register_qwen_coder.py is required: sitecustomize.py imports it, so leaving it
# out makes every benchmark subprocess fail at startup.
scp "$D/p1/run_p1.sh" "$D/p1/toolcall_proxy.py" "$D/p1/analyze_p1.py" \
  "$D/p1/bfcl_registration.py" "$D/p1/sitecustomize.py" \
  "$D/p1/register_qwen_coder.py" "$D/p1/make_manifest.py" \
  "$D/p1/validate_p1.py" "$H:$R/"
scp "$D/runs/final/plugin/qwen2_5_coder_tool_parser.py" \
  "$D/runs/final/plugin/tool_chat_template_qwen2_5_coder.jinja" "$H:$R/plugin/"
ssh "$H" "chmod +x '$R/run_p1.sh'; test -f /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct/config.json"

echo "== start detached =="
ssh "$H" "cd '$R' && rm -f P1_INVALID && setsid --fork ./run_p1.sh </dev/null >p1_boot.log 2>&1"
sleep 3
ssh "$H" "cd '$R' && ps -eo pid,etime,args | grep '[r]un_p1.sh' || true; tail -30 p1_boot.log; tail -30 p1_run.log 2>/dev/null || true"

echo
echo "P1 deployed to $H:$R"
echo "Progress: ssh $H 'tail -f $R/p1_run.log'"
