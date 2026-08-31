#!/bin/bash
# 开卡后第一条命令。内存从 2 GiB 变成几百 GB，safe_workers 会自动从
# 并行度 1 放开到 112，无卡模式要跑 90 分钟的三个验证在这里约 5 分钟。
#
# 不含装栈——那个可能要调试，单独手动做。
set -uo pipefail
source /root/autodl-tmp/env.sh
cd "$PROJ" || exit 1
: > runall.done

echo "=== 0. 补齐模型下载（断点续传）==="
for M in Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  D=/root/autodl-tmp/models/$(basename "$M")
  hf download "$M" --local-dir "$D" > /dev/null 2>&1 && echo "  ok $(basename "$M")"
done

echo "=== 1. KodCode 数据质量（7669 条）==="
python verify_kodcode.py  2>&1 | tail -6 | tee -a runall.done

echo "=== 2. EvalPlus 官方解必须 542/542 ==="
python verify_evalplus.py 2>&1 | tail -6 | tee -a runall.done

echo "=== 3. 生成 542 份 scripted 修复探针 ==="
python build_probes.py    2>&1 | tail -8 | tee -a runall.done

echo ""
echo "=== 开卡前四项核对 ==="
V=$(python -c "import json;print(len(json.load(open('verified_ids.json'))['verified_index']))" 2>/dev/null || echo 0)
P=$(wc -l < probes_repair.jsonl 2>/dev/null || echo 0)
echo "verified_ids.json     $V 条   $([ "$V" -ge 2600 ] && echo '✅ 够用(需≥2600)' || echo '❌ 不足')"
echo "probes_repair.jsonl   $P 条   $([ "$P" -ge 450 ] && echo '✅ 够用(需≥450)' || echo '❌ 不足')"
python -m pytest test_sandbox.py test_code_tool_core.py test_eval_decompose.py \
  -q -p no:cacheprovider 2>&1 | tail -2
