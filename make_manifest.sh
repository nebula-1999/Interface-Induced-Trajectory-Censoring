#!/bin/bash
# 生成实验清单并原子化打包。矩阵跑完后执行。
# 解决两个问题：结果包缺可复现信息；本地可能读到未写完的 tgz。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent
cd "$P" || exit 1

{
  echo "# 实验清单  $(date -u +%FT%TZ)"
  echo "## 代码指纹"
  for f in probe_react_full.py matrix.sh sandbox.py; do
    [ -f "$f" ] && echo "  $f  sha256=$(sha256sum "$f" | cut -d' ' -f1)"
  done
  echo "## 环境"
  python - <<'PY'
import importlib
for m in ("torch", "vllm", "transformers", "datasets"):
    try:
        print(f"  {m}={importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"  {m}=? ({type(e).__name__})")
PY
  echo "  python=$(python -V 2>&1 | cut -d' ' -f2)"
  echo "  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "## 模型 revision"
  for d in /root/autodl-tmp/models/Qwen2.5-Coder-*-Instruct; do
    [ -d "$d" ] || continue
    echo "  $(basename "$d")  config_sha=$(sha256sum "$d/config.json" 2>/dev/null | cut -c1-16)"
  done
  echo "## 题目 ID（各组前若干，完整清单见 JSONL 的 clean_index 字段）"
  for f in traj_*.jsonl; do
    [ -f "$f" ] || continue
    echo "  $f  n=$(wc -l < "$f")  ids=$(head -3 "$f" | python -c "
import sys,json
print(','.join(str(json.loads(l)['clean_index']) for l in sys.stdin))" 2>/dev/null)..."
  done
  echo "## 各格校验状态"
  grep -E "^(OK|BAD)" runall.done 2>/dev/null || echo "  (无)"
} > MANIFEST.txt

# 原子化：先写 .tmp，校验完整性，再 mv，最后立 .ready 标记
tar czf matrix_results.tgz.tmp MANIFEST.txt mx_*.log traj_*.jsonl \
    probe_react_full.py matrix.sh ablation_*.log poscontrol_*.log \
    n200_*.log diag*.log runall.done 2>/dev/null
if tar tzf matrix_results.tgz.tmp >/dev/null 2>&1; then
  mv matrix_results.tgz.tmp matrix_results.tgz
  touch matrix_results.ready
  echo "MANIFEST + 原子打包完成 $(date +%H:%M:%S)" >> runall.done
else
  rm -f matrix_results.tgz.tmp
  echo "打包校验失败" >> runall.done
fi
