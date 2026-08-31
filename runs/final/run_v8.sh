#!/bin/bash
# 定性 v3 Llama8B_fc(terse) 里首轮「发起了调用但拿不到 code」的那批题（实测 23 条）。
#
# 能回答什么 / 不能回答什么：
#   旧探针未保存函数名，因此**无法追溯**原始那 23 条各自是哪种失败。
#   本臂只能回答：在**相同配置**下重跑，它们分别复现为什么。
#   即使 temperature=0，不显式固定 seed 时输出仍可能变化，故本臂显式传 --seed 0，
#   并同时报告「复现一致率」与「恢复率」，不作追溯性断言。
set -uo pipefail
source /root/autodl-tmp/env.sh
P=/root/autodl-tmp/code-agent; cd "$P" || exit 1
for _ in $(seq 240); do { [ -f V6B_READY ] || [ -f V6B_INVALID ]; } && break; sleep 30; done
rm -f V8_READY V8_INVALID v8_manifest.txt
LL=/root/autodl-tmp/models/Llama-3.1-8B-Instruct
OUT=traj_v8_Llama8B_recheck.jsonl
rm -f "$OUT"

IDS=$(python - <<'PY'
import json
R=[json.loads(l) for l in open("traj_v3_Llama8B_fc.jsonl",encoding="utf-8") if l.strip()]
ids=[r["clean_index"] for r in R
     if (r["turns"] or [{}])[0].get("action") and not (r["turns"] or [{}])[0].get("code")]
print(",".join(str(i) for i in ids))
PY
)
NEXP=$(echo "$IDS" | tr ',' '\n' | grep -c .)
echo "待定性 $NEXP 条: $IDS"
echo "expected=$NEXP ids=$IDS" >> v8_manifest.txt

wait_port_free(){ for _ in $(seq 90); do ss -ltn 2>/dev/null | grep -q ":8000 " || return 0; sleep 2; done; return 1; }
wait_port_free
setsid --fork python -m vllm.entrypoints.openai.api_server \
  --model "$LL" --served-model-name "$LL" --port 8000 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  < /dev/null > vllm_v8.log 2>&1
UP=0
for _ in $(seq 240); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && { UP=1; break; }
  ps -eo args= | grep -q "[a]pi_server" || break; sleep 5; done
if [ "$UP" != "1" ]; then
  echo "SERVE_FAIL" >> v8_manifest.txt; grep -iE "error|Traceback" vllm_v8.log | head -4
  touch V8_INVALID; echo "V8_INVALID $(date +%H:%M:%S)" >> runall.done; exit 1
fi

# 与 v3 那一臂同配置：terse schema、仅 parser、无官方模板、cross_family
python probe_react_full.py --model "$LL" --port 8000 --only-ids "$IDS" \
  --protocol fc --strength optional --parser-adapter cross_family --seed 0 \
  --out "$OUT" 2>&1 | grep -vE "examples/s\]|it/s\]"
RC=${PIPESTATUS[0]}
LINES=$(wc -l < "$OUT" 2>/dev/null || echo 0)
NERR=$(python - <<PY
import json
try:
    R=[json.loads(l) for l in open("$OUT",encoding="utf-8") if l.strip()]
except FileNotFoundError:
    print(-1); raise SystemExit
print(sum(1 for r in R for t in r["turns"] if (t.get("raw_output") or "").startswith("__ERROR__")))
PY
)
echo "rc=$RC lines=$LINES n_err=$NERR" >> v8_manifest.txt
ps -eo pid=,args= | awk '/api_server/ && !/awk/ {print $1}' | xargs -r kill 2>/dev/null
wait_port_free

python - <<PY
import json
from collections import Counter
OUT="$OUT"; NEXP=$NEXP
R=[json.loads(l) for l in open(OUT,encoding="utf-8") if l.strip()]
print(f"\n########## 重跑定性  n={len(R)} / 预期 {NEXP} ##########")
cls=Counter(); detail=Counter()
for r in R:
    t=(r["turns"] or [{}])[0]
    mode=t.get("parse_mode"); name=t.get("_fc_tool_name"); err=t.get("_fc_arg_err")
    if mode=="request_error": cls["请求错误"]+=1
    elif t.get("_fc_wrong_tool") or mode=="unknown_tool":
        cls["wrong_tool"]+=1; detail[f"wrong_tool:{name}"]+=1
    elif name=="run_tests" and not t.get("code"):
        cls["run_tests + arg_err"]+=1; detail[str(err)]+=1
    elif name=="run_tests" and t.get("code"):
        cls["run_tests + 有效代码"]+=1
    else:
        cls[f"其他({mode})"]+=1
for k,v in cls.most_common(): print(f"   {k:<24} {v}")
if detail: print(f"   明细: {dict(detail)}")
same=cls.get("run_tests + arg_err",0)+cls.get("wrong_tool",0)
print(f"\n   复现一致率（仍未拿到代码）: {same}/{len(R)}")
print(f"   恢复率（重跑拿到有效代码）: {cls.get('run_tests + 有效代码',0)}/{len(R)}")
print("   ※ 本表描述的是相同配置下的重跑行为，不能追溯原始那批各自属于哪类")
PY

if [ "$RC" = "0" ] && [ "$LINES" = "$NEXP" ] && [ "$NERR" = "0" ]; then
  touch V8_READY; echo "V8_READY $(date +%H:%M:%S)" >> runall.done
else
  touch V8_INVALID; echo "V8_INVALID rc=$RC lines=$LINES/$NEXP n_err=$NERR $(date +%H:%M:%S)" >> runall.done
fi
echo "########## v8 manifest ##########"; cat v8_manifest.txt
