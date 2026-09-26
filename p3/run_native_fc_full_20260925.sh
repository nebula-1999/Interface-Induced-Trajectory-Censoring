#!/usr/bin/env bash
# Authorized full evaluation, frozen pilot protocol. No training or shutdown.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/native_fc_20260923
WORK="$ROOT/workspace"
PY=/root/code-venv/bin/python
RUN="$ROOT/full_20260925"
exec 9>"$ROOT/native_fc_pilot.lock"
flock -n 9 || { echo 'An evaluation already holds the lock'; exit 1; }
mkdir "$RUN"
trap 'rc=$?; printf "failed rc=%s at %s\n" "$rc" "$(date -Is)" > "$RUN/INVALID"; exit "$rc"' ERR
cd "$WORK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo 'dd31b34547df7740431b851ad87c4a33990c8ece66912ac950f2bc1de60d3b46  p3/native_fc_eval.py' | sha256sum -c -
sha256sum p3/native_fc_eval.py qwen_tools_parser.py code_tool_core.py sandbox.py eval_manifest.json > "$RUN/input_hashes.txt"
printf 'Start %s; 542 tasks x 3 arms; total runtime budget 21600 seconds\n' "$(date -Is)"
SECONDS=0
for arm in base broken repaired; do
  remaining=$((21600 - SECONDS))
  test "$remaining" -gt 0
  args=()
  if [[ "$arm" != base ]]; then args=(--adapter "$ROOT/${arm}_step150"); fi
  printf 'START arm=%s %s remaining_budget=%ss\n' "$arm" "$(date -Is)" "$remaining"
  timeout --signal=TERM --kill-after=30s "$remaining" "$PY" p3/native_fc_eval.py \
    --base /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
    --arm "$arm" "${args[@]}" --manifest eval_manifest.json \
    --out "$RUN/$arm" --allow-gpu > "$RUN/${arm}.log" 2>&1
  "$PY" - "$RUN/$arm" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); c=json.loads((p/'configuration.json').read_text())
r=[json.loads(s) for s in (p/'trajectories.jsonl').read_text().splitlines()]
assert not (p/'INVALID.json').exists()
assert json.loads((p/'COMPLETE.json').read_text())['n']==len(r)==542
assert c['smoke'] is False and len(set(c['ids']))==542
assert [x['task_id'] for x in r]==c['ids']
for x in r:
    for e in x['events']:
        assert hashlib.sha256(e['raw'].encode()).hexdigest()==e['raw_sha256']
        if e['executed']:
            assert e['observation']
            assert any(m['role']=='tool' and m.get('tool_call_id')==e['request_id'] for m in x['messages'])
print('ARM_AUDIT_PASS',c['arm'],'tasks',len(r),'success',sum(x['final_pass'] for x in r),
      'executions',sum(x['tool_executions'] for x in r))
PY
  printf 'DONE arm=%s %s\n' "$arm" "$(date -Is)"
done
printf 'Three full arms complete %s\n' "$(date -Is)" > "$RUN/COMPLETE"
cat "$RUN/COMPLETE"
