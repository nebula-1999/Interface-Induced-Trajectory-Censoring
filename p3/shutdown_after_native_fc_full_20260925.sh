#!/usr/bin/env bash
# User-authorized shutdown only after this exact full evaluation passes checks.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/native_fc_20260923
RUN="$ROOT/full_20260925"
exec 7>"$ROOT/full_20260925_shutdown.lock"
flock -n 7 || exit 0
trap 'echo "Shutdown withheld: validation or shutdown command failed at $(date -Is)"' ERR
echo "Armed at $(date -Is); waiting for full evaluation lock"
exec 8>"$ROOT/native_fc_pilot.lock"
flock -w 23000 8
test -f "$RUN/COMPLETE"
test ! -e "$RUN/INVALID"
/root/code-venv/bin/python - "$RUN" <<'PY'
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]); configs=[]
for arm in ['base','broken','repaired']:
    p=root/arm
    assert not (p/'INVALID.json').exists()
    c=json.loads((p/'configuration.json').read_text());configs.append(c)
    r=[json.loads(s) for s in (p/'trajectories.jsonl').read_text().splitlines()]
    assert c['arm']==arm and c['smoke'] is False
    assert json.loads((p/'COMPLETE.json').read_text())['n']==len(r)==542
    assert len(set(c['ids']))==542 and [x['task_id'] for x in r]==c['ids']
    for x in r:
        for e in x['events']:
            assert hashlib.sha256(e['raw'].encode()).hexdigest()==e['raw_sha256']
assert all(c['ids']==configs[0]['ids'] and c['sources']==configs[0]['sources'] for c in configs)
print('PRE_SHUTDOWN_AUDIT_PASS: 3 x 542 complete records, matched inputs, valid raw hashes')
PY
# Do not stop unrelated GPU work that may have been started meanwhile.
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
test -z "$gpu_pids"
printf 'Validated; shutdown requested at %s\n' "$(date -Is)" > "$RUN/SHUTDOWN_REQUESTED"
sync
echo "All evidence synced to data disk; requesting instance shutdown $(date -Is)"
/usr/bin/shutdown -h now
