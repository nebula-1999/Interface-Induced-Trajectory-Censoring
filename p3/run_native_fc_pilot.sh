#!/usr/bin/env bash
# Bounded evaluation only; no training, deletion, shutdown, or full-run relay.
set -Eeuo pipefail
ROOT=/root/autodl-tmp/native_fc_20260923
WORK="$ROOT/workspace"
PY=/root/code-venv/bin/python
RUN="$ROOT/pilot_24_20260923"
exec 9>"$ROOT/native_fc_pilot.lock"
flock -n 9 || { echo 'Another pilot holds the lock'; exit 1; }
mkdir "$RUN" # fail closed: do not mix/reuse outputs
trap 'rc=$?; printf "failed rc=%s at %s\n" "$rc" "$(date -Is)" > "$RUN/INVALID"; exit "$rc"' ERR
cd "$WORK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf 'Start %s; 24 fixed tasks per arm; 900-second cap per arm\n' "$(date -Is)"
for arm in base broken repaired; do
  args=()
  if [[ "$arm" != base ]]; then args=(--adapter "$ROOT/${arm}_step150"); fi
  printf 'START arm=%s %s\n' "$arm" "$(date -Is)"
  timeout --signal=TERM --kill-after=30s 900 "$PY" p3/native_fc_eval.py \
    --base /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct \
    --arm "$arm" "${args[@]}" --manifest eval_manifest.json \
    --out "$RUN/$arm" --limit 24 --allow-gpu > "$RUN/${arm}.log" 2>&1
  test -f "$RUN/$arm/COMPLETE.json"
  test ! -f "$RUN/$arm/INVALID.json"
  printf 'DONE arm=%s %s\n' "$arm" "$(date -Is)"
done
printf 'All three bounded arms complete %s\n' "$(date -Is)" > "$RUN/COMPLETE"
cat "$RUN/COMPLETE"
