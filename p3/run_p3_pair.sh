#!/bin/bash
# Sequential formal pair. If the host stops between arms, rerun only the missing
# arm manually with the same RUN_ID; existing arm directories are never reused.
set -euo pipefail
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${RUN_ID:?RUN_ID is required}"
: "${P3_ACCEPTANCE_RECORD:?P3_ACCEPTANCE_RECORD is required}"
export PROJ_DIR P3_ACCEPTANCE_RECORD P3_MODE=formal

for arm in broken repaired; do
  echo "[p3-pair] ===== start $arm ====="
  ARM="$arm" bash "$PROJ_DIR/p3/run_p3_arm.sh"
  echo "[p3-pair] ===== completed $arm ====="
done
