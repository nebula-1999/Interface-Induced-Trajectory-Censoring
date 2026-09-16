#!/bin/bash
# ARM=broken|repaired RUN_ID=<paired-id> [DRY=1]; defaults to a 1-step smoke.
set -euo pipefail
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PROJ_DIR
PY="${P3_PYTHON:-/root/code-venv/bin/python}"
exec "$PY" -I "$PROJ_DIR/p3/launch_arm.py"
