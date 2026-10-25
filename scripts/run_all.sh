#!/usr/bin/env bash
# MasterRTL: One-Click End-to-End Simulation Pipeline (Bash)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
else
    PYTHON_EXE="python3"
fi

echo "=========================================================================="
echo " MasterRTL: Running One-Click Full Simulation Pipeline (Bash)"
echo "=========================================================================="
echo "[Env] Using Python: $PYTHON_EXE"

"$PYTHON_EXE" scripts/run_all.py --python "$PYTHON_EXE" "$@"
