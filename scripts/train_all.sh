#!/usr/bin/env bash
# MasterRTL Model Training and Evaluation Script
set -euo pipefail

PYTHON_BIN="python3"
if [ -f "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
elif [ -f "./.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="./.venv/Scripts/python.exe"
fi

echo "Running MasterRTL ML Model Training and 10-Fold Cross-Validation..."
$PYTHON_BIN scripts/train_all.py "$@"

