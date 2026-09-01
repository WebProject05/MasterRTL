#!/usr/bin/env bash
# One-click Bash runner for generating all graph types into data/graphs/

set -e

echo "================================================================="
echo " MasterRTL: Multi-Representation Graph Generation Suite"
echo "================================================================="

PYTHON_EXE="./.venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="./.venv/Scripts/python.exe"
fi
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

echo "[Env] Using Python: $PYTHON_EXE"
echo "[Run] Launching multi-representation graph generator..."

$PYTHON_EXE -m src.data_prep.batch_graphs data/raw_rtl -o data/graphs --stage-dir data/yosys_stages

echo "================================================================="
echo " All graph representations successfully generated in data/graphs!"
echo " Manifest: data/graphs/graph_manifest.json"
echo " Summary:  data/graphs/graph_summary.md"
echo "================================================================="

