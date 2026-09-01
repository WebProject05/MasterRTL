#!/usr/bin/env bash
# MasterRTL End-to-End Execution Script for Bash / Linux / macOS
set -euo pipefail

RAW_DIR="${1:-data/raw_rtl}"
SYNTH_DIR="${2:-data/yosys_synth}"
SOG_DIR="${3:-data/sog_graphs}"
REPORT_DIR="${4:-data/reports}"
PYTHON_BIN="${5:-python3}"

if [ -f "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
elif [ -f "./.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="./.venv/Scripts/python.exe"
fi

echo "================================================================="
echo " MasterRTL: Pre-synthesis SOG Graph Generation & Stats Pipeline"
echo "================================================================="
echo "Using Python: $PYTHON_BIN"
echo "Processing designs from: $RAW_DIR"

$PYTHON_BIN -m src.data_prep.batch_parse "$RAW_DIR" \
    -o "$SYNTH_DIR" \
    --sog-dir "$SOG_DIR" \
    --report-dir "$REPORT_DIR"

echo "================================================================="
echo " Pipeline execution complete!"
echo " Reports available in $REPORT_DIR:"
echo "   - $REPORT_DIR/sog_statistics.csv"
echo "   - $REPORT_DIR/sog_statistics.json"
echo "   - $REPORT_DIR/sog_summary.md"
echo "================================================================="

