#!/bin/bash
set -e
VENV="venv/bin/python"

run_job() {
    local desc="$1"
    shift
    echo ""
    echo "=============================================="
    echo "  STARTING: $desc"
    echo "  $(date)"
    echo "=============================================="
    $VENV scripts/train_ship_classify.py "$@"
    echo ""
    echo "  DONE: $desc  ($(date))"
    echo "=============================================="
}

run_job "SLC HIGH 3-class CROMA — match-slc" \
    --high --three-class --backbone croma --match-slc

run_job "SLC HIGH 3-class DINOv2 — full (rerun)" \
    --high --three-class

echo ""
echo "All jobs complete. $(date)"
