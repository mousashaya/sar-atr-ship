#!/bin/bash
# Trains flat ConvNeXt baselines (no angle routing) on the corrected acquisition-key split.
# Each model trains on ALL depression angles combined — no --angle flag.
# These are the internal baselines that isolate the routing benefit:
#   flat (all angles) vs. routed (angle-specific) on the same corrected split.
#
# Configs: 4class, 21class, 40class
#
# Usage:
#   bash scripts/train_flat_baseline.sh 2>&1 | tee logs/train_flat_baseline.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
LOG=$CD/logs/train_flat_baseline.log

mkdir -p $CD/logs
cd $CD

echo "=== Flat baseline training started $(date) ===" | tee -a $LOG

run_train() {
    local CONFIG=$1
    echo "" | tee -a $LOG
    echo "--- Training flat config=$CONFIG $(date) ---" | tee -a $LOG
    $PYTHON scripts/train.py \
        --model convnext \
        --config $CONFIG \
        --epochs 10 \
        --batch-size 32 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "convnext_${CONFIG}_flat_v1" \
        2>&1 | tee -a $LOG
    echo "--- Done flat config=$CONFIG $(date) ---" | tee -a $LOG
}

run_train 4class
run_train 21class
run_train 40class

echo "" | tee -a $LOG
echo "=== ALL FLAT BASELINES DONE $(date) ===" | tee -a $LOG
echo "Next: run eval_eoc.py for both routed and flat models" | tee -a $LOG
