#!/bin/bash
# Retrains all exp_006-017 models using the corrected scene-based split.
# New runs get new exp IDs (018+) and are auto-discovered by eval scripts
# (find_vehicle_classifier picks the alphabetically latest match).
#
# Models retrained:
#   4class  x4 angles (15/30/45/60)
#   21class x4 angles
#   40class x4 angles
#
# Usage:
#   bash scripts/retrain_fixed_split.sh
#   bash scripts/retrain_fixed_split.sh 2>&1 | tee logs/retrain_fixed_split.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
LOG=$CD/logs/retrain_fixed_split.log

mkdir -p $CD/logs
cd $CD

echo "=== Retrain fixed split started $(date) ===" | tee -a $LOG

run_train() {
    local CONFIG=$1
    local ANGLE=$2
    echo "" | tee -a $LOG
    echo "--- Training config=$CONFIG angle=${ANGLE}deg $(date) ---" | tee -a $LOG
    $PYTHON scripts/train.py \
        --model convnext \
        --config $CONFIG \
        --angle $ANGLE \
        --epochs 10 \
        --batch-size 32 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "convnext_${CONFIG}_angle${ANGLE}_v1" \
        2>&1 | tee -a $LOG
    echo "--- Done config=$CONFIG angle=${ANGLE}deg $(date) ---" | tee -a $LOG
}

# ── 4class ────────────────────────────────────────────────────────────────────
for ANGLE in 15 30 45 60; do
    run_train 4class $ANGLE
done

# ── 21class ───────────────────────────────────────────────────────────────────
for ANGLE in 15 30 45 60; do
    run_train 21class $ANGLE
done

# ── 40class ───────────────────────────────────────────────────────────────────
for ANGLE in 15 30 45 60; do
    run_train 40class $ANGLE
done

echo "" | tee -a $LOG
echo "=== ALL RETRAINING DONE $(date) ===" | tee -a $LOG
echo "Next: run eval_eoc.py --all-settings and eval_eoc_depression.py" | tee -a $LOG
