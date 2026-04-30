#!/bin/bash
# Cascade Level 2: subclass classifiers conditioned on main class + angle.
# For each (main_class, angle) pair, trains a ConvNeXt on the subclasses
# of that main class only (images filtered by main_class AND depression_angle).
#
# Level 1 (4-class angle-specific): exp_022-025 — already done.
# Level 2 output: 16 models, one per (Bus/Car/Special/Truck) x (15/30/45/60).
#
# Usage:
#   bash scripts/train_cascade_level2.sh 2>&1 | tee logs/train_cascade_level2.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
LOG=$CD/logs/train_cascade_level2.log

mkdir -p $CD/logs
cd $CD

echo "=== Cascade Level 2 training started $(date) ===" | tee -a $LOG

# Prevent display/idle/disk sleep for the entire training run
caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

run_l2() {
    local PARENT=$1
    local ANGLE=$2
    echo "" | tee -a $LOG
    echo "--- L2 parent=$PARENT angle=${ANGLE}deg $(date) ---" | tee -a $LOG
    $PYTHON scripts/train.py \
        --model convnext \
        --config 21class \
        --parent-class "$PARENT" \
        --angle $ANGLE \
        --epochs 15 \
        --batch-size 32 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "cascade_L2_${PARENT}_angle${ANGLE}_v1" \
        2>&1 | tee -a $LOG
    echo "--- Done L2 parent=$PARENT angle=${ANGLE}deg $(date) ---" | tee -a $LOG
}

for ANGLE in 15 30 45 60; do
    run_l2 Bus     $ANGLE
    run_l2 Car     $ANGLE
    run_l2 Speacial $ANGLE
    run_l2 Truck   $ANGLE
done

echo "" | tee -a $LOG
echo "=== ALL LEVEL 2 DONE $(date) ===" | tee -a $LOG
echo "Next: review per-class results then decide on Level 3" | tee -a $LOG
