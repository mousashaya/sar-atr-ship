#!/bin/bash
# Cascade Level 3: specific vehicle type classifiers conditioned on subclass + angle.
# Only runs for subclasses that have >1 target type (9 are already leaves).
#
# Leaves (no L3 needed):
#   Bus:     Large_Bus
#   Special: Ambulance, ECV, Road_Roller, Shovel_Loader (all 4)
#   Truck:   Forklift, Heavy_DT, Light_PV, Mixer_Truck
#
# L3 models: 12 subclasses × 4 angles = 48 models
#
# Usage:
#   caffeinate -dim bash scripts/train_cascade_level3.sh 2>&1 | tee logs/train_cascade_level3.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs
cd $CD

echo "=== Cascade Level 3 training started $(date) ===" | tee -a logs/train_cascade_level3.log

# Prevent display/idle/disk sleep for the entire run
caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

run_l3() {
    local SUBCLASS=$1
    local ANGLE=$2
    local SAFE=$(echo "$SUBCLASS" | tr "' " "__")
    echo "" | tee -a logs/train_cascade_level3.log
    echo "--- L3 subclass=$SUBCLASS angle=${ANGLE}deg $(date) ---" | tee -a logs/train_cascade_level3.log
    $PYTHON scripts/train.py \
        --model convnext \
        --config 40class \
        --subclass "$SUBCLASS" \
        --angle $ANGLE \
        --epochs 20 \
        --patience 3 \
        --batch-size 64 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "cascade_L3_${SAFE}_angle${ANGLE}_v1" \
        2>&1 | tee -a logs/train_cascade_level3.log
    echo "--- Done L3 subclass=$SUBCLASS angle=${ANGLE}deg $(date) ---" | tee -a logs/train_cascade_level3.log
}

# Bus (2 subclasses with multiple types)
for ANGLE in 15 30 45 60; do
    run_l3 "Medium_Bus"   $ANGLE
    run_l3 "Small_Bus"    $ANGLE
done

# Car (all 5 subclasses have multiple types)
for ANGLE in 15 30 45 60; do
    run_l3 "Compact_SUV"  $ANGLE
    run_l3 "Large_Car"    $ANGLE
    run_l3 "Medium_Car"   $ANGLE
    run_l3 "Medium_SUV"   $ANGLE
    run_l3 "Mini_Car"     $ANGLE
done

# Truck (5 subclasses with multiple types)
for ANGLE in 15 30 45 60; do
    run_l3 "Heavy_FT"     $ANGLE
    run_l3 "Heavy_ST"     $ANGLE
    run_l3 "Light_DT"     $ANGLE
    run_l3 "Medium_TT"    $ANGLE
    run_l3 "Pickup"       $ANGLE
done

echo "" | tee -a logs/train_cascade_level3.log
echo "=== ALL LEVEL 3 DONE $(date) ===" | tee -a logs/train_cascade_level3.log
echo "Next: evaluate full cascade end-to-end, then run flat 40-class baseline" | tee -a logs/train_cascade_level3.log
