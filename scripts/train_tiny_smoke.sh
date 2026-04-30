#!/bin/bash
# ConvNeXt-Tiny smoke test — L2 only, two conditions.
# Car/15°  (exp_031, worst L2: 79.9%)  — hardest case, expect most degradation
# Truck/15° (exp_033, best L2: 98.1%) — easiest case, should hold up fine
#
# If Tiny matches Base within ~2%, L2+L3 can switch to Tiny:
#   Base cascade: L1(Base) + L2(Base) + L3(Base) = ~45 GFLOPs = 3× flat
#   Tiny cascade: L1(Base) + L2(Tiny) + L3(Tiny) = ~24 GFLOPs = ~1.6× flat
#
# Usage:
#   caffeinate -dim bash scripts/train_tiny_smoke.sh 2>&1 | tee logs/tiny_smoke.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs
cd $CD

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

echo "=== ConvNeXt-Tiny smoke test started $(date) ===" | tee -a logs/tiny_smoke.log

echo "" | tee -a logs/tiny_smoke.log
echo "--- Tiny: Car/15° (hardest L2) $(date) ---" | tee -a logs/tiny_smoke.log
$PYTHON scripts/train.py \
    --model convnext_tiny \
    --config 21class \
    --parent-class Car \
    --angle 15 \
    --epochs 10 \
    --patience 2 \
    --batch-size 64 \
    --lr 0.0001 \
    --num-workers 4 \
    --name "tiny_L2_Car_angle15_v1" \
    2>&1 | tee -a logs/tiny_smoke.log
echo "--- Done Car/15° $(date) ---" | tee -a logs/tiny_smoke.log

echo "" | tee -a logs/tiny_smoke.log
echo "--- Tiny: Truck/15° (easiest L2) $(date) ---" | tee -a logs/tiny_smoke.log
$PYTHON scripts/train.py \
    --model convnext_tiny \
    --config 21class \
    --parent-class Truck \
    --angle 15 \
    --epochs 10 \
    --patience 2 \
    --batch-size 64 \
    --lr 0.0001 \
    --num-workers 4 \
    --name "tiny_L2_Truck_angle15_v1" \
    2>&1 | tee -a logs/tiny_smoke.log
echo "--- Done Truck/15° $(date) ---" | tee -a logs/tiny_smoke.log

echo "" | tee -a logs/tiny_smoke.log
echo "=== TINY SMOKE TEST DONE $(date) ===" | tee -a logs/tiny_smoke.log
echo "Compare results vs Base:" | tee -a logs/tiny_smoke.log
echo "  Car/15°   Base=79.9%  →  Tiny=?" | tee -a logs/tiny_smoke.log
echo "  Truck/15° Base=98.1%  →  Tiny=?" | tee -a logs/tiny_smoke.log
