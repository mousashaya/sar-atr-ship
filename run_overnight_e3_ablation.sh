#!/bin/bash
# Overnight: E3 data efficiency n=300/400/500, angle ablation, DINOv2 prototypical
# Start watcher separately: python3 scripts/watch_overnight.py

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
LOG=/Users/mousashaya/Projects/defense/logs/overnight_e3_ablation.log
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs $CD/logs/data_efficiency $CD/logs/angle_ablation $CD/logs/prototypical_dinov2

echo "=== Overnight run started $(date) ===" | tee -a $LOG

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

cd $CD

# ── E3 extension: n=300, 400, 500 ─────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== E3 data efficiency n=300 $(date) ===" | tee -a $LOG
$PYTHON scripts/train_data_efficiency.py --max-per-group 300 --epochs 5 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== E3 data efficiency n=400 $(date) ===" | tee -a $LOG
$PYTHON scripts/train_data_efficiency.py --max-per-group 400 --epochs 5 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== E3 data efficiency n=500 $(date) ===" | tee -a $LOG
$PYTHON scripts/train_data_efficiency.py --max-per-group 500 --epochs 5 2>&1 | tee -a $LOG

# ── Angle ablation ─────────────────────────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== Angle ablation: 15+60 $(date) ===" | tee -a $LOG
$PYTHON scripts/train_angle_ablation.py --angles 15 60 --epochs 5 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Angle ablation: 15+30+60 $(date) ===" | tee -a $LOG
$PYTHON scripts/train_angle_ablation.py --angles 15 30 60 --epochs 5 2>&1 | tee -a $LOG

# ── DINOv2 prototypical (no training, ~30-45 min) ─────────────────────────────
echo "" | tee -a $LOG
echo "=== DINOv2 prototypical sweep $(date) ===" | tee -a $LOG
$PYTHON scripts/eval_prototypical_dinov2.py 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL DONE $(date) ===" | tee -a $LOG

# Plot updated E3 curve with all N values
echo "=== Plotting E3 curve ===" | tee -a $LOG
$PYTHON scripts/train_data_efficiency.py --plot-curve 2>&1 | tee -a $LOG

echo "=== Plotting angle ablation ===" | tee -a $LOG
# Angle ablation plot is generated at end of --all run, already done above

echo "=== Log at: $LOG ===" | tee -a $LOG
