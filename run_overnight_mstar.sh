#!/bin/bash
# Overnight: MSTAR SOC baseline, MSTAR prototypical, cross-dataset transfer
# Run AFTER run_overnight_e3_ablation.sh finishes (GPU serialized)

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
LOG=/Users/mousashaya/Projects/defense/logs/overnight_mstar.log
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs $CD/logs/mstar_prototypical $CD/logs/mstar_transfer

echo "=== MSTAR overnight run started $(date) ===" | tee -a $LOG

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

cd $CD

# ── Step 1: MSTAR SOC baseline (~1 hr) ────────────────────────────────────────
echo "" | tee -a $LOG
echo "=== MSTAR SOC: train 17° → test 15° $(date) ===" | tee -a $LOG
$PYTHON scripts/train_mstar_soc.py --epochs 15 2>&1 | tee -a $LOG
echo "=== MSTAR SOC done $(date) ===" | tee -a $LOG

# ── Step 2: MSTAR prototypical (~20 min) ──────────────────────────────────────
echo "" | tee -a $LOG
echo "=== MSTAR prototypical $(date) ===" | tee -a $LOG
$PYTHON scripts/eval_mstar_prototypical.py 2>&1 | tee -a $LOG
echo "=== MSTAR prototypical done $(date) ===" | tee -a $LOG

# ── Step 3: Cross-dataset transfer (~20 min) ───────────────────────────────────
echo "" | tee -a $LOG
echo "=== Cross-dataset transfer: ATRNet-STAR → MSTAR $(date) ===" | tee -a $LOG
$PYTHON scripts/eval_mstar_transfer.py 2>&1 | tee -a $LOG
echo "=== Cross-dataset transfer done $(date) ===" | tee -a $LOG

echo "" | tee -a $LOG
echo "=== ALL MSTAR DONE $(date) ===" | tee -a $LOG
