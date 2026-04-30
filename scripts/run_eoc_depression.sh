#!/bin/bash
# EOC-Depression flat vs cascade comparison using ConvNeXt-Small.
#
# Cascade runs first (more models, want results together), then flat baseline.
# All models trained on 15° data only, tested on 30/45/60°.
#
# After this script finishes, run:
#   python3 scripts/eval_eoc_cascade.py
#
# Usage:
#   caffeinate -dim bash scripts/run_eoc_depression.sh 2>&1 | tee logs/eoc_depression.log

set -e

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
mkdir -p $CD/logs
cd $CD

echo "=== EOC-Depression experiment started $(date) ===" | tee -a logs/eoc_depression.log

COMMON="--model convnext_small --epochs 15 --patience 3 --batch-size 64 --lr 0.0001 --num-workers 4"
EOC="--setting EOC_depression --use-atrnet-splits"

# ─── Experiment B: Cascade (L1 first, then 4 L2 models) ──────────────────────

echo "" | tee -a logs/eoc_depression.log
echo "--- Cascade L1 (4-way main class) $(date) ---" | tee -a logs/eoc_depression.log
$PYTHON scripts/train.py \
    $COMMON $EOC \
    --config 4class \
    --name "eoc_dep_L1_v1" \
    2>&1 | tee -a logs/eoc_depression.log
echo "--- L1 done $(date) ---" | tee -a logs/eoc_depression.log

for MC in "Bus" "Car" "Speacial" "Truck"; do
    echo "" | tee -a logs/eoc_depression.log
    echo "--- Cascade L2 $MC $(date) ---" | tee -a logs/eoc_depression.log
    $PYTHON scripts/train.py \
        $COMMON $EOC \
        --config 40class \
        --parent-class "$MC" \
        --name "eoc_dep_L2_${MC}_v1" \
        2>&1 | tee -a logs/eoc_depression.log
    echo "--- L2 $MC done $(date) ---" | tee -a logs/eoc_depression.log
done

# ─── Experiment A: Flat 40-class baseline ────────────────────────────────────

echo "" | tee -a logs/eoc_depression.log
echo "--- Flat 40-class baseline $(date) ---" | tee -a logs/eoc_depression.log
$PYTHON scripts/train.py \
    $COMMON $EOC \
    --config 40class \
    --name "eoc_dep_flat_v1" \
    2>&1 | tee -a logs/eoc_depression.log
echo "--- Flat done $(date) ---" | tee -a logs/eoc_depression.log

echo "" | tee -a logs/eoc_depression.log
echo "=== All training done $(date) ===" | tee -a logs/eoc_depression.log
echo "Run: python3 scripts/eval_eoc_cascade.py" | tee -a logs/eoc_depression.log
