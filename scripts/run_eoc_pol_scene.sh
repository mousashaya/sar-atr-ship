#!/bin/bash
# EOC flat baselines: EOC_pol and EOC_scene.
# Run after run_eoc_depression.sh completes.
#
# Usage:
#   caffeinate -dim bash scripts/run_eoc_pol_scene.sh 2>&1 | tee -a logs/eoc_depression.log

set -e

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
mkdir -p $CD/logs
cd $CD

COMMON="--model convnext_small --epochs 15 --patience 3 --batch-size 64 --lr 0.0001 --num-workers 4 --config 40class"

echo "" | tee -a logs/eoc_depression.log
echo "=== EOC_pol + EOC_scene started $(date) ===" | tee -a logs/eoc_depression.log

echo "" | tee -a logs/eoc_depression.log
echo "--- EOC_pol flat (train HH, test HV/VH/VV) $(date) ---" | tee -a logs/eoc_depression.log
$PYTHON scripts/train.py \
    $COMMON \
    --setting EOC_pol \
    --use-atrnet-splits \
    --name "eoc_pol_flat_v1" \
    2>&1 | tee -a logs/eoc_depression.log
echo "--- EOC_pol done $(date) ---" | tee -a logs/eoc_depression.log

echo "" | tee -a logs/eoc_depression.log
echo "--- EOC_scene flat (train simple, test factory/urban/woodland) $(date) ---" | tee -a logs/eoc_depression.log
$PYTHON scripts/train.py \
    $COMMON \
    --setting EOC_scene \
    --use-atrnet-splits \
    --name "eoc_scene_flat_v1" \
    2>&1 | tee -a logs/eoc_depression.log
echo "--- EOC_scene done $(date) ---" | tee -a logs/eoc_depression.log

echo "" | tee -a logs/eoc_depression.log
echo "=== EOC_pol + EOC_scene done $(date) ===" | tee -a logs/eoc_depression.log
