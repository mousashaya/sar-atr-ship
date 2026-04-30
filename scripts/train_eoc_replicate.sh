#!/bin/bash
# Replication of paper's EOC benchmark using ConvNeXt.
# Each EOC condition trains on its own train split and evaluates on test split.
# Target numbers from paper Table 5 (ConvNeXt column):
#   EOC_depression: 37.2%
#   EOC_scene:      18.0%
#   EOC_pol:        70.5%
#   EOC_band:       83.8%
#   EOC_azimuth:    21.1%
#
# Usage:
#   caffeinate -dim bash scripts/train_eoc_replicate.sh 2>&1 | tee logs/eoc_replicate.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs
cd $CD

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

echo "=== EOC Replication started $(date) ===" | tee -a logs/eoc_replicate.log
echo "Target: reproduce paper ConvNeXt EOC numbers" | tee -a logs/eoc_replicate.log

run_eoc() {
    local SETTING=$1
    local TARGET=$2
    echo "" | tee -a logs/eoc_replicate.log
    echo "--- $SETTING (paper target: $TARGET) $(date) ---" | tee -a logs/eoc_replicate.log
    $PYTHON scripts/train.py \
        --model convnext \
        --config 40class \
        --setting $SETTING \
        --use-atrnet-splits \
        --epochs 15 \
        --patience 3 \
        --batch-size 64 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "eoc_replicate_${SETTING}_v1" \
        2>&1 | tee -a logs/eoc_replicate.log
    echo "--- Done $SETTING $(date) ---" | tee -a logs/eoc_replicate.log
}

run_eoc "EOC_depression"  "37.2%"
run_eoc "EOC_scene"       "18.0%"
run_eoc "EOC_pol"         "70.5%"
run_eoc "EOC_band"        "83.8%"
run_eoc "EOC_azimuth"     "21.1%"

echo "" | tee -a logs/eoc_replicate.log
echo "=== EOC REPLICATION DONE $(date) ===" | tee -a logs/eoc_replicate.log
echo "Compare results vs paper Table 5 ConvNeXt column" | tee -a logs/eoc_replicate.log
