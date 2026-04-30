#!/bin/bash
# Pol-routing smoke test — Level 1 only.
# Trains 4 pol-specific 4-class models (HH, HV, VH, VV) on SOC data.
# Compare EOC-Pol accuracy using pol-routing vs angle-routing (67.6%).
#
# Usage:
#   caffeinate -dim bash scripts/train_pol_l1_smoke.sh 2>&1 | tee logs/pol_l1_smoke.log

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense

mkdir -p $CD/logs
cd $CD

caffeinate -dim &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null" EXIT

echo "=== Pol-routing L1 smoke test started $(date) ===" | tee -a logs/pol_l1_smoke.log

for POL in HH HV VH VV; do
    echo "" | tee -a logs/pol_l1_smoke.log
    echo "--- L1 polarization=$POL $(date) ---" | tee -a logs/pol_l1_smoke.log
    $PYTHON scripts/train.py \
        --model convnext \
        --config 4class \
        --polarization $POL \
        --epochs 10 \
        --patience 2 \
        --batch-size 64 \
        --lr 0.0001 \
        --num-workers 4 \
        --name "pol_L1_${POL}_v1" \
        2>&1 | tee -a logs/pol_l1_smoke.log
    echo "--- Done L1 pol=$POL $(date) ---" | tee -a logs/pol_l1_smoke.log
done

echo "" | tee -a logs/pol_l1_smoke.log
echo "=== POL L1 DONE $(date) ===" | tee -a logs/pol_l1_smoke.log
echo "Next: run eval_pol_routing.py to compare pol-routing vs angle-routing on EOC-Pol" | tee -a logs/pol_l1_smoke.log
