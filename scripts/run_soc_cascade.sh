#!/bin/bash
# SOC cascade vs flat comparison.
#
# Tests whether routing by physical property (angle/band/pol/azimuth)
# helps a specialized L2 classifier beat a single flat 40-class model.
# All models train on full SOC data (clean acquisition-key split).
# Routing at test time uses known metadata — no routing error.
#
# Models trained (15 total):
#   1  flat SOC baseline
#   2  band cascade     (X, KU)
#   4  pol cascade      (HH, HV, VH, VV)
#   4  depression cascade (15, 30, 45, 60 degrees)
#   4  azimuth cascade  (0-90, 90-180, 180-270, 270-360 degrees)
#
# After training, run:
#   python3 scripts/eval_soc_cascade.py
#
# Usage:
#   caffeinate -dim bash scripts/run_soc_cascade.sh 2>&1 | tee logs/soc_cascade.log

set -e

PYTHON=/Users/mousashaya/Projects/defense/venv/bin/python3
CD=/Users/mousashaya/Projects/defense
mkdir -p $CD/logs
cd $CD

COMMON="--model convnext_small --config 40class --setting SOC_40classes \
        --epochs 15 --patience 3 --batch-size 64 --lr 0.0001 --num-workers 4"

echo "=== SOC cascade experiment started $(date) ===" | tee -a logs/soc_cascade.log

# ── Flat baseline ─────────────────────────────────────────────────────────────
echo "" | tee -a logs/soc_cascade.log
echo "--- Flat SOC baseline $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --name "soc_flat_v1" \
    2>&1 | tee -a logs/soc_cascade.log
echo "--- Flat done $(date) ---" | tee -a logs/soc_cascade.log

# ── Band cascade ──────────────────────────────────────────────────────────────
echo "" | tee -a logs/soc_cascade.log
echo "--- Band cascade: X $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --band X \
    --name "soc_band_X_v1" \
    2>&1 | tee -a logs/soc_cascade.log

echo "" | tee -a logs/soc_cascade.log
echo "--- Band cascade: KU $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --band KU \
    --name "soc_band_KU_v1" \
    2>&1 | tee -a logs/soc_cascade.log
echo "--- Band done $(date) ---" | tee -a logs/soc_cascade.log

# ── Pol cascade ───────────────────────────────────────────────────────────────
for POL in HH HV VH VV; do
    echo "" | tee -a logs/soc_cascade.log
    echo "--- Pol cascade: $POL $(date) ---" | tee -a logs/soc_cascade.log
    $PYTHON scripts/train.py $COMMON \
        --polarization $POL \
        --name "soc_pol_${POL}_v1" \
        2>&1 | tee -a logs/soc_cascade.log
    echo "--- Pol $POL done $(date) ---" | tee -a logs/soc_cascade.log
done

# ── Depression angle cascade ──────────────────────────────────────────────────
for ANG in 15 30 45 60; do
    echo "" | tee -a logs/soc_cascade.log
    echo "--- Depression cascade: ${ANG}deg $(date) ---" | tee -a logs/soc_cascade.log
    $PYTHON scripts/train.py $COMMON \
        --angle $ANG \
        --name "soc_dep_${ANG}_v1" \
        2>&1 | tee -a logs/soc_cascade.log
    echo "--- Depression ${ANG}deg done $(date) ---" | tee -a logs/soc_cascade.log
done

# ── Azimuth cascade ───────────────────────────────────────────────────────────
echo "" | tee -a logs/soc_cascade.log
echo "--- Azimuth cascade: 0-90 $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --azimuth-min 0 --azimuth-max 90 \
    --name "soc_az_0_90_v1" \
    2>&1 | tee -a logs/soc_cascade.log

echo "" | tee -a logs/soc_cascade.log
echo "--- Azimuth cascade: 90-180 $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --azimuth-min 90 --azimuth-max 180 \
    --name "soc_az_90_180_v1" \
    2>&1 | tee -a logs/soc_cascade.log

echo "" | tee -a logs/soc_cascade.log
echo "--- Azimuth cascade: 180-270 $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --azimuth-min 180 --azimuth-max 270 \
    --name "soc_az_180_270_v1" \
    2>&1 | tee -a logs/soc_cascade.log

echo "" | tee -a logs/soc_cascade.log
echo "--- Azimuth cascade: 270-360 $(date) ---" | tee -a logs/soc_cascade.log
$PYTHON scripts/train.py $COMMON \
    --azimuth-min 270 --azimuth-max 360 \
    --name "soc_az_270_360_v1" \
    2>&1 | tee -a logs/soc_cascade.log
echo "--- Azimuth done $(date) ---" | tee -a logs/soc_cascade.log

echo "" | tee -a logs/soc_cascade.log
echo "=== All SOC cascade training done $(date) ===" | tee -a logs/soc_cascade.log
echo "Run: python3 scripts/eval_soc_cascade.py" | tee -a logs/soc_cascade.log
