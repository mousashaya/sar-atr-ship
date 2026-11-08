#!/bin/bash
# 2026-11 data expansion: AIS (Feb-Aug 2024, local national files) -> S1 GRD download -> chip (>=25 m) -> delete SAFE.
# Usage: bash scripts/run_bolster_regions.sh   (logs in logs/bolster_*.log)
cd /Users/mousashaya/Projects/defense
PY=venv/bin/python
AIS_PID=$(pgrep -f build_ais_new_regions | head -1)   # already running; started separately

for R in NewEngland GulfEast Oregon Hawaii; do
  echo "=== $R download $(date)"
  $PY -u scripts/download_new_regions.py --region $R --start 2024-02-01 --end 2024-08-31 >> logs/bolster_download_$R.log 2>&1 || { echo "download failed for $R - skipping chip"; continue; }
  while [ -n "$AIS_PID" ] && kill -0 $AIS_PID 2>/dev/null; do sleep 60; done   # AIS must finish before chipping
  echo "=== $R chip $(date)"
  $PY -u scripts/chip_ships_regions.py --region $R --min-length 25 --backbone-patches 0 --delete-scenes >> logs/bolster_chip_$R.log 2>&1
  echo "=== $R done $(date): $(find "/Volumes/Machine Learning/sar_ship_chips_$(echo $R | tr A-Z a-z)" -name '*.npz' | wc -l) chips"
done
echo "=== ALL DONE $(date)"
