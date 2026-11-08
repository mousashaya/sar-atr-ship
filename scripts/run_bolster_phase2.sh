#!/bin/bash
# Phase 2: Jan + Sep-Dec 2024 scenes for the 4 new regions. Waits for phase 1 and the AIS fill-in to finish.
cd /Users/mousashaya/Projects/defense
PY=venv/bin/python
while kill -0 64477 2>/dev/null || pgrep -f build_ais_new_regions >/dev/null; do sleep 120; done
for R in NewEngland GulfEast Oregon Hawaii; do
  for RANGE in "2024-01-01 2024-01-31" "2024-09-01 2024-12-31"; do
    set -- $RANGE
    echo "=== $R download $1..$2 $(date)"
    $PY -u scripts/download_new_regions.py --region $R --start $1 --end $2 >> logs/bolster_download_$R.log 2>&1 || { echo "download failed $R $1"; continue; }
    echo "=== $R chip $(date)"
    $PY -u scripts/chip_ships_regions.py --region $R --min-length 25 --backbone-patches 0 --delete-scenes >> logs/bolster_chip_$R.log 2>&1
  done
  echo "=== $R done $(date): $(find "/Volumes/Machine Learning/sar_ship_chips_$(echo $R | tr A-Z a-z)" -name '*.npz' | wc -l) chips"
done
echo "=== PHASE 2 DONE $(date)"
