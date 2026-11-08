#!/bin/bash
# Phase 3: all of 2025 (Sentinel-1A + 1C) for 9 regions, month by month: download -> chip (>=25 m) -> delete scenes.
# Output kept separate from the paper's 2024 data: sentinel1_<r>_2025 / sar_ship_chips_<r>_2025.
cd /Users/mousashaya/Projects/defense
PY=venv/bin/python
P2=$1     # phase-2 pid to wait for
while [ -n "$P2" ] && kill -0 $P2 2>/dev/null; do sleep 300; done
echo "=== start 2025 AIS $(date)"
$PY -u scripts/build_ais_new_regions.py --start 2025-01-01 --end 2025-12-31 \
    --regions NewEngland GulfEast Oregon Hawaii GulfOpen Puget Chesapeake LA NY > logs/bolster_ais_2025.log 2>&1 &
AIS=$!
ENDS="01-31 02-28 03-31 04-30 05-31 06-30 07-31 08-31 09-30 10-31 11-30 12-31"
for R in NewEngland GulfEast Oregon Hawaii Puget Chesapeake LA NY GulfOpen; do
  r=$(echo $R | tr A-Z a-z)
  AISDIR=$($PY -c "import sys;sys.path.insert(0,'scripts');from chip_ships_regions import REGIONS;print(REGIONS['$R']['ais_dir'])")
  for E in $ENDS; do
    M=${E%-*}
    echo "=== $R 2025-$M download $(date)"
    $PY -u scripts/download_new_regions.py --region $R --start 2025-$M-01 --end 2025-$E --out-dir sentinel1_${r}_2025 \
        >> logs/bolster2025_download_$R.log 2>&1 || { echo "download failed $R $M"; continue; }
    # month's AIS must exist before chipping (scenes without AIS would be deleted with 0 chips)
    while kill -0 $AIS 2>/dev/null && [ ! -f "/Volumes/Machine Learning/$AISDIR/ais-2025-$E.csv.zst" ]; do sleep 120; done
    $PY -u scripts/chip_ships_regions.py --region $R --min-length 25 --backbone-patches 0 --delete-scenes \
        --scene-dir sentinel1_${r}_2025 --out-dir sar_ship_chips_${r}_2025 >> logs/bolster2025_chip_$R.log 2>&1
  done
  echo "=== $R 2025 done $(date): $(find "/Volumes/Machine Learning/sar_ship_chips_${r}_2025" -name '*.npz' | wc -l) chips"
done
echo "=== 2025 ALL DONE $(date)"
