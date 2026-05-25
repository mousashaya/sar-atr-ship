#!/bin/bash
# Gulf pipeline: wait for unzip → chip → embed → umap

VENV="$(dirname "$0")/venv/bin/python"
SCRIPTS="$(dirname "$0")/scripts"
LOGS="$(dirname "$0")/logs"
GULF_DIR="/Volumes/Machine Learning/sentinel1_gulf"

echo "=== Gulf pipeline started $(date) ==="

# Step 1: wait for unzip to finish
echo "Waiting for unzip to finish..."
until ! pgrep -f "unzip.*sentinel1_gulf" > /dev/null; do
    done=$(find "$GULF_DIR" -maxdepth 1 -name "*.SAFE" -type d | wc -l | tr -d ' ')
    echo "  $(date +%H:%M:%S) — ${done} SAFE dirs unzipped so far..."
    sleep 30
done
echo "Unzip complete at $(date)"

# Step 2: chip
echo ""
echo "=== Chipping Gulf scenes $(date) ==="
"$VENV" "$SCRIPTS/chip_ships_gulf.py" 2>&1 | tee "$LOGS/chip_gulf_pipeline.log"
echo "Chipping done at $(date)"

# Step 3: embed
echo ""
echo "=== Embedding with DINOv2 $(date) ==="
"$VENV" "$SCRIPTS/embed_ships_dinov2.py" 2>&1 | tee "$LOGS/embed_gulf_pipeline.log"
echo "Embedding done at $(date)"

# Step 4: umap
echo ""
echo "=== Running UMAP $(date) ==="
"$VENV" "$SCRIPTS/umap_ship_reid.py" 2>&1 | tee "$LOGS/umap_gulf_pipeline.log"
echo "UMAP done at $(date)"

# Step 5: train metric learning model
echo ""
echo "=== Training triplet loss re-ID model $(date) ==="
"$VENV" "$SCRIPTS/train_ship_reid.py" --epochs 20 2>&1 | tee "$LOGS/train_reid_pipeline.log"
echo "Training done at $(date)"

# Step 6: refresh app index
echo ""
echo "=== Refreshing app ship index $(date) ==="
curl -s http://localhost:5051/api/sar/refresh && echo "" || echo "App not running — start run_viewer.py to see results"

echo ""
echo "=== Pipeline complete $(date) ==="
