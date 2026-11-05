#!/usr/bin/env python3
"""
Tile all Umbra GEC.tif files into 512x512 PNG patches for MAE backbone pretraining.

Output: /Volumes/Machine Learning/umbra_tiles/
        shard_000000/ ... shard_XXXXXX/  (1000 tiles per shard)
        tile_index.csv  (path, scene, x, y, mean, std)

Quality filter: skip tiles with >50% zero pixels OR std < 5 (uniform/empty)
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

# ── Config ────────────────────────────────────────────────────────────────────
SCENES_DIR  = Path('/Volumes/Machine Learning/umbra_open_data/global')
OUT_DIR     = Path('/Volumes/Machine Learning/umbra_tiles')
TILE_SIZE   = 512
STRIDE      = 512
MAX_ZERO_FRAC = 0.50   # skip tile if > 50% pixels are zero
MIN_STD       = 5.0    # skip tile if std < 5 (flat/uniform)
TILES_PER_SHARD = 1000
RESUME      = True     # skip already-completed scenes

INDEX_PATH  = OUT_DIR / 'tile_index.csv'
DONE_PATH   = OUT_DIR / 'completed_scenes.txt'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load completed scenes for resume
completed = set()
if RESUME and DONE_PATH.exists():
    with open(DONE_PATH) as f:
        completed = {line.strip() for line in f}

# Load existing tile count for shard numbering
existing_tiles = sum(1 for _ in OUT_DIR.glob('**/*.png'))
tile_counter   = existing_tiles
print(f"Resuming from {existing_tiles} existing tiles, {len(completed)} completed scenes")

scenes = sorted(SCENES_DIR.glob('*.tif'))
print(f"Total scenes: {len(scenes)}")
print(f"Remaining:    {len(scenes) - len(completed)}\n")

index_mode = 'a' if INDEX_PATH.exists() else 'w'
index_file = open(INDEX_PATH, index_mode, newline='')
writer = csv.writer(index_file)
if index_mode == 'w':
    writer.writerow(['rel_path', 'scene', 'x_off', 'y_off', 'mean', 'std'])

done_file = open(DONE_PATH, 'a')

t_start = time.time()
n_written = 0
n_skipped = 0

for scene_idx, scene_path in enumerate(scenes):
    scene_name = scene_path.stem
    if scene_name in completed:
        continue

    try:
        with rasterio.open(scene_path) as src:
            W, H = src.width, src.height
            img  = src.read(1)   # uint8 array (H, W)
    except Exception as e:
        print(f"  [ERROR] {scene_name}: {e}")
        continue

    scene_tiles = 0
    xs = range(0, W - TILE_SIZE + 1, STRIDE)
    ys = range(0, H - TILE_SIZE + 1, STRIDE)

    for y in ys:
        for x in xs:
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]

            # Quality filter
            zero_frac = (tile == 0).mean()
            if zero_frac > MAX_ZERO_FRAC:
                n_skipped += 1
                continue
            std = tile.std()
            if std < MIN_STD:
                n_skipped += 1
                continue

            # Shard directory
            shard_id  = tile_counter // TILES_PER_SHARD
            shard_dir = OUT_DIR / f'shard_{shard_id:06d}'
            shard_dir.mkdir(exist_ok=True)

            fname    = f'{scene_name}_{x}_{y}.png'
            out_path = shard_dir / fname
            Image.fromarray(tile).save(out_path, optimize=True)

            rel_path = f'shard_{shard_id:06d}/{fname}'
            writer.writerow([rel_path, scene_name, x, y,
                             f'{tile.mean():.1f}', f'{std:.1f}'])

            tile_counter  += 1
            scene_tiles   += 1
            n_written     += 1

    done_file.write(scene_name + '\n')
    done_file.flush()
    index_file.flush()
    completed.add(scene_name)

    elapsed  = time.time() - t_start
    rate     = (scene_idx + 1) / elapsed * 60
    remaining = (len(scenes) - scene_idx - 1) / (rate / 60) / 3600
    print(f"[{scene_idx+1:4d}/{len(scenes)}] {scene_name[:40]:40s} "
          f"+{scene_tiles:4d} tiles  "
          f"total={tile_counter:,}  "
          f"ETA={remaining:.1f}h", flush=True)

index_file.close()
done_file.close()

elapsed = time.time() - t_start
print(f"\nDone in {elapsed/3600:.1f}h")
print(f"Tiles written: {n_written:,}")
print(f"Tiles skipped: {n_skipped:,}")
print(f"Total tiles:   {tile_counter:,}")
print(f"Index:         {INDEX_PATH}")
