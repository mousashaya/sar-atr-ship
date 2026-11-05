"""
scripts/extract_slc_patches.py
================================
Pre-extract SAR patches from SLC SAFE scenes to the internal SSD as uint8 numpy files.
Eliminates per-batch TIFF reads during training — replaces slow random I/O with
fast sequential reads from internal SSD.

Input:  /Volumes/Machine Learning/sentinel1_slc_gulf/ (208 SAFE scenes)
Output: /Users/mousashaya/MLData/sar_patches_slc_gulf/patches.npy  (uint8, ~47GB)
        /Users/mousashaya/MLData/sar_patches_slc_gulf/meta.json    (patch count, scene list)

Each patch: (3, 224, 224) uint8
  channel 0: VV amplitude in dB, percentile-normalized to [0, 255]
  channel 1: VH amplitude in dB, percentile-normalized to [0, 255]
  channel 2: VV-VH ratio in dB, percentile-normalized to [0, 255]

Usage:
    venv/bin/python scripts/extract_slc_patches.py
    venv/bin/python scripts/extract_slc_patches.py --patches-per-swath 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows

SCENE_DIR        = Path('/Volumes/Machine Learning/sentinel1_slc_gulf')
OUT_DIR          = Path('/Users/mousashaya/MLData/sar_patches_slc_gulf')
PATCH_SIZE       = 224
SEED             = 42


def find_subswath_pairs(scene_dir):
    pairs = []
    for safe in sorted(scene_dir.glob('*.SAFE')):
        meas = safe / 'measurement'
        for sw in ['iw1', 'iw2', 'iw3']:
            vv_files = sorted(meas.glob(f'*-{sw}-slc-vv-*.tiff'))
            vh_files = sorted(meas.glob(f'*-{sw}-slc-vh-*.tiff'))
            if vv_files and vh_files:
                pairs.append((vv_files[0], vh_files[0]))
    return pairs


def norm_channel(x):
    valid = x[x > 0]
    if valid.size == 0:
        return np.zeros(x.shape, dtype=np.uint8)
    p2, p98 = np.percentile(valid, [2, 98])
    clipped = np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1)
    return (clipped * 255).astype(np.uint8)


def patch_to_uint8(vv_raw, vh_raw):
    vv_db = 10 * np.log10(np.abs(vv_raw).astype(np.float32) + 1.0)
    vh_db = 10 * np.log10(np.abs(vh_raw).astype(np.float32) + 1.0)
    return np.stack([
        norm_channel(vv_db),
        norm_channel(vh_db),
        norm_channel(vv_db - vh_db),
    ], axis=0)  # (3, 224, 224) uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches-per-swath', type=int, default=500)
    args = parser.parse_args()

    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = find_subswath_pairs(SCENE_DIR)
    print(f"Found {len(pairs)} subswath pairs ({len(pairs)//3} scenes × 3 subswaths)")

    total_patches = len(pairs) * args.patches_per_swath
    size_gb = total_patches * 3 * PATCH_SIZE * PATCH_SIZE / 1e9
    print(f"Target: {total_patches:,} patches  (~{size_gb:.1f} GB as uint8)")
    print(f"Output: {OUT_DIR}/patches.npy")
    print(f"Extracting...")

    out_path = OUT_DIR / 'patches.npy'
    fp = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.uint8,
        shape=(total_patches, 3, PATCH_SIZE, PATCH_SIZE)
    )

    idx = 0
    t0 = time.time()
    skipped = 0

    for i, (vv_path, vh_path) in enumerate(pairs):
        try:
            with rasterio.open(vv_path) as src:
                h, w = src.height, src.width

            margin = PATCH_SIZE
            if h < 2 * margin + PATCH_SIZE or w < 2 * margin + PATCH_SIZE:
                skipped += args.patches_per_swath
                continue

            rows = np.random.randint(margin, h - margin - PATCH_SIZE, size=args.patches_per_swath * 2)
            cols = np.random.randint(margin, w - margin - PATCH_SIZE, size=args.patches_per_swath * 2)

            kept = 0
            for r, c in zip(rows, cols):
                if kept >= args.patches_per_swath:
                    break
                window = rasterio.windows.Window(int(c), int(r), PATCH_SIZE, PATCH_SIZE)
                try:
                    with rasterio.open(vv_path) as src:
                        vv = src.read(1, window=window)
                    with rasterio.open(vh_path) as src:
                        vh = src.read(1, window=window)
                    fp[idx] = patch_to_uint8(vv, vh)
                    idx += 1
                    kept += 1
                except Exception:
                    continue

        except Exception as e:
            print(f"  Warning: {vv_path.parent.parent.name}: {e}")
            skipped += args.patches_per_swath
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            remaining = (total_patches - idx) / max(rate, 1)
            print(f"  [{i+1}/{len(pairs)}]  {idx:,} patches  "
                  f"{elapsed/60:.0f}m elapsed  ~{remaining/60:.0f}m remaining")

    actual_patches = idx
    fp.flush()
    del fp

    if actual_patches < total_patches:
        print(f"  Trimming to {actual_patches:,} actual patches (skipped {skipped})")
        full = np.load(out_path, mmap_mode='r')
        np.save(out_path, full[:actual_patches])

    meta = {
        'n_patches': actual_patches,
        'patch_size': PATCH_SIZE,
        'channels': ['vv_db_norm', 'vh_db_norm', 'vv_vh_diff_norm'],
        'dtype': 'uint8',
        'patches_per_swath': args.patches_per_swath,
        'n_pairs': len(pairs),
        'scene_dir': str(SCENE_DIR),
    }
    with open(OUT_DIR / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    size_gb = (out_path.stat().st_size / 1e9)
    print(f"\nDone: {actual_patches:,} patches  {size_gb:.1f} GB  ({elapsed/60:.0f} min)")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
