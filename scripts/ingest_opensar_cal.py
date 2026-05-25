"""
scripts/ingest_opensar_cal.py
==============================
Re-ingest OpenSARShip 2.0 using Patch_Cal (calibrated float32 chips).

Previous ingest used Patch_RGB — full 875×656 scenes, not ship chips.
This script:
  1. Scans all Patch_Cal/ directories under OpenSARShip2/
  2. Reads each 2-channel (VV, VH) float32 TIFF using tifffile
  3. Converts each channel to 8-bit grayscale (dB normalization) and saves as PNG
  4. Assigns scene-level train/val/test splits (70/15/15) — no scene leakage
  5. Truncates defense.opensar_images and inserts fresh records

Output chips saved to:
  /Volumes/Machine Learning/opensar_ship/chips_cal/{scene_id}/{stem}_{pol}.png

Usage:
  venv/bin/python3 scripts/ingest_opensar_cal.py
  venv/bin/python3 scripts/ingest_opensar_cal.py --dry-run
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed.  pip install tifffile")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG
from notify_email import send_email

OPENSAR_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')
CHIPS_DIR    = Path('/Volumes/Machine Learning/opensar_ship/chips_cal')
SEED         = 42

# dB normalization range — clip SAR amplitude to this window before 0-255 mapping
# 10*log10(amplitude): typical ship values -25 to +5 dB
DB_MIN = -30.0
DB_MAX =  10.0


def to_uint8(arr_2d: np.ndarray) -> np.ndarray:
    """Convert float32 amplitude to 8-bit via dB normalization."""
    db = 10.0 * np.log10(np.maximum(arr_2d, 1e-6))
    db = np.clip(db, DB_MIN, DB_MAX)
    scaled = (db - DB_MIN) / (DB_MAX - DB_MIN) * 255.0
    return scaled.astype(np.uint8)


def scene_split(scene_names: list, seed: int = SEED):
    """Assign train/val/test at scene level (70/15/15)."""
    rng   = np.random.default_rng(seed)
    names = sorted(scene_names)
    rng.shuffle(names)
    n     = len(names)
    n_val  = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    splits = {}
    for s in names[:n_val]:          splits[s] = 'val'
    for s in names[n_val:n_val+n_test]: splits[s] = 'test'
    for s in names[n_val+n_test:]:   splits[s] = 'train'
    return splits


def polarization_from_scene(scene_name: str) -> str:
    """Infer dominant polarization from Sentinel-1 scene name."""
    s = scene_name.upper()
    if '1SDV' in s or '1SDH' in s:
        return 'dual'    # VV+VH or HH+HV
    if '1SSV' in s:      return 'VV'
    if '1SSH' in s:      return 'HH'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show counts without writing files or DB rows')
    args = parser.parse_args()

    # ── Discover Patch_Cal directories ────────────────────────────────────────
    cal_dirs = sorted(OPENSAR_ROOT.glob('*/Patch_Cal'))
    if not cal_dirs:
        print(f"ERROR: No Patch_Cal directories found under {OPENSAR_ROOT}")
        sys.exit(1)

    scene_names = [d.parent.name for d in cal_dirs]
    splits = scene_split(scene_names)
    split_counts = defaultdict(int)
    for s in splits.values(): split_counts[s] += 1
    print(f"Scenes: {len(cal_dirs)}  "
          f"(train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']})")

    # ── Count chips ───────────────────────────────────────────────────────────
    all_chips = []
    for cal_dir in cal_dirs:
        scene = cal_dir.parent.name
        for tif in sorted(cal_dir.glob('*.tif')):
            all_chips.append((scene, tif))
    print(f"Total chips: {len(all_chips):,}  "
          f"→ {len(all_chips)*2:,} records (VV + VH each)")

    if args.dry_run:
        # Show type breakdown
        type_counts = defaultdict(int)
        for scene, tif in all_chips:
            ship_type = tif.stem.split('_x')[0]
            type_counts[ship_type] += 1
        print("\nPer type:")
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:30s}: {n:,} chips → {n*2:,} records")
        print("\nDry run done.")
        return

    send_email(
        "Ship re-ingest started — Patch_Cal",
        f"Starting OpenSARShip re-ingest from Patch_Cal.\n\n"
        f"  Scenes: {len(cal_dirs)}\n"
        f"  Chips:  {len(all_chips):,} (×2 for VV+VH = {len(all_chips)*2:,} records)\n"
        f"  Split:  scene-level 70/15/15 (no leakage)\n\n"
        f"Will convert float32 → 8-bit dB PNG and insert into DB."
    )

    # ── Create output directory ───────────────────────────────────────────────
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Truncate existing data ─────────────────────────────────────────────────
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE defense.opensar_images RESTART IDENTITY")
    conn.commit()
    print("Truncated defense.opensar_images")

    # ── Process chips and insert ──────────────────────────────────────────────
    inserted = 0
    errors   = 0

    for chip_idx, (scene, tif_path) in enumerate(all_chips):
        # Parse ship type from filename: "Cargo_x10712_y8238.tif"
        stem      = tif_path.stem
        ship_type = stem.split('_x')[0]

        scene_chips_dir = CHIPS_DIR / scene
        scene_chips_dir.mkdir(exist_ok=True)

        # Read 2-channel float32 tiff
        try:
            arr = tifffile.imread(str(tif_path))  # (H, W, 2) float32
        except Exception as e:
            print(f"  ERROR reading {tif_path.name}: {e}")
            errors += 1
            continue

        if arr.ndim == 2:
            # Some scenes may be single-channel
            arr = arr[:, :, np.newaxis]
            arr = np.concatenate([arr, arr], axis=2)

        split    = splits[scene]
        scene_pol = polarization_from_scene(scene)

        # Save VV (channel 0) and VH (channel 1) as separate PNGs
        pols = [('VV', 0), ('VH', 1)]
        for pol_name, ch_idx in pols:
            if ch_idx >= arr.shape[2]:
                continue

            channel  = arr[:, :, ch_idx]
            uint8    = to_uint8(channel)
            out_path = scene_chips_dir / f'{stem}_{pol_name}.png'

            Image.fromarray(uint8, mode='L').save(str(out_path))

            image_id = f'{scene}__{stem}_{pol_name}'

            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO defense.opensar_images
                            (image_id, local_path, scene_id, ship_type,
                             polarization, image_width, image_height, split)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (image_id) DO NOTHING
                    """, (
                        image_id,
                        str(out_path),
                        scene,
                        ship_type,
                        pol_name,
                        uint8.shape[1],
                        uint8.shape[0],
                        split,
                    ))
                inserted += cur.rowcount
            except Exception as e:
                print(f"  DB error on {image_id}: {e}")
                conn.rollback()
                errors += 1
                continue

        conn.commit()

        if (chip_idx + 1) % 500 == 0:
            print(f"  {chip_idx+1:,}/{len(all_chips):,} chips processed "
                  f"({inserted:,} records inserted)")

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone.")
    print(f"  Inserted: {inserted:,} records")
    print(f"  Errors:   {errors}")

    # Verify split distribution
    conn2 = psycopg2.connect(**DB_CONFIG)
    with conn2.cursor() as cur:
        cur.execute("""
            SELECT split, ship_type, COUNT(*) as n
            FROM defense.opensar_images
            GROUP BY split, ship_type
            ORDER BY split, ship_type
        """)
        rows = cur.fetchall()
    conn2.close()

    summary_lines = ["Split / type distribution:"]
    for split_name, ship_type, n in rows:
        summary_lines.append(f"  {split_name:6s}  {ship_type:25s}  {n:,}")
    summary = '\n'.join(summary_lines)
    print('\n' + summary)

    send_email(
        f"Ship re-ingest DONE — {inserted:,} records",
        f"OpenSARShip Patch_Cal ingest complete.\n\n"
        f"  Inserted: {inserted:,} records (VV + VH)\n"
        f"  Errors:   {errors}\n"
        f"  Chips dir: {CHIPS_DIR}\n\n"
        f"{summary}\n\n"
        f"Next: run train_ship_soc.py to retrain baseline."
    )


if __name__ == '__main__':
    main()
