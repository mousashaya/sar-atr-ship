"""
scripts/ingest_atrnet.py
========================
Downloads ATRNet-STAR from HuggingFace and loads image metadata into the DB.

ATRNet-STAR: 194K SAR images, 128x128, 40 vehicle types / 21 subclasses / 4 main classes
Source: https://huggingface.co/datasets/waterdisappear/ATRNet-STAR

Usage:
    # First time — download and ingest all images:
    python3 scripts/ingest_atrnet.py

    # Re-run safely — skips already-ingested images.

What it does:
    1. Downloads the HuggingFace dataset (saved to data/atrnet/images/)
    2. Saves each image as a PNG chip (128x128)
    3. Parses the filename for depression_angle, azimuth_angle, band, polarization
    4. Inserts a row into defense.atrnet_images for each image

File naming convention (from ATRNet-STAR paper):
    {band}_{polarization}_{depression_angle}_{azimuth_angle}_{id}.png
    e.g. C_HH_17_000_00001.png
"""

import sys
import re
from pathlib import Path
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

DATA_DIR = Path(__file__).parent.parent / 'data' / 'atrnet'
IMAGE_DIR = DATA_DIR / 'images'

# ATRNet hierarchy — maps dataset folder names to (main_class, subclass)
# Fill in from the ATRNet-STAR paper Table 1 once you have the actual folder names.
# This is the approximate mapping based on the paper description.
HIERARCHY = {
    # main_class → subclass → [target_types]
    # To be populated after inspecting the downloaded folder structure.
}


def parse_filename(name: str) -> dict:
    """
    Extract band, polarization, depression angle, azimuth angle from filename.
    Format: {band}_{polarization}_{depression}_{azimuth}_{id}.png
    """
    stem = Path(name).stem
    parts = stem.split('_')
    result = {'band': None, 'polarization': None,
              'depression_angle': None, 'azimuth_angle': None}
    if len(parts) >= 4:
        result['band']             = parts[0]
        result['polarization']     = parts[1]
        try: result['depression_angle'] = float(parts[2])
        except ValueError: pass
        try: result['azimuth_angle']    = float(parts[3])
        except ValueError: pass
    return result


def ingest(conn, setting: str, split: str, target_type: str,
           subclass: str, main_class: str, image_path: Path) -> bool:
    """Insert one image into the DB. Returns False if already exists."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM defense.atrnet_images WHERE local_path = %s",
            (str(image_path),)
        )
        if cur.fetchone():
            return False  # already ingested

        meta = parse_filename(image_path.name)
        cur.execute("""
            INSERT INTO defense.atrnet_images
                (local_path, target_type, subclass, main_class,
                 scene, split, experiment_setting,
                 depression_angle, azimuth_angle, polarization, band)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(image_path),
            target_type, subclass, main_class,
            None,        # scene — parsed from path structure if present
            split, setting,
            meta['depression_angle'], meta['azimuth_angle'],
            meta['polarization'], meta['band'],
        ))
    return True


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: install huggingface datasets first:")
        print("  venv/bin/pip install datasets pillow")
        sys.exit(1)

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ATRNet-STAR from HuggingFace (this may take a while)...")
    print("Dataset size: ~194K images")
    print()

    # Load the dataset — HuggingFace will cache it locally
    ds = load_dataset('waterdisappear/ATRNet-STAR', trust_remote_code=True)

    print(f"Dataset splits available: {list(ds.keys())}")
    print(f"Features: {ds[list(ds.keys())[0]].features}")
    print()

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    total_new = 0
    total_skip = 0

    for split_name, split_ds in ds.items():
        print(f"Processing split: {split_name} ({len(split_ds)} images)")

        for i, record in enumerate(split_ds):
            # ATRNet records likely have: image, label, target_type, subclass, etc.
            # Adjust field names after inspecting ds.features
            image     = record.get('image') or record.get('img')
            label     = record.get('label') or record.get('target_type', 'unknown')
            subclass  = record.get('subclass', '')
            main_cls  = record.get('main_class', '')
            setting   = record.get('setting', 'default')
            filename  = record.get('filename') or record.get('file_name', f'{i:06d}.png')

            # Save image to disk
            out_path = IMAGE_DIR / split_name / str(label) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if not out_path.exists():
                image.save(str(out_path))

            added = ingest(conn, setting, split_name, label, subclass, main_cls, out_path)
            if added:
                total_new += 1
            else:
                total_skip += 1

            if (i + 1) % 5000 == 0:
                conn.commit()
                print(f"  {i+1:,}/{len(split_ds):,} processed  "
                      f"({total_new:,} new, {total_skip:,} skipped)")

        conn.commit()
        print(f"  Split done: {len(split_ds):,} images")

    conn.close()
    print(f"\nIngestion complete: {total_new:,} new images, {total_skip:,} already existed")
    print(f"Images saved to: {IMAGE_DIR}")


if __name__ == '__main__':
    main()
