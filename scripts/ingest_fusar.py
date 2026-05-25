"""
scripts/ingest_fusar.py
========================
Ingest FUSAR-Ship 1.0 into defense.fusar_images.

Dataset structure:
  {Category}/{Subcategory}/{image_id}.tiff
  meta.csv  -- id, mmsi, length, width, polarMode, centerLookAngle, heightspace, widthspace, path

Images are already 512x512 uint8 preprocessed SAR chips — no cropping needed.
Train/test split is 80/20 by random seed for reproducibility.

Usage:
    python3 scripts/ingest_fusar.py
    python3 scripts/ingest_fusar.py --dry-run
    python3 scripts/ingest_fusar.py --root "/Volumes/Machine Learning/fusar_ship"
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

FUSAR_ROOT = Path('/Volumes/Machine Learning/fusar_ship')
TRAIN_FRAC = 0.8
RANDOM_SEED = 42

SKIP_DIRS = {'SAR', 'Reserved', 'Unspecified', 'WingInGrnd'}


def load_meta(root: Path) -> dict:
    meta_path = root / 'meta.csv'
    if not meta_path.exists():
        print("  WARNING: meta.csv not found — metadata will be empty")
        return {}
    meta = {}
    with open(meta_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            iid = row['id'].strip()
            try:
                length = float(row['length']) if row.get('length') else None
            except ValueError:
                length = None
            try:
                width = float(row['width']) if row.get('width') else None
            except ValueError:
                width = None
            try:
                angle = float(row['centerLookAngle']) if row.get('centerLookAngle') else None
            except ValueError:
                angle = None
            try:
                hres = float(row['heightspace']) if row.get('heightspace') else None
            except ValueError:
                hres = None
            try:
                wres = float(row['widthspace']) if row.get('widthspace') else None
            except ValueError:
                wres = None
            meta[iid] = {
                'mmsi':         row.get('mmsi', '').strip() or None,
                'ship_length_m': length,
                'ship_width_m':  width,
                'polar_mode':   row.get('polarMode', '').strip() or None,
                'look_angle':   angle,
                'height_res':   hres,
                'width_res':    wres,
            }
    print(f"  meta.csv: {len(meta):,} entries")
    return meta


def collect_images(root: Path) -> list:
    images = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in SKIP_DIRS:
            continue
        category = cat_dir.name
        for sub_dir in sorted(cat_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            subcategory = sub_dir.name
            for tiff in sorted(sub_dir.glob('*.tiff')):
                images.append((tiff, category, subcategory))
    return images


def assign_splits(images: list) -> dict:
    rng = random.Random(RANDOM_SEED)
    by_subcat = {}
    for tiff, cat, sub in images:
        key = (cat, sub)
        by_subcat.setdefault(key, []).append((tiff, cat, sub))
    splits = {}
    for key, group in by_subcat.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * TRAIN_FRAC))
        for i, (tiff, cat, sub) in enumerate(shuffled):
            splits[tiff] = 'train' if i < n_train else 'test'
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=FUSAR_ROOT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    root = args.root
    print(f"Root: {root}")

    meta = load_meta(root)
    images = collect_images(root)
    print(f"  Found {len(images):,} images across categories:")

    cat_counts = {}
    for _, cat, sub in images:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, n in sorted(cat_counts.items()):
        print(f"    {cat:20} {n:>5}")

    splits = assign_splits(images)
    n_train = sum(1 for v in splits.values() if v == 'train')
    n_test  = sum(1 for v in splits.values() if v == 'test')
    print(f"\n  Split: {n_train:,} train / {n_test:,} test")

    if args.dry_run:
        print("\nDRY RUN — no DB writes")
        return

    conn = psycopg2.connect(**DB_CONFIG)
    inserted = skipped = errors = 0

    with conn.cursor() as cur:
        for tiff, category, subcategory in images:
            image_id = tiff.stem
            m = meta.get(image_id, {})
            split = splits.get(tiff, 'train')
            try:
                cur.execute("""
                    INSERT INTO defense.fusar_images
                        (image_id, local_path, category, subcategory,
                         mmsi, ship_length_m, ship_width_m, polar_mode,
                         look_angle, height_res, width_res, split)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (image_id) DO NOTHING
                """, (
                    image_id, str(tiff), category, subcategory,
                    m.get('mmsi'), m.get('ship_length_m'), m.get('ship_width_m'),
                    m.get('polar_mode'), m.get('look_angle'),
                    m.get('height_res'), m.get('width_res'), split,
                ))
                if cur.rowcount:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Error on {image_id}: {e}")
                conn.rollback()
                errors += 1

    conn.commit()
    conn.close()
    print(f"\n  inserted={inserted:,}  skipped={skipped:,}  errors={errors:,}")
    print("\nDone. Check with:")
    print("  SELECT category, COUNT(*) FROM defense.fusar_images GROUP BY category ORDER BY COUNT(*) DESC;")


if __name__ == '__main__':
    main()
