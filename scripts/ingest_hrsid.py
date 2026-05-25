"""
scripts/ingest_hrsid.py
========================
Ingest HRSID into defense.hrsid_chips.

For each COCO annotation (bounding box) in HRSID train/test splits, this script
crops the ship chip from the 800x800 scene image, saves it to the chips dir,
and inserts metadata into the DB.

HRSID has only one category ('ship') — no fine-grained type labels. The value
here is high-res SAR ship chips (0.5–3m) for detection, UMAP, and future
fusion with FUSAR-Ship labels.

Usage:
    python3 scripts/ingest_hrsid.py
    python3 scripts/ingest_hrsid.py --dry-run
    python3 scripts/ingest_hrsid.py --pad 24 --chips-dir "/Volumes/Machine Learning/hrsid/chips"
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

HRSID_ROOT  = Path('/Volumes/Machine Learning/hrsid/HRSID_JPG')
CHIPS_DIR   = Path('/Volumes/Machine Learning/hrsid/chips')
IMAGES_DIR  = HRSID_ROOT / 'JPEGImages'
ANNS_DIR    = HRSID_ROOT / 'annotations'
INSHORE_DIR = HRSID_ROOT / 'inshore_offshore'

DEFAULT_PAD = 16  # pixels added around each bbox


def load_inshore_set() -> set:
    """Return set of image_ids labeled as inshore."""
    inshore_path = INSHORE_DIR / 'inshore.json'
    if not inshore_path.exists():
        return set()
    with open(inshore_path) as f:
        d = json.load(f)
    return {img['id'] for img in d.get('images', [])}


def ingest_split(split: str, ann_path: Path, inshore_ids: set,
                 pad: int, chips_dir: Path, dry_run: bool, conn):
    with open(ann_path) as f:
        coco = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    annotations = coco['annotations']
    print(f"\n[{split}] {len(annotations):,} annotations across {len(coco['images']):,} images")

    split_dir = chips_dir / split
    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    inserted = skipped = errors = 0

    for ann in annotations:
        ann_id    = ann['id']
        image_id  = ann['image_id']
        filename  = id_to_filename.get(image_id)
        if not filename:
            continue

        bx, by, bw, bh = [int(v) for v in ann['bbox']]
        area      = ann.get('area', bw * bh)
        scene     = 'inshore' if image_id in inshore_ids else 'offshore'
        chip_id   = f"{split}_{ann_id}"
        chip_path = split_dir / f"{chip_id}.jpg"

        if dry_run:
            inserted += 1
            continue

        # Crop chip with padding
        src_path = IMAGES_DIR / filename
        try:
            with Image.open(src_path) as im:
                W, H = im.size
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(W, bx + bw + pad)
                y1 = min(H, by + bh + pad)
                chip = im.crop((x0, y0, x1, y1))
                chip_w, chip_h = chip.size
                chip.save(chip_path, quality=95)
        except Exception as e:
            print(f"  Error cropping {filename} ann {ann_id}: {e}")
            errors += 1
            continue

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO defense.hrsid_chips
                        (chip_id, annotation_id, image_id, source_filename,
                         local_path, bbox_x, bbox_y, bbox_w, bbox_h,
                         area, scene, split, chip_width, chip_height)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chip_id) DO NOTHING
                """, (
                    chip_id, ann_id, image_id, filename,
                    str(chip_path), bx, by, bw, bh,
                    area, scene, split, chip_w, chip_h,
                ))
                if cur.rowcount:
                    inserted += 1
                else:
                    skipped += 1
            conn.commit()
        except Exception as e:
            print(f"  DB error on {chip_id}: {e}")
            conn.rollback()
            errors += 1

    print(f"  inserted={inserted:,}  skipped={skipped:,}  errors={errors:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pad', type=int, default=DEFAULT_PAD,
                        help='Pixels of context padding around each bbox')
    parser.add_argument('--chips-dir', type=Path, default=CHIPS_DIR)
    args = parser.parse_args()

    inshore_ids = load_inshore_set()
    print(f"Inshore image count: {len(inshore_ids):,}")

    splits = [
        ('train', ANNS_DIR / 'train2017.json'),
        ('test',  ANNS_DIR / 'test2017.json'),
    ]

    conn = None if args.dry_run else psycopg2.connect(**DB_CONFIG)

    for split, ann_path in splits:
        ingest_split(split, ann_path, inshore_ids, args.pad,
                     args.chips_dir, args.dry_run, conn)

    if conn:
        conn.close()

    print("\nDone.")
    if not args.dry_run:
        print("Check with:")
        print("  SELECT scene, split, COUNT(*) FROM defense.hrsid_chips GROUP BY scene, split ORDER BY split, scene;")


if __name__ == '__main__':
    main()
