"""
scripts/ingest_atrnet.py
========================
Ingests extracted ATRNet-STAR images into the defense.atrnet_images table.

Reads metadata from the XML file accompanying each .tif image — no guessing,
everything comes straight from the annotation files.

Folder structure:
    images/{setting}/{split}/{target_type}/{band}_{pol}_{dep}_{az}_{id}.tif
    images/{setting}/{split}/{target_type}/{band}_{pol}_{dep}_{az}_{id}.xml

Usage:
    python3 scripts/ingest_atrnet.py              # ingest all settings
    python3 scripts/ingest_atrnet.py --setting SOC_40classes  # one setting only
    python3 scripts/ingest_atrnet.py --dry-run    # count files without inserting
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

IMAGE_DIR = Path('/Volumes/Machine Learning/atrnet/images')


def parse_xml(xml_path: Path) -> dict:
    """Parse one ATRNet annotation XML file and return a metadata dict."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return {}

    def get(path, default=None):
        el = root.find(path)
        return el.text.strip() if el is not None and el.text else default

    # Strip degree symbol from angles
    dep = get('sensor/depression_angle', '')
    az  = get('sensor/target_azimuth_angle', '')
    dep = dep.replace('°', '').strip()
    az  = az.replace('°', '').strip()

    def get_int(path):
        val = get(path)
        try: return int(val) if val else None
        except ValueError: return None

    return {
        'main_class':       get('object/class'),
        'subclass':         get('object/subclass'),
        'target_type':      get('object/type'),
        'scene':            get('scene/scene_name'),
        'band':             get('sensor/band'),
        'polarization':     get('sensor/polarization'),
        'depression_angle': float(dep) if dep else None,
        'azimuth_angle':    float(az)  if az  else None,
        'bbox_xmin':        get_int('object/xmin'),
        'bbox_xmax':        get_int('object/xmax'),
        'bbox_ymin':        get_int('object/ymin'),
        'bbox_ymax':        get_int('object/ymax'),
    }


def ingest_setting(conn, setting_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Walk one setting folder and ingest all .tif images. Returns (new, skipped)."""
    setting_name = setting_dir.name
    new = skipped = 0

    tif_files = list(setting_dir.rglob('*.tif'))
    print(f"  {setting_name}: {len(tif_files):,} images found")

    with conn.cursor() as cur:
        # Bulk-fetch already-ingested paths for this setting to avoid per-row queries
        cur.execute(
            "SELECT local_path FROM defense.atrnet_images WHERE experiment_setting = %s",
            (setting_name,)
        )
        already_done = {row[0] for row in cur.fetchall()}

    batch = []
    for tif_path in tif_files:
        path_str = str(tif_path)
        if path_str in already_done:
            skipped += 1
            continue

        xml_path = tif_path.with_suffix('.xml')
        meta = parse_xml(xml_path) if xml_path.exists() else {}

        # Fall back to folder/filename parsing if XML missing
        parts = tif_path.parts
        split_name   = parts[-3] if len(parts) >= 3 else None  # train/test
        folder_type  = parts[-2] if len(parts) >= 2 else None  # target_type folder

        batch.append((
            path_str,
            meta.get('target_type') or folder_type,
            meta.get('subclass'),
            meta.get('main_class'),
            meta.get('scene'),
            split_name,
            setting_name,
            meta.get('depression_angle'),
            meta.get('azimuth_angle'),
            meta.get('polarization'),
            meta.get('band'),
            meta.get('bbox_xmin'),
            meta.get('bbox_xmax'),
            meta.get('bbox_ymin'),
            meta.get('bbox_ymax'),
        ))

        if len(batch) >= 1000 and not dry_run:
            _flush(conn, batch)
            new += len(batch)
            batch = []

    if batch and not dry_run:
        _flush(conn, batch)
        new += len(batch)
    elif dry_run:
        new += len(batch)

    return new, skipped


def _flush(conn, batch):
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO defense.atrnet_images
                (local_path, target_type, subclass, main_class,
                 scene, split, experiment_setting,
                 depression_angle, azimuth_angle, polarization, band,
                 bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, batch)
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default='',
                        help='Ingest only this setting (e.g. SOC_40classes). Default: all.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count files without inserting into DB.')
    args = parser.parse_args()

    if not IMAGE_DIR.exists():
        print(f"ERROR: {IMAGE_DIR} not found.")
        print("Run the 7z extraction first.")
        sys.exit(1)

    settings = sorted([d for d in IMAGE_DIR.iterdir() if d.is_dir()])
    if args.setting:
        settings = [d for d in settings if d.name == args.setting]
        if not settings:
            print(f"ERROR: setting '{args.setting}' not found in {IMAGE_DIR}")
            sys.exit(1)

    print(f"Settings to ingest: {[d.name for d in settings]}")
    print(f"Dry run: {args.dry_run}\n")

    conn = psycopg2.connect(**DB_CONFIG)

    total_new = total_skipped = 0
    for setting_dir in settings:
        new, skipped = ingest_setting(conn, setting_dir, dry_run=args.dry_run)
        total_new     += new
        total_skipped += skipped
        print(f"    → {new:,} new, {skipped:,} already in DB")

    if conn:
        conn.close()

    print(f"\nDone: {total_new:,} new images ingested, {total_skipped:,} skipped")


if __name__ == '__main__':
    main()
