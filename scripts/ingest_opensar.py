"""
scripts/ingest_opensar.py
==========================
Ingest OpenSARShip 2.0 into defense.opensar_images.

OpenSARShip 2.0 structure (inside the zip):
  OpenSARShip2/
    {scene_id}.rar          -- one RAR per Sentinel-1 scene, contains chip PNGs
    openSARship.xlsx        -- label spreadsheet with AIS metadata

Label spreadsheet columns (from paper):
  Ship ID | SAR scene ID | Ship length | Ship width | Ship type (AIS) | ...

Steps:
  1. Unzip OpenSARShip_2.zip  →  OpenSARShip2/ folder
  2. Extract each .rar        →  chip images
  3. Parse openSARship.xlsx   →  label dict keyed by image filename
  4. Insert into DB

Requirements:
  pip install rarfile openpyxl   (rarfile also needs 'unrar' CLI)
  brew install unar              (handles rar extraction on macOS)

Usage:
  python3 scripts/ingest_opensar.py --zip-path "/Volumes/Machine Learning/opensar_ship/OpenSARShip_2.zip"
  python3 scripts/ingest_opensar.py --zip-path ... --dry-run
  python3 scripts/ingest_opensar.py --already-extracted "/Volumes/Machine Learning/opensar_ship/OpenSARShip2"
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

DEST_DIR = Path('/Volumes/Machine Learning/opensar_ship')

# AIS MMSI type codes → human-readable labels used in OpenSARShip
# (the dataset uses these 6 consolidated classes)
KNOWN_TYPES = {'Cargo', 'Tanker', 'Fishing', 'Dredging', 'Passenger', 'Tug'}


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, dest: Path):
    print(f"Extracting zip → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    print("  Zip extracted")


def extract_rars(root_dir: Path):
    """Extract all .rar files found under root_dir using unar (macOS) or unrar."""
    rar_files = list(root_dir.rglob('*.rar'))
    if not rar_files:
        print("  No .rar files found — skipping RAR extraction")
        return
    print(f"  Extracting {len(rar_files)} RAR archives...")
    for i, rar in enumerate(rar_files, 1):
        out_dir = rar.parent / rar.stem
        out_dir.mkdir(exist_ok=True)
        # Try unar first (macOS Homebrew), fall back to unrar
        for cmd in [['unar', '-o', str(out_dir), '-f', str(rar)],
                    ['unrar', 'x', '-o+', str(rar), str(out_dir) + '/']]:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                if result.returncode == 0:
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        if i % 50 == 0 or i == len(rar_files):
            print(f"    {i}/{len(rar_files)} done")


# ── Label parsing ─────────────────────────────────────────────────────────────

def parse_labels(root_dir: Path) -> dict:
    """
    Find and parse the OpenSARShip Excel label file.
    Returns dict keyed by image filename → metadata dict.
    """
    xlsx_files = list(root_dir.rglob('*.xlsx')) + list(root_dir.rglob('*.xls'))
    if not xlsx_files:
        print("  WARNING: No Excel label file found — ship_type will be NULL")
        return {}

    xlsx = xlsx_files[0]
    print(f"  Parsing labels from {xlsx.name}")

    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except ImportError:
        print("  openpyxl not installed — run: pip install openpyxl")
        return {}

    if not rows:
        return {}

    # Detect header row
    header = [str(c).strip().lower() if c else '' for c in rows[0]]
    print(f"  Columns: {header}")

    def col(name_fragments):
        for frag in name_fragments:
            for i, h in enumerate(header):
                if frag in h:
                    return i
        return None

    idx_id     = col(['ship id', 'image id', 'id'])
    idx_scene  = col(['scene', 'sar'])
    idx_type   = col(['ship type', 'type'])
    idx_length = col(['length'])
    idx_width  = col(['width'])

    labels = {}
    for row in rows[1:]:
        if not row or all(c is None for c in row):
            continue
        def get(i):
            return str(row[i]).strip() if i is not None and i < len(row) and row[i] is not None else None

        image_id  = get(idx_id)
        if not image_id:
            continue

        ship_type = get(idx_type)
        # Normalise to one of the 6 known classes
        if ship_type:
            matched = next((t for t in KNOWN_TYPES if t.lower() in ship_type.lower()), None)
            ship_type_norm = matched if matched else ship_type
        else:
            ship_type_norm = None

        try:
            length = float(row[idx_length]) if idx_length is not None and row[idx_length] else None
        except (TypeError, ValueError):
            length = None
        try:
            width  = float(row[idx_width])  if idx_width  is not None and row[idx_width]  else None
        except (TypeError, ValueError):
            width  = None

        labels[image_id] = {
            'scene_id':     get(idx_scene),
            'ship_type':    ship_type_norm,
            'ship_type_ais': ship_type,
            'ship_length_m': length,
            'ship_width_m':  width,
        }

    print(f"  {len(labels):,} label entries loaded")
    return labels


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest(root_dir: Path, labels: dict, dry_run: bool = False):
    # Find all chip images
    image_files = (list(root_dir.rglob('*.png')) +
                   list(root_dir.rglob('*.jpg')) +
                   list(root_dir.rglob('*.jpeg')) +
                   list(root_dir.rglob('*.tif')))
    print(f"\n  Found {len(image_files):,} chip images")

    if not image_files:
        print("  No images found — extraction may not have completed yet")
        return

    # Quick size sample
    from PIL import Image as PILImage
    sample = image_files[0]
    try:
        with PILImage.open(sample) as im:
            w, h = im.size
        print(f"  Sample chip size: {w}×{h}px  ({sample.name})")
    except Exception:
        w, h = None, None

    if dry_run:
        print(f"  DRY RUN — would insert up to {len(image_files):,} rows")
        # Show label coverage
        matched = sum(1 for f in image_files if f.stem in labels or f.name in labels)
        print(f"  Label match: {matched:,}/{len(image_files):,}")
        return

    conn = psycopg2.connect(**DB_CONFIG)
    inserted = skipped = 0

    with conn.cursor() as cur:
        for img_path in image_files:
            stem = img_path.stem
            name = img_path.name
            meta = labels.get(stem) or labels.get(name) or {}

            # Ship type encoded in filename: Cargo_x10101_y14959_vh.tif
            # Some scenes prefix with Visual_ or Cal_: Visual_Cargo_x..._vh.tif
            ship_type = meta.get('ship_type')
            if not ship_type:
                parts = stem.split('_')
                # Skip known non-type prefixes
                skip = {'visual', 'cal', 'patch'}
                for part in parts:
                    if part.lower() in skip:
                        continue
                    if part.lower().startswith('x') and part[1:].isdigit():
                        break  # hit the coordinate part
                    # Reconstruct multi-word types like "Other Type"
                    # by checking if the rest before _x forms a known type
                    ship_type = part
                    break
                # Handle "Other Type" (two words joined by underscore in filename)
                if ship_type and ship_type.lower() == 'other':
                    idx = parts.index(ship_type) if ship_type in parts else -1
                    if idx >= 0 and idx + 1 < len(parts) and not parts[idx+1].lower().startswith('x'):
                        ship_type = ship_type + ' ' + parts[idx+1]

            # Scene: grandparent dir is the scene folder (child is Patch/ or Patch_Cal/)
            scene_id = meta.get('scene_id') or img_path.parent.parent.name

            # Polarization from filename suffix (_vv, _vh, _hh, _hv) or scene name
            pol = meta.get('polarization')
            if not pol:
                name_upper = img_path.name.upper()
                scene_upper = scene_id.upper()
                for p in ['VV', 'VH', 'HH', 'HV']:
                    if f'_{p}.' in name_upper or f'_{p}_' in name_upper:
                        pol = p; break
                if not pol:
                    for p in ['SDV', 'SSV', 'SDH', 'SSH']:
                        if p in scene_upper:
                            pol = {'SDV': 'VV+VH', 'SSV': 'VV',
                                   'SDH': 'HH+HV', 'SSH': 'HH'}.get(p)
                            break

            try:
                cur.execute("""
                    INSERT INTO defense.opensar_images
                        (image_id, local_path, scene_id, ship_type, ship_type_ais,
                         ship_length_m, ship_width_m, polarization, image_width, image_height)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (image_id) DO NOTHING
                """, (
                    stem,
                    str(img_path),
                    scene_id,
                    ship_type,
                    meta.get('ship_type_ais'),
                    meta.get('ship_length_m'),
                    meta.get('ship_width_m'),
                    pol,
                    w, h,
                ))
                if cur.rowcount:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  Error on {stem}: {e}")
                conn.rollback()
                continue

        conn.commit()

    conn.close()
    print(f"  Inserted: {inserted:,}  Skipped (already exists): {skipped:,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip-path', type=Path,
                        default=DEST_DIR / 'OpenSARShip_2.zip',
                        help='Path to downloaded OpenSARShip_2.zip')
    parser.add_argument('--already-extracted', type=Path, default=None,
                        help='Skip extraction, point to existing OpenSARShip2/ folder')
    parser.add_argument('--skip-rar', action='store_true',
                        help='Skip RAR extraction (already done)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.already_extracted:
        root_dir = args.already_extracted
        print(f"Using pre-extracted dir: {root_dir}")
    else:
        if not args.zip_path.exists():
            print(f"ERROR: zip not found at {args.zip_path}")
            print("Is the download still in progress? Check:")
            print(f"  ls -lh {args.zip_path}")
            sys.exit(1)
        extract_zip(args.zip_path, DEST_DIR)
        root_dir = DEST_DIR / 'OpenSARShip2'

    if not args.skip_rar:
        extract_rars(root_dir)

    labels = parse_labels(root_dir)
    ingest(root_dir, labels, dry_run=args.dry_run)

    print("\nDone. Check with:")
    print("  SELECT ship_type, COUNT(*) FROM defense.opensar_images GROUP BY ship_type ORDER BY COUNT(*) DESC;")


if __name__ == '__main__':
    main()
