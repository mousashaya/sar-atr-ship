"""
scripts/ingest_mstar.py
========================
Ingest MSTAR public dataset into defense.mstar_images.

MSTAR binary format:
  - ASCII Phoenix header (variable length, ends with [EndofPhoenixHeader])
  - Raw float32 magnitude block (big-endian, Sun format)
  - Raw float32 phase block

All metadata comes from the header — no guessing.

Dataset structure:
  MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/
    TARGETS/TRAIN/{depression_deg}/{vehicle}/SN_{serial}/{files}
    TARGETS/TEST/{depression_deg}/{vehicle}/SN_{serial}/{files}
    SLICY/{depression_deg}/{files}          -- background clutter
  MSTAR_PUBLIC_MIXED_TARGETS_CD1/
    {depression_deg}/COL1|COL2/SCENE1/{vehicle}/{files}
  MSTAR_PUBLIC_MIXED_TARGETS_CD2/
    {depression_deg}/COL1|COL2/SCENE1/{vehicle}/{files}

Usage:
    python3 scripts/ingest_mstar.py
    python3 scripts/ingest_mstar.py --dry-run
    python3 scripts/ingest_mstar.py --convert-png   # also save PNG chips
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

MSTAR_ROOT = Path('/Volumes/Machine Learning/mstar')
PNG_DIR    = Path('/Volumes/Machine Learning/mstar/chips_png')

# Map raw TargetType strings to clean short names
TYPE_MAP = {
    't72_tank':  'T72',
    'bmp2_tank': 'BMP2',
    'btr70_tank':'BTR70',
    'btr60':     'BTR60',
    '2s1':       '2S1',
    'brdm_2':    'BRDM2',
    'd7':        'D7',
    't62':       'T62',
    'zsu_23_4':  'ZSU234',
    'zsu23/4':   'ZSU234',
    'slicy':     'SLICY',
}


# ── MSTAR binary reader ───────────────────────────────────────────────────────

def parse_header(data: bytes) -> dict:
    """Extract all key-value pairs from the Phoenix ASCII header."""
    end_tag = b'[EndofPhoenixHeader]'
    idx = data.find(end_tag)
    if idx < 0:
        return {}
    header_text = data[:idx].decode('ascii', errors='replace')
    header_end_byte = idx + len(end_tag)

    meta = {'_header_end': header_end_byte}
    for line in header_text.splitlines():
        line = line.strip()
        if '=' in line and not line.startswith('['):
            key, _, val = line.partition('=')
            meta[key.strip()] = val.strip()
    return meta


def read_mstar(path: Path) -> tuple[dict, np.ndarray]:
    """
    Read one MSTAR binary file.
    Returns (metadata_dict, magnitude_array).
    magnitude_array is float32, shape (rows, cols).
    """
    with open(path, 'rb') as f:
        raw = f.read()

    meta = parse_header(raw)
    if not meta:
        return {}, None

    n_rows = int(meta.get('NumberOfRows', 0))
    n_cols = int(meta.get('NumberOfColumns', 0))

    if n_rows == 0 or n_cols == 0:
        return meta, None

    # PhoenixHeaderLength is the padded data-start offset (may include null padding)
    data_start = int(meta.get('PhoenixHeaderLength', meta['_header_end']))
    n_pixels   = n_rows * n_cols
    mag_bytes  = n_pixels * 4

    mag_raw = raw[data_start: data_start + mag_bytes]
    if len(mag_raw) < mag_bytes:
        return meta, None

    mag = np.frombuffer(mag_raw, dtype='>f4').astype(np.float32)
    mag = mag.reshape(n_rows, n_cols)
    return meta, mag


def mag_to_uint8(mag: np.ndarray) -> np.ndarray:
    """Convert raw SAR magnitude to 8-bit for PNG export."""
    # Log-scale stretch: common for SAR visualisation
    mag = np.clip(mag, 1e-6, None)
    log = np.log10(mag)
    lo, hi = np.percentile(log, [1, 99])
    stretched = np.clip((log - lo) / (hi - lo + 1e-9), 0, 1)
    return (stretched * 255).astype(np.uint8)


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_file(cur, path: Path, split: str, dry_run: bool, convert_png: bool) -> bool:
    """Parse one MSTAR file and insert into DB. Returns True if inserted."""
    meta, mag = read_mstar(path)
    if not meta:
        return False

    raw_type = meta.get('TargetType', '').lower().replace(' ', '_')
    target_short = TYPE_MAP.get(raw_type)
    if not target_short:
        # Try partial match
        for k, v in TYPE_MAP.items():
            if k in raw_type or raw_type in k:
                target_short = v
                break
        if not target_short:
            target_short = raw_type.upper()[:10] if raw_type else 'UNKNOWN'

    def flt(key):
        try: return float(meta[key]) if key in meta else None
        except: return None

    record = {
        'filename':          path.name,
        'local_path':        str(path),
        'target_type':       meta.get('TargetType', ''),
        'target_short':      target_short,
        'serial_number':     meta.get('TargetSerNum', ''),
        'azimuth_angle':     flt('TargetAz'),
        'depression_angle':  flt('MeasuredDepression'),
        'desired_depression':flt('DesiredDepression'),
        'polarization':      meta.get('Polarization', ''),
        'n_cols':            int(meta.get('NumberOfColumns', 0)) or None,
        'n_rows':            int(meta.get('NumberOfRows', 0)) or None,
        'split':             split,
        'collection_date':   meta.get('CollectionDate', ''),
        'radar_mode':        meta.get('RadarMode', ''),
        'site':              meta.get('Site', ''),
    }

    if dry_run:
        return True

    try:
        cur.execute("""
            INSERT INTO defense.mstar_images
                (filename, local_path, target_type, target_short, serial_number,
                 azimuth_angle, depression_angle, desired_depression, polarization,
                 n_cols, n_rows, split, collection_date, radar_mode, site)
            VALUES (%(filename)s,%(local_path)s,%(target_type)s,%(target_short)s,
                    %(serial_number)s,%(azimuth_angle)s,%(depression_angle)s,
                    %(desired_depression)s,%(polarization)s,%(n_cols)s,%(n_rows)s,
                    %(split)s,%(collection_date)s,%(radar_mode)s,%(site)s)
            ON CONFLICT (filename) DO NOTHING
        """, record)
        inserted = cur.rowcount > 0
    except Exception as e:
        print(f"  DB error {path.name}: {e}")
        return False

    if inserted and convert_png and mag is not None:
        save_png(path, target_short, mag, record['desired_depression'])

    return inserted


def save_png(src_path: Path, target_short: str, mag: np.ndarray, depression):
    from PIL import Image as PILImage
    out_dir = PNG_DIR / target_short / f'{int(depression or 0):02d}deg'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src_path.stem + '.png')
    if out.exists():
        return
    img = PILImage.fromarray(mag_to_uint8(mag), mode='L').convert('RGB')
    img.save(out)


def determine_split(path: Path) -> str:
    """Infer TRAIN/TEST from path components."""
    parts = [p.upper() for p in path.parts]
    if 'TRAIN' in parts:
        return 'TRAIN'
    if 'TEST' in parts:
        return 'TEST'
    return 'MIXED'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',     action='store_true')
    parser.add_argument('--convert-png', action='store_true',
                        help='Export magnitude chips as PNG (for viewer)')
    args = parser.parse_args()

    # Find all binary data files (no extension or numeric extension like .015)
    all_files = []
    for root_dir in MSTAR_ROOT.iterdir():
        if not root_dir.is_dir() or root_dir.name.endswith('.zip'):
            continue
        for f in root_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() not in {
                '.htm', '.html', '.gif', '.jpg', '.jpeg', '.png',
                '.txt', '.xml', '.zip', '.rar'
            }:
                all_files.append(f)

    print(f"Found {len(all_files):,} MSTAR binary files")

    if args.dry_run:
        # Quick sample to verify reader
        sample = all_files[0]
        meta, mag = read_mstar(sample)
        print(f"\nSample: {sample.name}")
        print(f"  Target:     {meta.get('TargetType')}")
        print(f"  Depression: {meta.get('DesiredDepression')}°")
        print(f"  Azimuth:    {meta.get('TargetAz')}°")
        print(f"  Size:       {meta.get('NumberOfRows')}×{meta.get('NumberOfColumns')}")
        print(f"  Polar:      {meta.get('Polarization')}")
        if mag is not None:
            print(f"  Mag range:  {mag.min():.2f} – {mag.max():.2f}")
        print(f"\nDRY RUN — would insert {len(all_files):,} rows")
        return

    conn = psycopg2.connect(**DB_CONFIG)
    inserted = skipped = errors = 0
    with conn.cursor() as cur:
        for i, f in enumerate(all_files, 1):
            split = determine_split(f)
            ok = ingest_file(cur, f, split, dry_run=False,
                             convert_png=args.convert_png)
            if ok:
                inserted += 1
            else:
                skipped += 1
            if i % 1000 == 0:
                conn.commit()
                print(f"  {i:,}/{len(all_files):,}  inserted={inserted:,}")
        conn.commit()
    conn.close()

    print(f"\nDone. Inserted: {inserted:,}  Skipped: {skipped:,}")
    print("\nBreakdown:")

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT target_short, desired_depression, split, COUNT(*)
            FROM defense.mstar_images
            GROUP BY target_short, desired_depression, split
            ORDER BY target_short, desired_depression
        """)
        cur_target = None
        for row in cur.fetchall():
            tgt, dep, split, cnt = row
            if tgt != cur_target:
                print(f"\n  {tgt}")
                cur_target = tgt
            print(f"    {dep:>4}°  {split:<6}  {cnt:>4} images")
    conn.close()


if __name__ == '__main__':
    main()
