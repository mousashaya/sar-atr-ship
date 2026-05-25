"""
scripts/populate_ship_lengths.py
==================================
Parse Ship.xml from every OpenSARShip2 scene and populate
defense.opensar_images.ship_length_m and ship_width_m.

Manual_Length  → ship_length_m   (SAR-measured, most reliable)
Breadth_extreme → ship_width_m   (from MarineTrafficInformation)

Matching key: (scene_id, Center_x, Center_y) from image_id field.

Usage:
    venv/bin/python3 scripts/populate_ship_lengths.py
"""

import re
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

PATCH_CAL_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')


def _tag(text: bytes, tag: str):
    """Extract first occurrence of <tag>value</tag> as string, or None."""
    m = re.search(rb'<' + tag.encode() + rb'>(.*?)</' + tag.encode() + rb'>', text)
    return m.group(1).decode('utf-8', errors='ignore').strip() if m else None


def parse_ship_xml(xml_path: Path) -> dict:
    """Return dict keyed by (cx, cy) → {length_m, width_m}.
    Uses regex instead of XML parser to handle malformed ship name strings."""
    out = {}
    try:
        raw = xml_path.read_bytes()
        # Split into per-ship blocks on <ship> boundaries
        blocks = re.split(rb'<ship>', raw)[1:]   # skip header before first <ship>
        for block in blocks:
            # SARShipInformation block
            sar_m = re.search(rb'<SARShipInformation>(.*?)</SARShipInformation>',
                              block, re.DOTALL)
            if not sar_m:
                continue
            sar = sar_m.group(1)
            cx = _tag(sar, 'Center_x')
            cy = _tag(sar, 'Center_y')
            ml = _tag(sar, 'Manual_Length')
            if None in (cx, cy, ml):
                continue
            # MarineTrafficInformation for width
            width = None
            mt_m = re.search(rb'<MarineTrafficInformation>(.*?)</MarineTrafficInformation>',
                             block, re.DOTALL)
            if mt_m:
                w = _tag(mt_m.group(1), 'Breadth_extreme')
                if w and w != 'NaN':
                    try:
                        width = float(w)
                    except ValueError:
                        pass
            try:
                out[(int(cx), int(cy))] = {
                    'length_m': float(ml),
                    'width_m':  width,
                }
            except ValueError:
                pass
    except Exception as e:
        print(f"  WARN: {xml_path.name}: {e}")
    return out


def extract_center(image_id: str, polarization: str) -> tuple:
    """Parse (cx, cy) from image_id like 'S1A_...__Cargo_x191_y9482_VV'."""
    try:
        stem_pol = image_id.split('__', 1)[1]           # Cargo_x191_y9482_VV
        stem     = stem_pol.rsplit(f'_{polarization}', 1)[0]  # Cargo_x191_y9482
        parts    = stem.split('_')
        cx = int(next(p[1:] for p in parts if p.startswith('x')))
        cy = int(next(p[1:] for p in parts if p.startswith('y')))
        return cx, cy
    except Exception:
        return None, None


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Load all records (one per chip, VV only to avoid duplicates)
    cur.execute("""
        SELECT id, image_id, scene_id, polarization
        FROM defense.opensar_images
    """)
    rows = cur.fetchall()
    print(f"Loaded {len(rows):,} records from DB")

    # Build scene → Ship.xml lookup
    xml_cache = {}
    scenes = {r['scene_id'] for r in rows}
    print(f"Scenes to parse: {len(scenes)}")
    found_xml, missing_xml = 0, 0
    for scene_id in sorted(scenes):
        xml_path = PATCH_CAL_ROOT / scene_id / 'Ship.xml'
        if xml_path.exists():
            xml_cache[scene_id] = parse_ship_xml(xml_path)
            found_xml += 1
        else:
            xml_cache[scene_id] = {}
            missing_xml += 1
    print(f"  Found: {found_xml}  Missing: {missing_xml}")

    # Match records and update
    updates = []
    matched, unmatched = 0, 0
    for r in rows:
        cx, cy = extract_center(r['image_id'], r['polarization'])
        if cx is None:
            unmatched += 1
            continue
        entry = xml_cache.get(r['scene_id'], {}).get((cx, cy))
        if entry:
            updates.append((entry['length_m'], entry['width_m'], r['id']))
            matched += 1
        else:
            unmatched += 1

    print(f"  Matched: {matched:,}  Unmatched: {unmatched:,}")

    # Bulk update
    upd_cur = conn.cursor()
    upd_cur.executemany("""
        UPDATE defense.opensar_images
        SET ship_length_m = %s, ship_width_m = %s
        WHERE id = %s
    """, updates)
    conn.commit()
    print(f"Updated {upd_cur.rowcount:,} rows")

    # Summary by ship type (VV only)
    cur.execute("""
        SELECT ship_type,
               COUNT(*)              AS n,
               COUNT(ship_length_m)  AS has_len,
               ROUND(AVG(ship_length_m)::numeric, 1) AS avg_m,
               ROUND(MIN(ship_length_m)::numeric, 1) AS min_m,
               ROUND(MAX(ship_length_m)::numeric, 1) AS max_m
        FROM defense.opensar_images
        WHERE polarization = 'VV'
          AND ship_type IN ('Cargo', 'Tanker', 'Other Type')
        GROUP BY ship_type ORDER BY ship_type
    """)
    print("\nLength coverage by class (VV):")
    print(f"  {'Type':<14} {'n':>6} {'has_len':>8} {'avg_m':>8} {'min_m':>8} {'max_m':>8}")
    for r in cur.fetchall():
        print(f"  {r['ship_type']:<14} {r['n']:>6} {r['has_len']:>8} "
              f"{str(r['avg_m']):>8} {str(r['min_m']):>8} {str(r['max_m']):>8}")

    conn.close()


if __name__ == '__main__':
    main()
