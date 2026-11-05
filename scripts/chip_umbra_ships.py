"""
AIS-matched ship chipping pipeline for Umbra open data GEC.tif files.

For each Umbra GEC scene:
  1. Parse UTC acquisition time from filename (YYYY-MM-DD-HH-MM-SS_UMBRA-XX_GEC.tif)
  2. Load AIS data for the scene date (±10 min window)
  3. Filter to vessels ≥50m within scene geographic bounds
  4. Convert lat/lon → pixel via rasterio direct transform (GEC is already geocorrected)
  5. Extract CHIP_SIZE×CHIP_SIZE ship chip
  6. Flag contamination (other large vessels in the chip)
  7. Save .npz with chip + metadata

Chip size 1024: at ~0.5m Umbra GSD, covers ~512m — captures most vessels including
large container ships (~300-400m).

Usage:
    venv/bin/python scripts/chip_umbra_ships.py --region LA
    venv/bin/python scripts/chip_umbra_ships.py --region Gulf
    venv/bin/python scripts/chip_umbra_ships.py  # all regions
"""

import argparse
import datetime
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import zstandard as zstd

CHIP_SIZE       = 1024
HALF            = CHIP_SIZE // 2
MIN_LENGTH      = 50
LARGE_THRESH    = 150
CLEAN_MAX_LARGE = 1

ML_BASE = Path("/Volumes/Machine Learning")

REGIONS = {
    'LA':         dict(name='LA/Long Beach',   scene_dir='umbra_open_data/LA',
                       ais_dir='ais_la',          out_dir='umbra_ship_chips_la'),
    'NY':         dict(name='NY/NJ Harbor',    scene_dir='umbra_open_data/NY',
                       ais_dir='ais_ny',           out_dir='umbra_ship_chips_ny'),
    'Gulf':       dict(name='Gulf/Houston',    scene_dir='umbra_open_data/Gulf',
                       ais_dir='ais_gulf',          out_dir='umbra_ship_chips_gulf'),
    'Chesapeake': dict(name='Chesapeake Bay',  scene_dir='umbra_open_data/Chesapeake',
                       ais_dir='ais_chesapeake',   out_dir='umbra_ship_chips_chesapeake'),
    'Puget':      dict(name='Puget Sound',     scene_dir='umbra_open_data/Puget',
                       ais_dir='ais_puget',         out_dir='umbra_ship_chips_puget'),
    'Houston':    dict(name='Houston Ship Channel', scene_dir='umbra_open_data/Houston',
                       ais_dir='ais_houston',        out_dir='umbra_ship_chips_houston'),
    'SFBay':      dict(name='SF Bay/Oakland',  scene_dir='umbra_open_data/SFBay',
                       ais_dir='ais_sfbay',         out_dir='umbra_ship_chips_sfbay'),
    'Savannah':   dict(name='Savannah/Brunswick', scene_dir='umbra_open_data/Savannah',
                       ais_dir='ais_savannah',      out_dir='umbra_ship_chips_savannah'),
    'Miami':      dict(name='Miami/Port Everglades', scene_dir='umbra_open_data/Miami',
                       ais_dir='ais_miami',         out_dir='umbra_ship_chips_miami'),
}


def parse_acq_time(fname):
    """Parse UTC time from 2025-01-08-06-32-24_UMBRA-10_GEC.tif."""
    m = re.match(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_', fname)
    if not m:
        raise ValueError(f"Cannot parse time from {fname}")
    return datetime.datetime.strptime(m.group(1), '%Y-%m-%d-%H-%M-%S').replace(
        tzinfo=datetime.timezone.utc)


def load_ais_window(ais_dir, date_str, t_center, window_minutes=10):
    path = ais_dir / f"ais-{date_str}.csv.zst"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        df = pd.read_csv(io.TextIOWrapper(dctx.stream_reader(f)), low_memory=False)
    df['base_date_time'] = pd.to_datetime(df['base_date_time'], utc=True, errors='coerce')
    t_lo = t_center - datetime.timedelta(minutes=window_minutes)
    t_hi = t_center + datetime.timedelta(minutes=window_minutes)
    return df[(df['base_date_time'] >= t_lo) & (df['base_date_time'] <= t_hi)].copy()


def lonlat_to_pixel(transform, lon, lat):
    """Convert geographic (lon, lat) to pixel (col, row) using rasterio affine transform."""
    rows, cols = rasterio.transform.rowcol(transform, [lon], [lat])
    return int(cols[0]), int(rows[0])


def extract_chip(src, col, row):
    r0 = max(0, row - HALF); r1 = min(src.height, row + HALF)
    c0 = max(0, col - HALF); c1 = min(src.width,  col + HALF)
    raw = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
    chip = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    dr = HALF - (row - r0); dc = HALF - (col - c0)
    chip[dr:dr + raw.shape[0], dc:dc + raw.shape[1]] = raw.astype(np.float32)
    return chip


def count_others_in_chip(target_col, target_row, all_cols, all_rows, all_lengths, target_idx):
    n_any = n_large = 0
    min_dist = np.inf
    for i, (c, r, length) in enumerate(zip(all_cols, all_rows, all_lengths)):
        if i == target_idx:
            continue
        if abs(c - target_col) <= HALF and abs(r - target_row) <= HALF:
            n_any += 1
            if length >= LARGE_THRESH:
                n_large += 1
        if length >= LARGE_THRESH:
            min_dist = min(min_dist, np.sqrt((c - target_col) ** 2 + (r - target_row) ** 2))
    return n_any, n_large, (float(min_dist) if min_dist < np.inf else -1.0)


def process_scene(tif_path, ais_dir, out_base):
    fname = tif_path.name
    acq_time = parse_acq_time(fname)
    date_str  = acq_time.strftime('%Y-%m-%d')
    scene_id  = fname.replace('_GEC.tif', '')

    with rasterio.open(tif_path) as src:
        transform = src.transform
        bounds    = src.bounds      # BoundingBox(left, bottom, right, top) in EPSG:4326
        W, H      = src.width, src.height

    ais = load_ais_window(ais_dir, date_str, acq_time)
    if ais is None or len(ais) == 0:
        return 0

    # Filter to scene footprint
    ais['longitude'] = pd.to_numeric(ais['longitude'], errors='coerce')
    ais['latitude']  = pd.to_numeric(ais['latitude'],  errors='coerce')
    ais = ais[
        (ais['longitude'] >= bounds.left)  & (ais['longitude'] <= bounds.right) &
        (ais['latitude']  >= bounds.bottom) & (ais['latitude']  <= bounds.top)
    ]

    ais['dt_sec'] = (ais['base_date_time'] - pd.Timestamp(acq_time)).abs().dt.total_seconds()
    vessels = ais.sort_values('dt_sec').groupby('mmsi').first().reset_index()
    vessels = vessels[vessels['length'].fillna(0) >= MIN_LENGTH].reset_index(drop=True)

    if len(vessels) == 0:
        return 0

    all_cols = np.array([lonlat_to_pixel(transform, v['longitude'], v['latitude'])[0]
                         for _, v in vessels.iterrows()])
    all_rows = np.array([lonlat_to_pixel(transform, v['longitude'], v['latitude'])[1]
                         for _, v in vessels.iterrows()])
    all_lengths = vessels['length'].fillna(0).values

    scene_out = out_base / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    with rasterio.open(tif_path) as src:
        for idx, row in vessels.iterrows():
            col = int(all_cols[idx])
            r   = int(all_rows[idx])
            if not (HALF <= col < W - HALF and HALF <= r < H - HALF):
                continue
            chip = extract_chip(src, col, r)
            if chip.max() == 0:
                continue
            n_other, n_large, nearest_px = count_others_in_chip(
                col, r, all_cols, all_rows, all_lengths, idx)
            out_path = scene_out / f"{int(row['mmsi'])}_{acq_time.strftime('%Y%m%dT%H%M%S')}.npz"
            np.savez_compressed(str(out_path),
                chip=chip,
                mmsi=int(row['mmsi']),
                vessel_name=str(row.get('vessel_name', '')),
                vessel_type=int(row['vessel_type'] if pd.notna(row.get('vessel_type')) else -1),
                length_m=float(row['length'] if pd.notna(row.get('length')) else 0),
                lat=float(row['latitude']), lon=float(row['longitude']),
                sog=float(row['sog'] if pd.notna(row.get('sog')) else 0),
                heading=float(row['heading'] if pd.notna(row.get('heading')) else 0),
                scene_id=scene_id, acq_time=acq_time.isoformat(),
                pixel_col=col, pixel_row=r, dt_sec=float(row['dt_sec']),
                n_other_ships=n_other, n_other_large=n_large,
                nearest_large_px=nearest_px,
                clean=bool(n_large <= CLEAN_MAX_LARGE),
            )
            n_saved += 1

    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=list(REGIONS.keys()), default=None)
    args = parser.parse_args()

    active = {args.region: REGIONS[args.region]} if args.region else REGIONS

    for key, r in active.items():
        scene_dir = ML_BASE / r['scene_dir']
        ais_dir   = ML_BASE / r['ais_dir']
        out_base  = ML_BASE / r['out_dir']
        out_base.mkdir(exist_ok=True)

        tifs = sorted(scene_dir.glob('*_GEC.tif'))
        print(f"\nRegion: {r['name']}  ({len(tifs)} scenes, AIS: {ais_dir})")

        total = 0
        for tif in tifs:
            scene_id = tif.name.replace('_GEC.tif', '')
            out_dir  = out_base / scene_id
            if out_dir.exists() and any(out_dir.iterdir()):
                n = sum(1 for _ in out_dir.iterdir())
                print(f"  {tif.name}  already done ({n} chips)")
                total += n
                continue
            print(f"  {tif.name}...", end=' ', flush=True)
            try:
                n = process_scene(tif, ais_dir, out_base)
                print(f"{n} chips")
            except Exception as e:
                print(f"ERROR — {e}")
                n = 0
            total += n

        print(f"  → {total} chips total for {r['name']}")
        print(f"     Saved to: {out_base}")


if __name__ == '__main__':
    main()
