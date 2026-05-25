"""
SAR ship chipping pipeline.

For each Sentinel-1 IW GRD scene (12:23 UTC descending pass — Houston Ship Channel):
  1. Parse acquisition time from filename
  2. Load matching AIS file (±10 min window)
  3. Filter AIS to vessels ≥50m within scene bounds
  4. Compute pixel coordinates via degree-2 polynomial GCP fit
  5. Extract 512×512 dual-pol chip (captures Kelvin wake)
  6. Flag contamination: record how many other AIS vessels fall inside the chip

Chip size: 512×512 pixels = 5.12km × 5.12km at 10m/pixel.
The Kelvin wake at 12 knots extends ~1.5km — fully captured in this window.

Output: /Volumes/Machine Learning/sar_ship_chips/<scene_id>/<MMSI>_<acq_time>.npz
Each file contains:
  chip_vv, chip_vh              (512, 512) float32 amplitude
  mmsi, vessel_name, vessel_type, length_m, lat, lon, sog, heading
  scene_id, acq_time, pixel_col, pixel_row, dt_sec
  n_other_ships                 int — AIS vessels ≥50m also inside chip
  n_other_large                 int — AIS vessels ≥150m also inside chip
  clean                         bool — no other vessel ≥150m in chip
"""

import os
import re
import glob
import datetime
import io
import numpy as np
import pandas as pd
import zstandard as zstd
import rasterio
from pathlib import Path

CHIP_SIZE = 512
HALF = CHIP_SIZE // 2

AIS_DIR = "/Volumes/Machine Learning/ais_noaa"
SCENE_DIR = "/Volumes/Machine Learning/sentinel1_houston"
OUT_DIR = "/Volumes/Machine Learning/sar_ship_chips"
os.makedirs(OUT_DIR, exist_ok=True)

# Only process descending-pass scenes (~12:23 UTC) — they cover Houston Ship Channel.
# Ascending-pass scenes (~00:19 UTC) cover inland Texas.
WATER_SCENE_PATTERN = re.compile(r'_\d{8}T12\d{4}_')

LAT_MIN, LAT_MAX = 28.0, 32.0
LON_MIN, LON_MAX = -98.0, -93.5
MIN_LENGTH = 50        # meters — minimum vessel length to chip
LARGE_THRESHOLD = 150  # meters — threshold for "large" contamination flag
CLEAN_MAX_LARGE = 1   # "clean" = at most this many large ships in chip (0 = strict, 1 = practical)


def parse_acq_time(scene_name):
    m = re.search(r'(\d{8}T\d{6})_(\d{8}T\d{6})', scene_name)
    if not m:
        raise ValueError(f"Cannot parse time from {scene_name}")
    t1 = datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
    t2 = datetime.datetime.strptime(m.group(2), '%Y%m%dT%H%M%S')
    return t1 + (t2 - t1) / 2


def load_ais_window(date_str, t_center, window_minutes=10):
    path = os.path.join(AIS_DIR, f"ais-{date_str}.csv.zst")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        df = pd.read_csv(io.TextIOWrapper(reader), low_memory=False)
    df['base_date_time'] = pd.to_datetime(df['base_date_time'])
    t_lo = t_center - datetime.timedelta(minutes=window_minutes)
    t_hi = t_center + datetime.timedelta(minutes=window_minutes)
    mask = (
        (df['base_date_time'] >= t_lo) &
        (df['base_date_time'] <= t_hi) &
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    )
    return df[mask].copy()


def build_gcp_transform(gcps):
    """Degree-2 polynomial fit: (lon, lat) -> (col, row)."""
    lons = np.array([g.x for g in gcps])
    lats = np.array([g.y for g in gcps])
    cols = np.array([g.col for g in gcps])
    rows = np.array([g.row for g in gcps])
    A = np.column_stack([np.ones(len(lons)), lons, lats, lons*lats, lons**2, lats**2])
    col_c, _, _, _ = np.linalg.lstsq(A, cols, rcond=None)
    row_c, _, _, _ = np.linalg.lstsq(A, rows, rcond=None)
    return col_c, row_c


def lonlat_to_pixel(lon, lat, col_c, row_c):
    pt = np.array([1, lon, lat, lon*lat, lon**2, lat**2])
    return int(pt @ col_c), int(pt @ row_c)


def in_scene_bounds(lon, lat, gcp_lons, gcp_lats):
    return (gcp_lons.min() <= lon <= gcp_lons.max() and
            gcp_lats.min() <= lat <= gcp_lats.max())


def extract_chip(src, col, row):
    """Extract CHIP_SIZE×CHIP_SIZE chip, zero-padded if near scene edge."""
    r0 = max(0, row - HALF)
    r1 = min(src.height, row + HALF)
    c0 = max(0, col - HALF)
    c1 = min(src.width, col + HALF)
    raw = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
    chip = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    dr = HALF - (row - r0)
    dc = HALF - (col - c0)
    chip[dr:dr + raw.shape[0], dc:dc + raw.shape[1]] = raw
    return chip


def count_others_in_chip(target_col, target_row, all_cols, all_rows, all_lengths, target_idx):
    """Count other vessels in chip bounds; return counts and nearest-neighbour pixel distance."""
    n_any = 0
    n_large = 0
    min_dist_px = np.inf
    for i, (c, r, length) in enumerate(zip(all_cols, all_rows, all_lengths)):
        if i == target_idx:
            continue
        dist = np.sqrt((c - target_col)**2 + (r - target_row)**2)
        if dist <= HALF * np.sqrt(2):  # within chip square (diagonal check)
            if abs(c - target_col) <= HALF and abs(r - target_row) <= HALF:
                n_any += 1
                if length >= LARGE_THRESHOLD:
                    n_large += 1
        if length >= LARGE_THRESHOLD:
            min_dist_px = min(min_dist_px, dist)
    return n_any, n_large, (float(min_dist_px) if min_dist_px < np.inf else -1.0)


def process_scene(safe_path):
    safe_name = os.path.basename(safe_path)
    acq_time = parse_acq_time(safe_name)
    date_str = acq_time.strftime('%Y-%m-%d')

    vv_files = glob.glob(os.path.join(safe_path, 'measurement', '*-vv-*.tiff'))
    vh_files = glob.glob(os.path.join(safe_path, 'measurement', '*-vh-*.tiff'))
    if not vv_files:
        print(f"  No VV file in {safe_name}")
        return 0

    vv_path = vv_files[0]
    vh_path = vh_files[0] if vh_files else None

    with rasterio.open(vv_path) as src:
        gcps, _ = src.gcps
        gcp_lons = np.array([g.x for g in gcps])
        gcp_lats = np.array([g.y for g in gcps])
        W, H = src.width, src.height
        col_c, row_c = build_gcp_transform(gcps)

    ais = load_ais_window(date_str, acq_time)
    if ais is None or len(ais) == 0:
        print(f"  No AIS data for {date_str}")
        return 0

    # Best ping per vessel (closest to SAR acquisition)
    ais['dt_sec'] = (ais['base_date_time'] - pd.Timestamp(acq_time)).abs().dt.total_seconds()
    vessel_best = ais.sort_values('dt_sec').groupby('mmsi').first().reset_index()

    # Filter: minimum length, inside scene bounds
    vessel_best = vessel_best[
        vessel_best['length'].fillna(0) >= MIN_LENGTH
    ].copy()
    vessel_best = vessel_best[
        vessel_best.apply(
            lambda r: in_scene_bounds(r['longitude'], r['latitude'], gcp_lons, gcp_lats),
            axis=1
        )
    ].reset_index(drop=True)

    if len(vessel_best) == 0:
        return 0

    # Precompute pixel coords for all vessels (for contamination check)
    all_cols, all_rows = [], []
    for _, v in vessel_best.iterrows():
        c, r = lonlat_to_pixel(v['longitude'], v['latitude'], col_c, row_c)
        all_cols.append(c)
        all_rows.append(r)
    all_cols = np.array(all_cols)
    all_rows = np.array(all_rows)
    all_lengths = vessel_best['length'].fillna(0).values

    scene_out_dir = os.path.join(OUT_DIR, safe_name.replace('.SAFE', ''))
    os.makedirs(scene_out_dir, exist_ok=True)

    n_saved = 0
    with rasterio.open(vv_path) as src_vv:
        vh_src = rasterio.open(vh_path) if vh_path else None
        try:
            for idx, row in vessel_best.iterrows():
                col = int(all_cols[idx])
                r = int(all_rows[idx])

                # Require full chip fits within scene (no edge padding)
                if not (HALF <= col < W - HALF and HALF <= r < H - HALF):
                    continue

                chip_vv = extract_chip(src_vv, col, r)
                chip_vh = extract_chip(vh_src, col, r) if vh_src else None

                # Contamination: how many other AIS vessels are inside this chip?
                n_other, n_other_large, nearest_large_px = count_others_in_chip(
                    col, r, all_cols, all_rows, all_lengths, idx
                )

                out_name = f"{int(row['mmsi'])}_{acq_time.strftime('%Y%m%dT%H%M%S')}.npz"
                out_path = os.path.join(scene_out_dir, out_name)

                save_dict = dict(
                    chip_vv=chip_vv,
                    mmsi=int(row['mmsi']),
                    vessel_name=str(row.get('vessel_name', '')),
                    vessel_type=int(row['vessel_type'] if pd.notna(row.get('vessel_type')) else -1),
                    length_m=float(row['length'] if pd.notna(row.get('length')) else 0),
                    lat=float(row['latitude']),
                    lon=float(row['longitude']),
                    sog=float(row['sog'] if pd.notna(row.get('sog')) else 0),
                    heading=float(row['heading'] if pd.notna(row.get('heading')) else 0),
                    scene_id=safe_name,
                    acq_time=acq_time.isoformat(),
                    pixel_col=col,
                    pixel_row=int(r),
                    dt_sec=float(row['dt_sec']),
                    n_other_ships=n_other,
                    n_other_large=n_other_large,
                    nearest_large_px=nearest_large_px,
                    clean=bool(n_other_large <= CLEAN_MAX_LARGE),
                )
                if chip_vh is not None:
                    save_dict['chip_vh'] = chip_vh

                np.savez_compressed(out_path, **save_dict)
                n_saved += 1
        finally:
            if vh_src:
                vh_src.close()

    return n_saved


def main():
    all_safe = sorted(glob.glob(os.path.join(SCENE_DIR, '**/*.SAFE'), recursive=True))
    # Deduplicate (same SAFE may live in scene_water/ and scenes_water/)
    seen = {}
    for path in all_safe:
        name = os.path.basename(path)
        if name not in seen:
            seen[name] = path
    # Keep only 12:23 UTC descending-pass scenes
    safe_dirs = [p for name, p in sorted(seen.items()) if WATER_SCENE_PATTERN.search(name)]
    print(f"Found {len(safe_dirs)} water-scene SAFE directories")

    total = 0
    for safe_path in safe_dirs:
        safe_name = os.path.basename(safe_path)
        out_dir = os.path.join(OUT_DIR, safe_name.replace('.SAFE', ''))
        if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 10:
            n = len(os.listdir(out_dir))
            print(f"Already done {safe_name} ({n} chips) — skipping")
            total += n
            continue
        print(f"Processing {safe_name}...")
        n = process_scene(safe_path)
        print(f"  Saved {n} chips")
        total += n

    print(f"\nTotal chips: {total}")


if __name__ == '__main__':
    main()
