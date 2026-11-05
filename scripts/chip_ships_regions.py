"""
Generalized SAR ship chipping pipeline for multiple regions.

For each Sentinel-1 IW GRD scene:
  1. Parse acquisition time from filename
  2. Load matching AIS file for the region (±10 min window)
  3. Filter AIS to vessels ≥50m within scene bounds
  4. Compute pixel coordinates via degree-2 polynomial GCP fit
  5. Extract 512×512 dual-pol ship chip
  6. Flag contamination
  7. Extract random background patches for backbone pre-training

Usage:
    venv/bin/python scripts/chip_ships_regions.py --region LA
    venv/bin/python scripts/chip_ships_regions.py --region Chesapeake --delete-scenes
    venv/bin/python scripts/chip_ships_regions.py --region Puget --delete-scenes --backbone-patches 30
"""

import argparse
import datetime
import glob
import io
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import zstandard as zstd

CHIP_SIZE       = 512
HALF            = CHIP_SIZE // 2
MIN_LENGTH      = 50
LARGE_THRESH    = 150
CLEAN_MAX_LARGE = 1
BACKBONE_PATCHES_DEFAULT = 20  # random patches per scene for backbone pre-training

ML_BASE = Path("/Volumes/Machine Learning")

REGIONS = {
    'LA':         dict(name='LA/Long Beach',  scene_dir='sentinel1_la',
                       ais_dir='ais_la',          out_dir='sar_ship_chips_la'),
    'NY':         dict(name='NY/NJ Harbor',   scene_dir='sentinel1_ny',
                       ais_dir='ais_ny',          out_dir='sar_ship_chips_ny'),
    'Chesapeake': dict(name='Chesapeake Bay', scene_dir='sentinel1_chesapeake',
                       ais_dir='ais_chesapeake',  out_dir='sar_ship_chips_chesapeake'),
    'Puget':      dict(name='Puget Sound',     scene_dir='sentinel1_puget',
                       ais_dir='ais_puget',         out_dir='sar_ship_chips_puget'),
    'Norway':     dict(name='Norwegian Coast', scene_dir='sentinel1_norway',
                       ais_dir='ais_norway',         out_dir='sar_ship_chips_norway'),
}


def parse_acq_time(scene_name):
    m = re.search(r'(\d{8}T\d{6})_(\d{8}T\d{6})', scene_name)
    if not m:
        raise ValueError(f"Cannot parse time from {scene_name}")
    t1 = datetime.datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
    t2 = datetime.datetime.strptime(m.group(2), '%Y%m%dT%H%M%S')
    return t1 + (t2 - t1) / 2


def load_ais_window(ais_dir, date_str, t_center, window_minutes=10):
    path = ais_dir / f"ais-{date_str}.csv.zst"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        df = pd.read_csv(io.TextIOWrapper(dctx.stream_reader(f)), low_memory=False)
    df['base_date_time'] = pd.to_datetime(df['base_date_time'])
    t_lo = t_center - datetime.timedelta(minutes=window_minutes)
    t_hi = t_center + datetime.timedelta(minutes=window_minutes)
    return df[(df['base_date_time'] >= t_lo) & (df['base_date_time'] <= t_hi)].copy()


def build_gcp_transform(gcps):
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
    r0 = max(0, row - HALF); r1 = min(src.height, row + HALF)
    c0 = max(0, col - HALF); c1 = min(src.width,  col + HALF)
    raw = src.read(1, window=rasterio.windows.Window(c0, r0, c1-c0, r1-r0))
    chip = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    dr = HALF - (row - r0); dc = HALF - (col - c0)
    chip[dr:dr+raw.shape[0], dc:dc+raw.shape[1]] = raw
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
            min_dist = min(min_dist, np.sqrt((c-target_col)**2 + (r-target_row)**2))
    return n_any, n_large, (float(min_dist) if min_dist < np.inf else -1.0)


def extract_backbone_patches(src_vv, vh_src, W, H, ship_cols, ship_rows,
                              out_dir, scene_id, n_patches, rng):
    """Sample random 512x512 patches avoiding ship locations."""
    saved = 0
    attempts = 0
    while saved < n_patches and attempts < n_patches * 20:
        attempts += 1
        col = rng.integers(HALF, W - HALF)
        row = rng.integers(HALF, H - HALF)
        # Skip if too close to any known ship
        if len(ship_cols) > 0:
            dists = np.sqrt((ship_cols - col)**2 + (ship_rows - row)**2)
            if dists.min() < HALF * 2:
                continue
        chip_vv = extract_chip(src_vv, col, row)
        if chip_vv.max() == 0:
            continue
        chip_vh = extract_chip(vh_src, col, row) if vh_src else None
        save_dict = dict(chip_vv=chip_vv, scene_id=scene_id, pixel_col=col, pixel_row=row)
        if chip_vh is not None:
            save_dict['chip_vh'] = chip_vh
        out_path = out_dir / f"{scene_id}_{col}_{row}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        saved += 1
    return saved


def process_scene(safe_path, ais_dir, out_base, backbone_out=None, n_backbone=0):
    safe_name = os.path.basename(safe_path)
    acq_time  = parse_acq_time(safe_name)
    date_str  = acq_time.strftime('%Y-%m-%d')

    vv_files = glob.glob(os.path.join(safe_path, 'measurement', '*-vv-*.tiff'))
    vh_files = glob.glob(os.path.join(safe_path, 'measurement', '*-vh-*.tiff'))
    if not vv_files:
        return 0

    with rasterio.open(vv_files[0]) as src:
        gcps, _ = src.gcps
        gcp_lons = np.array([g.x for g in gcps])
        gcp_lats = np.array([g.y for g in gcps])
        W, H = src.width, src.height
        col_c, row_c = build_gcp_transform(gcps)

    ais = load_ais_window(ais_dir, date_str, acq_time)
    if ais is None or len(ais) == 0:
        return 0

    ais['dt_sec'] = (ais['base_date_time'] - pd.Timestamp(acq_time)).abs().dt.total_seconds()
    vessels = ais.sort_values('dt_sec').groupby('mmsi').first().reset_index()
    vessels = vessels[vessels['length'].fillna(0) >= MIN_LENGTH]
    vessels = vessels[vessels.apply(
        lambda r: in_scene_bounds(r['longitude'], r['latitude'], gcp_lons, gcp_lats), axis=1
    )].reset_index(drop=True)

    if len(vessels) == 0:
        return 0

    all_cols = np.array([lonlat_to_pixel(v['longitude'], v['latitude'], col_c, row_c)[0]
                         for _, v in vessels.iterrows()])
    all_rows = np.array([lonlat_to_pixel(v['longitude'], v['latitude'], col_c, row_c)[1]
                         for _, v in vessels.iterrows()])
    all_lengths = vessels['length'].fillna(0).values

    scene_out = out_base / safe_name.replace('.SAFE', '')
    scene_out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    with rasterio.open(vv_files[0]) as src_vv:
        vh_src = rasterio.open(vh_files[0]) if vh_files else None
        try:
            for idx, row in vessels.iterrows():
                col = int(all_cols[idx])
                r   = int(all_rows[idx])
                if not (HALF <= col < W-HALF and HALF <= r < H-HALF):
                    continue
                chip_vv = extract_chip(src_vv, col, r)
                chip_vh = extract_chip(vh_src, col, r) if vh_src else None
                n_other, n_large, nearest_px = count_others_in_chip(
                    col, r, all_cols, all_rows, all_lengths, idx)
                out_path = scene_out / f"{int(row['mmsi'])}_{acq_time.strftime('%Y%m%dT%H%M%S')}.npz"
                save_dict = dict(
                    chip_vv=chip_vv,
                    mmsi=int(row['mmsi']),
                    vessel_name=str(row.get('vessel_name', '')),
                    vessel_type=int(row['vessel_type'] if pd.notna(row.get('vessel_type')) else -1),
                    length_m=float(row['length'] if pd.notna(row.get('length')) else 0),
                    lat=float(row['latitude']), lon=float(row['longitude']),
                    sog=float(row['sog'] if pd.notna(row.get('sog')) else 0),
                    heading=float(row['heading'] if pd.notna(row.get('heading')) else 0),
                    scene_id=safe_name, acq_time=acq_time.isoformat(),
                    pixel_col=col, pixel_row=r, dt_sec=float(row['dt_sec']),
                    n_other_ships=n_other, n_other_large=n_large,
                    nearest_large_px=nearest_px,
                    clean=bool(n_large <= CLEAN_MAX_LARGE),
                )
                if chip_vh is not None:
                    save_dict['chip_vh'] = chip_vh
                np.savez_compressed(str(out_path), **save_dict)
                n_saved += 1

            # Random background patches for backbone pre-training
            if backbone_out is not None and n_backbone > 0:
                backbone_out.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(abs(hash(safe_name)) % (2**31))
                extract_backbone_patches(src_vv, vh_src, W, H,
                                         all_cols, all_rows,
                                         backbone_out, safe_name.replace('.SAFE', ''),
                                         n_backbone, rng)
        finally:
            if vh_src:
                vh_src.close()

    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', required=True, choices=list(REGIONS.keys()))
    parser.add_argument('--delete-scenes', action='store_true',
                        help='Delete each .SAFE directory after chipping it')
    parser.add_argument('--backbone-patches', type=int, default=BACKBONE_PATCHES_DEFAULT,
                        help='Random background patches per scene for backbone pre-training (0 to disable)')
    args = parser.parse_args()

    r           = REGIONS[args.region]
    scene_dir   = ML_BASE / r['scene_dir']
    ais_dir     = ML_BASE / r['ais_dir']
    out_base    = ML_BASE / r['out_dir']
    backbone_out = ML_BASE / f"sar_backbone_patches" if args.backbone_patches > 0 else None
    out_base.mkdir(exist_ok=True)

    safe_dirs = sorted(scene_dir.glob('*.SAFE'))
    print(f"Region: {r['name']}")
    print(f"Scenes: {len(safe_dirs)}  |  AIS: {ais_dir}  |  Out: {out_base}")
    if backbone_out:
        print(f"Backbone patches: {args.backbone_patches}/scene  →  {backbone_out}")

    total = 0
    for safe_path in safe_dirs:
        out_dir = out_base / safe_path.stem
        if out_dir.exists() and len(list(out_dir.iterdir())) > 0:
            n = len(list(out_dir.iterdir()))
            print(f"  Already done {safe_path.name} ({n} chips) — skipping")
            total += n
            if args.delete_scenes and safe_path.exists():
                shutil.rmtree(safe_path)
            continue

        print(f"  {safe_path.name}...", end=' ', flush=True)
        n = process_scene(str(safe_path), ais_dir, out_base,
                          backbone_out=backbone_out, n_backbone=args.backbone_patches)
        print(f"{n} chips")
        total += n

        if args.delete_scenes and n >= 0:
            shutil.rmtree(safe_path)
            print(f"    Deleted scene")

    print(f"\nTotal chips: {total}")
    print(f"Saved to:    {out_base}")
    if backbone_out:
        n_backbone = len(list(backbone_out.glob('*.npz'))) if backbone_out.exists() else 0
        print(f"Backbone patches: {n_backbone}  →  {backbone_out}")


if __name__ == '__main__':
    main()
