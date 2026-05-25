"""
SAR ship chipping pipeline — Gulf of Mexico SLC scenes (amplitude, native resolution).

For each Sentinel-1 IW SLC scene:
  1. Parse acquisition time from filename
  2. Load matching AIS file (±10 min window)
  3. Filter AIS to vessels ≥50m within scene bounds
  4. For each vessel: find containing sub-swath (iw1/iw2/iw3)
  5. Compute pixel coordinates via degree-2 polynomial GCP fit per sub-swath
  6. Read complex data, compute amplitude (no multi-looking)
  7. Extract patch covering 1km × 1km physical extent (~429 range × ~72 az pixels)
  8. Resize patch to 512×512 with bilinear interpolation
  9. Flag contamination

Pixel spacing (IW SLC): ~2.33m range × ~14m azimuth.
Chip footprint: 1km × 1km. No multi-looking — preserves native range resolution.
Resize to 512×512 for compatibility with re-ID models.

Output: /Volumes/Machine Learning/sar_ship_chips_slc_gulf/<scene_id>/<MMSI>_<acq_time>.npz
Each file contains:
  chip_vv, chip_vh              (512, 512) float32 amplitude (bilinear resize from native SLC)
  mmsi, vessel_name, vessel_type, length_m, lat, lon, sog, heading
  scene_id, acq_time, pixel_col, pixel_row, dt_sec, subswath
  range_px_spacing, az_px_spacing  (meters, from annotation XML)
  n_other_ships, n_other_large, nearest_large_px, clean
"""

import os
import re
import glob
import datetime
import io
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import zstandard as zstd
import rasterio
from scipy.ndimage import zoom as nd_zoom
from multiprocessing import Pool, cpu_count

N_WORKERS = 1  # SLC subswaths are ~500MB-1GB each; loading all upfront × workers exceeds RAM

OUTPUT_SIZE = 512
PATCH_M = 1000.0   # 1km × 1km physical chip footprint

AIS_DIR = "/Volumes/Machine Learning/ais_noaa"
SCENE_DIR = "/Volumes/Machine Learning/sentinel1_slc_gulf"
OUT_DIR = "/Volumes/Machine Learning/sar_ship_chips_slc_gulf"
os.makedirs(OUT_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 24.5, 29.5
LON_MIN, LON_MAX = -96.5, -88.0
MIN_LENGTH = 50
LARGE_THRESHOLD = 150
CLEAN_MAX_LARGE = 1

SUBSWATHS = ["iw1", "iw2", "iw3"]

# Fallback pixel spacings if annotation XML is missing (IW SLC nominal values)
DEFAULT_RANGE_SPACING = 2.3296
DEFAULT_AZ_SPACING = 13.968


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


def read_pixel_spacing(annotation_xml):
    """Parse range/azimuth pixel spacing from sub-swath annotation XML."""
    try:
        root = ET.parse(annotation_xml).getroot()
        r = float(root.find('.//rangePixelSpacing').text)
        a = float(root.find('.//azimuthPixelSpacing').text)
        return r, a
    except Exception:
        return DEFAULT_RANGE_SPACING, DEFAULT_AZ_SPACING


def build_gcp_transform(gcps):
    """Degree-2 polynomial fit: (lon, lat) -> (col, row)."""
    lons = np.array([g.x for g in gcps])
    lats = np.array([g.y for g in gcps])
    cols = np.array([g.col for g in gcps])
    rows = np.array([g.row for g in gcps])
    A = np.column_stack([np.ones(len(lons)), lons, lats, lons * lats, lons**2, lats**2])
    col_c, _, _, _ = np.linalg.lstsq(A, cols, rcond=None)
    row_c, _, _, _ = np.linalg.lstsq(A, rows, rcond=None)
    return col_c, row_c


def lonlat_to_pixel(lon, lat, col_c, row_c):
    pt = np.array([1, lon, lat, lon * lat, lon**2, lat**2])
    return int(pt @ col_c), int(pt @ row_c)


def in_bounds(lon, lat, gcp_lons, gcp_lats):
    return (gcp_lons.min() <= lon <= gcp_lons.max() and
            gcp_lats.min() <= lat <= gcp_lats.max())


def find_subswath(lon, lat, swath_gcps):
    """Return (subswath_name, gcps) for the sub-swath containing (lon, lat)."""
    for sw in ["iw2", "iw1", "iw3"]:
        if sw not in swath_gcps:
            continue
        gcps = swath_gcps[sw]
        lons = np.array([g.x for g in gcps])
        lats = np.array([g.y for g in gcps])
        if in_bounds(lon, lat, lons, lats):
            return sw, gcps
    return None, None


def extract_and_resize(amp, col, row, half_range, half_az):
    """Extract native-resolution patch and resize to OUTPUT_SIZE × OUTPUT_SIZE."""
    H, W = amp.shape
    r0 = max(0, row - half_az)
    r1 = min(H, row + half_az)
    c0 = max(0, col - half_range)
    c1 = min(W, col + half_range)
    raw = amp[r0:r1, c0:c1]

    patch_h = half_az * 2
    patch_w = half_range * 2
    patch = np.zeros((patch_h, patch_w), dtype=np.float32)
    dr = half_az - (row - r0)
    dc = half_range - (col - c0)
    patch[dr:dr + raw.shape[0], dc:dc + raw.shape[1]] = raw

    zoom_r = OUTPUT_SIZE / patch_h
    zoom_c = OUTPUT_SIZE / patch_w
    chip = nd_zoom(patch, (zoom_r, zoom_c), order=1).astype(np.float32)
    return chip


def read_chip_windowed(tiff_path, col, row, half_range, half_az):
    """Open a subswath tiff, read only the chip window, return amplitude float32.
    Never loads the full array — rasterio reads only the requested pixels."""
    with rasterio.open(tiff_path) as src:
        H, W = src.height, src.width
        r0 = max(0, row - half_az)
        r1 = min(H, row + half_az)
        c0 = max(0, col - half_range)
        c1 = min(W, col + half_range)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        raw = src.read(1, window=win)

    if np.iscomplexobj(raw):
        raw = np.abs(raw)
    raw = raw.astype(np.float32)

    patch_h = half_az * 2
    patch_w = half_range * 2
    patch = np.zeros((patch_h, patch_w), dtype=np.float32)
    dr = half_az - (row - r0)
    dc = half_range - (col - c0)
    patch[dr:dr + raw.shape[0], dc:dc + raw.shape[1]] = raw

    zoom_r = OUTPUT_SIZE / patch_h
    zoom_c = OUTPUT_SIZE / patch_w
    return nd_zoom(patch, (zoom_r, zoom_c), order=1).astype(np.float32)


def count_others_in_chip(target_col, target_row, cols, rows, lengths, half_range, half_az):
    n_any = 0
    n_large = 0
    min_dist_px = np.inf
    for c, r, length in zip(cols, rows, lengths):
        if abs(c - target_col) <= half_range and abs(r - target_row) <= half_az:
            n_any += 1
            if length >= LARGE_THRESHOLD:
                n_large += 1
        if length >= LARGE_THRESHOLD:
            dist = np.sqrt((c - target_col)**2 + (r - target_row)**2)
            min_dist_px = min(min_dist_px, dist)
    return n_any, n_large, (float(min_dist_px) if min_dist_px < np.inf else -1.0)


def get_swath_dims(tiff_path):
    """Return (height, width) without loading any data."""
    with rasterio.open(tiff_path) as src:
        return src.height, src.width


def process_scene(safe_path):
    safe_name = os.path.basename(safe_path)
    acq_time = parse_acq_time(safe_name)
    date_str = acq_time.strftime('%Y-%m-%d')

    # Collect per-sub-swath file paths and pixel spacings — no data loaded yet
    swath_vv = {}
    swath_vh = {}
    swath_spacing = {}
    swath_gcps = {}
    swath_dims = {}   # sw -> (H, W)
    for sw in SUBSWATHS:
        vv = glob.glob(os.path.join(safe_path, 'measurement', f'*-{sw}-slc-vv-*.tiff'))
        vh = glob.glob(os.path.join(safe_path, 'measurement', f'*-{sw}-slc-vh-*.tiff'))
        ann = glob.glob(os.path.join(safe_path, 'annotation', f'*-{sw}-slc-vv-*.xml'))
        if not vv:
            continue
        swath_vv[sw] = vv[0]
        if vh:
            swath_vh[sw] = vh[0]
        r_sp, a_sp = read_pixel_spacing(ann[0]) if ann else (DEFAULT_RANGE_SPACING, DEFAULT_AZ_SPACING)
        swath_spacing[sw] = (r_sp, a_sp)
        # Read GCPs and scene dimensions without loading pixel data
        try:
            with rasterio.open(vv[0]) as src:
                gcps, _ = src.gcps
                swath_gcps[sw] = gcps
                swath_dims[sw] = (src.height, src.width)
        except Exception as e:
            print(f"  Warning: {sw} GCP read failed: {e}", flush=True)

    if not swath_gcps:
        print(f"  No VV files in {safe_name}")
        return 0

    ais = load_ais_window(date_str, acq_time)
    if ais is None or len(ais) == 0:
        print(f"  No AIS data for {date_str}", flush=True)
        return 0

    ais['dt_sec'] = (ais['base_date_time'] - pd.Timestamp(acq_time)).abs().dt.total_seconds()
    vessel_best = ais.sort_values('dt_sec').groupby('mmsi').first().reset_index()
    vessel_best = vessel_best[vessel_best['length'].fillna(0) >= MIN_LENGTH].copy()

    def in_any_swath(lon, lat):
        for gcps in swath_gcps.values():
            lons = np.array([g.x for g in gcps])
            lats = np.array([g.y for g in gcps])
            if in_bounds(lon, lat, lons, lats):
                return True
        return False

    vessel_best = vessel_best[
        vessel_best.apply(lambda r: in_any_swath(r['longitude'], r['latitude']), axis=1)
    ].reset_index(drop=True)

    if len(vessel_best) == 0:
        return 0

    sw_transforms = {sw: build_gcp_transform(gcps) for sw, gcps in swath_gcps.items()}

    vessel_sw, vessel_cols, vessel_rows = [], [], []
    for _, v in vessel_best.iterrows():
        sw, _ = find_subswath(v['longitude'], v['latitude'], swath_gcps)
        if sw is None:
            vessel_sw.append(None); vessel_cols.append(-1); vessel_rows.append(-1)
            continue
        col_c, row_c = sw_transforms[sw]
        col, row = lonlat_to_pixel(v['longitude'], v['latitude'], col_c, row_c)
        vessel_sw.append(sw); vessel_cols.append(col); vessel_rows.append(row)

    scene_out_dir = os.path.join(OUT_DIR, safe_name.replace('.SAFE', ''))
    os.makedirs(scene_out_dir, exist_ok=True)

    all_lengths = vessel_best['length'].fillna(0).values
    n_saved = 0

    for idx, row in vessel_best.iterrows():
        sw = vessel_sw[idx]
        col = vessel_cols[idx]
        r = vessel_rows[idx]

        if sw is None or col < 0 or sw not in swath_vv:
            continue

        r_sp, a_sp = swath_spacing[sw]
        half_range = int(round(PATCH_M / (2 * r_sp)))
        half_az = int(round(PATCH_M / (2 * a_sp)))

        H, W = swath_dims[sw]
        if not (half_az <= r < H - half_az and half_range <= col < W - half_range):
            continue

        # Windowed read — only the chip region is loaded into memory
        chip_vv = read_chip_windowed(swath_vv[sw], col, r, half_range, half_az)

        chip_vh = None
        if sw in swath_vh:
            chip_vh = read_chip_windowed(swath_vh[sw], col, r, half_range, half_az)

        same_sw_idxs = [i for i, s in enumerate(vessel_sw) if s == sw and i != idx]
        n_other, n_other_large, nearest_large_px = count_others_in_chip(
            col, r,
            [vessel_cols[i] for i in same_sw_idxs],
            [vessel_rows[i] for i in same_sw_idxs],
            [all_lengths[i] for i in same_sw_idxs],
            half_range, half_az,
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
            subswath=sw,
            range_px_spacing=r_sp,
            az_px_spacing=a_sp,
            n_other_ships=n_other,
            n_other_large=n_other_large,
            nearest_large_px=nearest_large_px,
            clean=bool(n_other_large <= CLEAN_MAX_LARGE),
        )
        if chip_vh is not None:
            save_dict['chip_vh'] = chip_vh

        np.savez_compressed(out_path, **save_dict)
        n_saved += 1

    return n_saved


def process_scene_safe(safe_path):
    safe_name = os.path.basename(safe_path)
    out_dir = os.path.join(OUT_DIR, safe_name.replace('.SAFE', ''))
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 10:
        n = len(os.listdir(out_dir))
        print(f"Already done {safe_name} ({n} chips) — skipping", flush=True)
        return n
    print(f"Processing {safe_name}...", flush=True)
    try:
        n = process_scene(safe_path)
    except Exception as e:
        print(f"  ERROR in {safe_name}: {e}", flush=True)
        return 0
    print(f"  Saved {n} chips — {safe_name}", flush=True)
    return n


def main():
    all_safe = sorted(glob.glob(os.path.join(SCENE_DIR, '**/*.SAFE'), recursive=True))
    seen = {}
    for path in all_safe:
        name = os.path.basename(path)
        if name not in seen:
            seen[name] = path
    safe_dirs = [p for name, p in sorted(seen.items())]
    print(f"Found {len(safe_dirs)} SLC SAFE directories — using {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:
        results = pool.map(process_scene_safe, safe_dirs)

    total = sum(results)
    print(f"\nTotal chips: {total}")


if __name__ == '__main__':
    main()
