"""Quick test: chip one SLC scene and print diagnostics."""

import glob
import numpy as np
import rasterio
from chip_ships_slc_gulf import process_scene, SCENE_DIR, AIS_DIR, parse_acq_time

safe_dirs = sorted(glob.glob(f"{SCENE_DIR}/**/*.SAFE", recursive=True))
if not safe_dirs:
    print("No SAFE dirs found")
    exit(1)

safe = safe_dirs[0]
print(f"Testing: {safe}\n")

# Check that rasterio reads the TIFF as complex
import os
tiffs = glob.glob(os.path.join(safe, "measurement", "*-iw2-slc-vv-*.tiff"))
if not tiffs:
    tiffs = glob.glob(os.path.join(safe, "measurement", "*-slc-vv-*.tiff"))
if tiffs:
    with rasterio.open(tiffs[0]) as src:
        print(f"TIFF dtype : {src.dtypes}")
        print(f"TIFF shape : {src.height} rows × {src.width} cols")
        print(f"Num GCPs   : {len(src.gcps[0])}")
        sample = src.read(1, out_shape=(1, 64, 64))
        print(f"Sample dtype after read: {sample.dtype}")
        print(f"Is complex : {np.iscomplexobj(sample)}")
        if np.iscomplexobj(sample):
            amp = np.abs(sample)
            print(f"Amplitude range: {amp.min():.1f} – {amp.max():.1f}")
else:
    print("No VV TIFF found")

print()

# Check scene footprint via GCPs
if tiffs:
    with rasterio.open(tiffs[0]) as src:
        gcps, _ = src.gcps
        lons = [g.x for g in gcps]
        lats = [g.y for g in gcps]
        print(f"Scene lon : {min(lons):.2f} – {max(lons):.2f}")
        print(f"Scene lat : {min(lats):.2f} – {max(lats):.2f}")

# Check AIS file exists
import datetime, os
acq = parse_acq_time(os.path.basename(safe))
ais_path = os.path.join(AIS_DIR, f"ais-{acq.strftime('%Y-%m-%d')}.csv.zst")
print(f"AIS date  : {acq.strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"AIS file  : {'EXISTS' if os.path.exists(ais_path) else 'MISSING'} — {ais_path}")

print()
n = process_scene(safe)
print(f"\nChips saved: {n}")
