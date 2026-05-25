"""
Download and process NOAA national AIS data for the Gulf of Mexico / Houston area.

Source: marinecadastre.gov daily national AIS files
URL pattern: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_09_01.zip

Steps per day:
  1. Download zip (if not already on disk)
  2. Read CSV from zip
  3. Rename columns to match chip_ships_sar.py expectations
  4. Filter to Gulf + Houston bbox
  5. Save as ais-YYYY-MM-DD.csv.zst

Usage:
    python scripts/download_ais_noaa.py                        # Sep 2 - Dec 31, 2024
    python scripts/download_ais_noaa.py --start 2024-01-01     # full year from Jan
    python scripts/download_ais_noaa.py --process-zips-only    # process existing Jan zips
"""

import os
import io
import argparse
import datetime
import zipfile
import requests
import pandas as pd
import zstandard as zstd
from pathlib import Path

AIS_DIR = Path("/Volumes/Machine Learning/ais_noaa")
BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024"

# Broad bbox covering Gulf shipping lanes + Houston Ship Channel
LAT_MIN, LAT_MAX = 25.0, 33.0
LON_MIN, LON_MAX = -100.0, -87.0

# Column rename: NOAA raw → chip_ships_sar.py format
COL_MAP = {
    'MMSI':         'mmsi',
    'BaseDateTime': 'base_date_time',
    'LAT':          'latitude',
    'LON':          'longitude',
    'SOG':          'sog',
    'COG':          'cog',
    'Heading':      'heading',
    'VesselName':   'vessel_name',
    'VesselType':   'vessel_type',
    'Length':       'length',
    'Width':        'width',
    'IMO':          'imo',
    'Status':       'status',
}


def process_csv_bytes(csv_bytes, date_str):
    """Read raw AIS CSV bytes, filter to bbox, return filtered DataFrame."""
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df = df.rename(columns=COL_MAP)
    df = df[df.columns.intersection(list(COL_MAP.values()))]

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    mask = (
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    )
    df = df[mask].copy()
    df['base_date_time'] = pd.to_datetime(df['base_date_time'], errors='coerce')
    return df


def save_zst(df, out_path):
    csv_bytes = df.to_csv(index=False).encode()
    cctx = zstd.ZstdCompressor(level=3)
    with open(out_path, 'wb') as f:
        f.write(cctx.compress(csv_bytes))


def process_zip_file(zip_path, date_str, out_path):
    """Process an already-downloaded zip file."""
    with zipfile.ZipFile(zip_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
        csv_bytes = z.read(csv_name)
    df = process_csv_bytes(csv_bytes, date_str)
    save_zst(df, out_path)
    return len(df)


def download_and_process(date, out_path):
    """Download zip from marinecadastre, process, save .csv.zst."""
    date_str = date.strftime('%Y_%m_%d')
    url = f"{BASE_URL}/AIS_{date_str}.zip"

    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code == 404:
            print(f"  {date_str}: not available (404) — skipping")
            return 0
        r.raise_for_status()

        zip_bytes = r.content
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
            csv_bytes = z.read(csv_name)

        df = process_csv_bytes(csv_bytes, date_str)
        save_zst(df, out_path)
        return len(df)

    except Exception as e:
        print(f"  {date_str}: ERROR — {e}")
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-09-02', help='Start date YYYY-MM-DD')
    parser.add_argument('--end',   default='2024-12-31', help='End date YYYY-MM-DD')
    parser.add_argument('--process-zips-only', action='store_true',
                        help='Only process existing downloaded zip files (no downloads)')
    args = parser.parse_args()

    AIS_DIR.mkdir(exist_ok=True)

    # First: process any existing raw zips not yet converted
    raw_zips = sorted(AIS_DIR.glob('AIS_*.zip'))
    if raw_zips:
        print(f"Processing {len(raw_zips)} existing raw zip files...")
        for zip_path in raw_zips:
            name = zip_path.stem  # e.g. AIS_2024_01_01
            parts = name.split('_')
            date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
            out_path = AIS_DIR / f"ais-{date_str}.csv.zst"
            if out_path.exists():
                print(f"  {date_str}: already processed — skipping")
                continue
            print(f"  Processing {zip_path.name}...", end=' ', flush=True)
            n = process_zip_file(zip_path, date_str, out_path)
            print(f"{n:,} rows")

    if args.process_zips_only:
        print("Done (zip processing only).")
        return

    # Download missing dates
    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

    missing = []
    for d in dates:
        out_path = AIS_DIR / f"ais-{d.isoformat()}.csv.zst"
        if not out_path.exists():
            missing.append(d)

    print(f"\nDownloading {len(missing)} missing days ({args.start} → {args.end})...")

    for d in missing:
        out_path = AIS_DIR / f"ais-{d.isoformat()}.csv.zst"
        print(f"  {d.isoformat()}...", end=' ', flush=True)
        n = download_and_process(d, out_path)
        if n > 0:
            print(f"{n:,} rows in bbox")
        elif n == 0:
            print("skipped")

    print("\nDone.")


if __name__ == '__main__':
    main()
