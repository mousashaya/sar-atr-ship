"""
Download NOAA national AIS data and filter to multiple regions simultaneously.

2024 format: daily .zip containing a CSV with UPPERCASE column names.
2025+ format: daily .csv.zst already compressed, already snake_case columns.

Each day's file is downloaded once, split into per-region .csv.zst files,
then the raw file is discarded — minimal disk footprint.

Output directories (one per region):
    /Volumes/Machine Learning/ais_la/ais-YYYY-MM-DD.csv.zst
    /Volumes/Machine Learning/ais_ny/ais-YYYY-MM-DD.csv.zst
    /Volumes/Machine Learning/ais_chesapeake/ais-YYYY-MM-DD.csv.zst
    /Volumes/Machine Learning/ais_puget/ais-YYYY-MM-DD.csv.zst
    /Volumes/Machine Learning/ais_gulf/ais-YYYY-MM-DD.csv.zst

Usage:
    venv/bin/python scripts/download_ais_regions.py
    venv/bin/python scripts/download_ais_regions.py --start 2025-01-01 --end 2025-08-31
    venv/bin/python scripts/download_ais_regions.py --region Gulf --start 2025-01-01
"""

import argparse
import datetime
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
import zstandard as zstd

ML_BASE = Path("/Volumes/Machine Learning")

REGIONS = {
    'LA':         dict(name='LA/Long Beach',   out_dir='ais_la',
                       lat_min=33.0, lat_max=34.5,  lon_min=-119.0, lon_max=-117.0),
    'NY':         dict(name='NY/NJ Harbor',    out_dir='ais_ny',
                       lat_min=40.3, lat_max=41.0,  lon_min=-74.5,  lon_max=-73.5),
    'Chesapeake': dict(name='Chesapeake Bay',  out_dir='ais_chesapeake',
                       lat_min=36.8, lat_max=39.5,  lon_min=-77.0,  lon_max=-75.5),
    'Puget':      dict(name='Puget Sound',     out_dir='ais_puget',
                       lat_min=47.0, lat_max=48.8,  lon_min=-123.0, lon_max=-122.0),
    'Gulf':       dict(name='Gulf/Houston',    out_dir='ais_gulf',
                       lat_min=28.0, lat_max=32.0,  lon_min=-98.0,  lon_max=-93.5),
    'Houston':    dict(name='Houston Ship Channel', out_dir='ais_houston',
                       lat_min=29.5, lat_max=30.0,  lon_min=-95.4,  lon_max=-94.7),
    'SFBay':      dict(name='SF Bay/Oakland',  out_dir='ais_sfbay',
                       lat_min=37.5, lat_max=38.0,  lon_min=-122.5, lon_max=-121.5),
    'Savannah':   dict(name='Savannah/Brunswick', out_dir='ais_savannah',
                       lat_min=30.8, lat_max=32.2,  lon_min=-81.5,  lon_max=-80.5),
    'NewEngland': dict(name='New England', out_dir='ais_newengland',
                       lat_min=40.5, lat_max=43.0, lon_min=-71.5, lon_max=-68.5),
    'GulfEast': dict(name='Central Gulf LA/MS', out_dir='ais_gulfeast',
                       lat_min=28.5, lat_max=30.5, lon_min=-91.5, lon_max=-88.0),
    'Oregon': dict(name='Oregon/Columbia River', out_dir='ais_oregon',
                       lat_min=45.0, lat_max=47.0, lon_min=-125.0, lon_max=-123.5),
    'Hawaii': dict(name='Hawaii', out_dir='ais_hawaii',
                       lat_min=20.0, lat_max=22.5, lon_min=-159.0, lon_max=-156.0),
    'GulfOpen': dict(name='Gulf open water (paper box)', out_dir='ais_gulfopen',
                       lat_min=26.0, lat_max=28.5, lon_min=-96.5, lon_max=-88.0),
    'Miami':      dict(name='Miami/Port Everglades', out_dir='ais_miami',
                       lat_min=25.6, lat_max=26.0,  lon_min=-80.3,  lon_max=-80.0),
}

# 2024 column mapping (ZIP files with uppercase headers)
COL_MAP_2024 = {
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

KEEP_COLS = list(set(COL_MAP_2024.values()))

cctx = zstd.ZstdCompressor(level=3)


def save_zst(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cctx.compress(df.to_csv(index=False).encode()))


def filter_region(df, r):
    return df[
        (df['latitude']  >= r['lat_min']) & (df['latitude']  <= r['lat_max']) &
        (df['longitude'] >= r['lon_min']) & (df['longitude'] <= r['lon_max'])
    ].copy()


def process_day_2024(date_str, active_regions):
    d = datetime.date.fromisoformat(date_str)
    url = (f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/"
           f"AIS_{d.strftime('%Y_%m_%d')}.zip")
    resp = requests.get(url, timeout=180, stream=True)
    if resp.status_code == 404:
        return 'not available'
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(n for n in z.namelist() if n.endswith('.csv'))
        csv_bytes = z.read(csv_name)

    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df = df.rename(columns=COL_MAP_2024)
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep]
    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    done = []
    for key, r in active_regions.items():
        out = ML_BASE / r['out_dir'] / f"ais-{date_str}.csv.zst"
        if not out.exists():
            save_zst(filter_region(df, r), out)
            done.append(key)
    return done


def process_day_2025(date_str, active_regions):
    url = (f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2025/"
           f"ais-{date_str}.csv.zst")
    resp = requests.get(url, timeout=180, stream=True)
    if resp.status_code == 404:
        return 'not available'
    resp.raise_for_status()

    raw = resp.content
    dctx = zstd.ZstdDecompressor()
    csv_bytes = dctx.decompress(raw)

    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    done = []
    for key, r in active_regions.items():
        out = ML_BASE / r['out_dir'] / f"ais-{date_str}.csv.zst"
        if not out.exists():
            save_zst(filter_region(df, r), out)
            done.append(key)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',  default='2025-01-01')
    parser.add_argument('--end',    default='2025-08-31')
    parser.add_argument('--region', choices=list(REGIONS.keys()), default=None)
    args = parser.parse_args()

    active = {args.region: REGIONS[args.region]} if args.region else REGIONS

    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)
    dates = [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

    def needs_download(d):
        ds = d.isoformat()
        return any(not (ML_BASE / r['out_dir'] / f"ais-{ds}.csv.zst").exists()
                   for r in active.values())

    todo = [d for d in dates if needs_download(d)]
    print(f"Regions: {', '.join(active.keys())}")
    print(f"Days to download: {len(todo)} / {len(dates)}")

    for d in todo:
        date_str = d.isoformat()
        print(f"  {date_str}...", end=' ', flush=True)
        try:
            fn = process_day_2025 if d.year >= 2025 else process_day_2024
            result = fn(date_str, {k: v for k, v in active.items()
                                   if not (ML_BASE / v['out_dir'] / f"ais-{date_str}.csv.zst").exists()})
            if result == 'not available':
                print("not available")
            else:
                print(f"done ({', '.join(result)})")
        except Exception as e:
            print(f"ERROR — {e}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
