"""
scripts/download_slc_gulf.py
=============================
Download Sentinel-1 IW SLC scenes for the Gulf of Mexico.

SLC vs GRD:
  GRD: amplitude only, 10m, ~700MB/scene  (what we have now)
  SLC: complex (amplitude + phase), 5m, ~4GB/scene

Strategy: only download SLC for dates where we already have GRD ship chips —
avoids downloading 1TB+ blindly. Matches by acquisition date ± 0 days
(SLC and GRD are acquired simultaneously, same orbit pass).

Output: /Volumes/Machine Learning/sentinel1_slc_gulf/
  One .zip per scene — unzip manually or run with --unzip

Credentials: read from ~/.netrc  (machine urs.earthdata.nasa.gov)

Usage:
    # Dry run — show what would be downloaded
    venv/bin/python scripts/download_slc_gulf.py --dry-run

    # Download SLC for all dates with existing GRD chips (default)
    venv/bin/python scripts/download_slc_gulf.py

    # Download SLC for a specific date range
    venv/bin/python scripts/download_slc_gulf.py --start 2024-08-11 --end 2024-09-30

    # Limit to N scenes (useful for first test)
    venv/bin/python scripts/download_slc_gulf.py --max-scenes 5

    # Download + unzip
    venv/bin/python scripts/download_slc_gulf.py --unzip
"""

import argparse
import datetime
import netrc
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import asf_search as asf
import numpy as np
import requests

GRD_CHIP_DIRS = [
    Path('/Volumes/Machine Learning/sar_ship_chips'),
    Path('/Volumes/Machine Learning/sar_ship_chips_gulf'),
]
GRD_SCENE_DIR = Path('/Volumes/Machine Learning/sentinel1_gulf')
SLC_OUT_DIR   = Path('/Volumes/Machine Learning/sentinel1_slc_gulf')

# Open-water Gulf bbox (same tighter bounds as chip script)
LAT_MIN, LAT_MAX =  26.0, 28.5
LON_MIN, LON_MAX = -96.5, -88.0


def get_credentials():
    try:
        n = netrc.netrc()
        auth = n.authenticators('urs.earthdata.nasa.gov')
        if auth:
            return auth[0], auth[2]
    except Exception:
        pass
    raise RuntimeError(
        "No Earthdata credentials found in ~/.netrc\n"
        "Add:  machine urs.earthdata.nasa.gov login <user> password <pass>"
    )


def dates_with_chips():
    """Return set of YYYYMMDD dates that have existing GRD ship chips."""
    dates = set()
    for chip_dir in GRD_CHIP_DIRS:
        if not chip_dir.exists():
            continue
        for npz in chip_dir.glob('**/*.npz'):
            if npz.name == 'embeddings.npz':
                continue
            # scene dir name contains the acquisition date
            # e.g. S1A_IW_GRDH_1SDV_20240811T001831_...
            name = npz.parent.name
            parts = name.split('_')
            for p in parts:
                if len(p) == 15 and p[8] == 'T' and p[:8].isdigit():
                    dates.add(p[:8])
                    break
    return dates


def dates_from_range(start_str, end_str):
    start = datetime.date.fromisoformat(start_str)
    end   = datetime.date.fromisoformat(end_str)
    dates = set()
    d = start
    while d <= end:
        dates.add(d.strftime('%Y%m%d'))
        d += datetime.timedelta(days=1)
    return dates


def search_slc_scenes(dates, max_scenes=None):
    """Query ASF for SLC scenes matching dates and bbox."""
    print(f"Searching ASF for SLC scenes on {len(dates)} dates...")

    # Build date ranges — ASF search works best with explicit date ranges
    # Group consecutive dates to reduce API calls
    sorted_dates = sorted(dates)
    all_results = []

    # Search in monthly chunks to avoid overly broad queries
    by_month = defaultdict(list)
    for d in sorted_dates:
        by_month[d[:6]].append(d)

    for month, month_dates in sorted(by_month.items()):
        start_dt = datetime.datetime.strptime(min(month_dates), '%Y%m%d')
        end_dt   = datetime.datetime.strptime(max(month_dates), '%Y%m%d') + datetime.timedelta(days=1)

        results = asf.search(
            platform          = [asf.PLATFORM.SENTINEL1],
            processingLevel   = [asf.PRODUCT_TYPE.SLC],
            beamMode          = ['IW'],
            polarization      = ['VV+VH'],
            start             = start_dt.strftime('%Y-%m-%dT00:00:00Z'),
            end               = end_dt.strftime('%Y-%m-%dT23:59:59Z'),
            intersectsWith    = f'POLYGON(({LON_MIN} {LAT_MIN},{LON_MAX} {LAT_MIN},{LON_MAX} {LAT_MAX},{LON_MIN} {LAT_MAX},{LON_MIN} {LAT_MIN}))',
        )

        # Filter to only dates we actually want
        for r in results:
            scene_date = r.properties['startTime'][:10].replace('-', '')
            if scene_date in month_dates:
                all_results.append(r)

        print(f"  {month}: {len(results)} scenes found, {sum(1 for r in results if r.properties['startTime'][:10].replace('-','') in month_dates)} matching chip dates")

    # Deduplicate by scene name
    seen = set()
    unique = []
    for r in all_results:
        name = r.properties['sceneName']
        if name not in seen:
            seen.add(name)
            unique.append(r)

    print(f"Total unique SLC scenes: {len(unique)}")

    if max_scenes:
        unique = unique[:max_scenes]
        print(f"Limiting to {max_scenes} scenes")

    return unique


def download_scenes(scenes, username, password, dry_run=False, unzip=False, limit_rate=None, token=None):
    SLC_OUT_DIR.mkdir(parents=True, exist_ok=True)

    sizes_gb = [r.properties.get('bytes', 0) / 1e9 for r in scenes]
    total_gb = sum(sizes_gb)

    print(f"\n{'='*60}")
    print(f"  SLC Download Plan")
    print(f"  Scenes:     {len(scenes)}")
    print(f"  Total size: {total_gb:.1f} GB")
    print(f"  Output:     {SLC_OUT_DIR}")
    if limit_rate:
        print(f"  Rate limit: {limit_rate}/s")
    print(f"{'='*60}")

    if dry_run:
        return

    def make_session():
        s = requests.Session()
        if token:
            clean_token = token.replace('\n', '').replace('\r', '').replace(' ', '')
            s.headers.update({'Authorization': f'Bearer {clean_token}'})
        else:
            s.auth = (username, password)
        return s

    session = make_session()
    MAX_RETRIES = 5

    for i, r in enumerate(scenes, 1):
        name = r.properties['sceneName']
        date = r.properties['startTime'][:10]
        size = r.properties.get('bytes', 0) / 1e9
        url  = r.properties['url']
        out_zip = SLC_OUT_DIR / f"{name}.zip"

        if out_zip.exists():
            print(f"  [{i:3d}/{len(scenes)}] Already exists: {name}")
            continue

        print(f"  [{i:3d}/{len(scenes)}] {date}  {name}  ({size:.1f}GB)", flush=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with session.get(url, stream=True, timeout=300) as resp:
                    resp.raise_for_status()
                    with open(out_zip, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                            f.write(chunk)
                break  # success
            except Exception as e:
                if out_zip.exists():
                    out_zip.unlink()
                if attempt < MAX_RETRIES:
                    print(f"    Attempt {attempt} failed ({e}) — retrying with new session...")
                    session = make_session()  # re-authenticate
                else:
                    print(f"    WARNING: download failed after {MAX_RETRIES} attempts — {e}")

        if unzip and out_zip.exists():
            print(f"    Unzipping {name}...")
            subprocess.run(['unzip', '-q', '-n', str(out_zip),
                           '-d', str(SLC_OUT_DIR)])

    print(f"\nDone. SLC scenes in {SLC_OUT_DIR}")
    if not unzip:
        print(f"To unzip: find '{SLC_OUT_DIR}' -name '*.zip' | "
              f"xargs -P 4 -I{{}} unzip -q -n {{}} -d '{SLC_OUT_DIR}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',    action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--start',      type=str, default=None,
                        help='Start date YYYY-MM-DD (overrides chip-date matching)')
    parser.add_argument('--end',        type=str, default=None,
                        help='End date YYYY-MM-DD')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Limit number of scenes (useful for testing)')
    parser.add_argument('--unzip',      action='store_true',
                        help='Unzip each scene after download')
    parser.add_argument('--limit-rate', type=str, default=None,
                        help='Max download speed e.g. 10M, 50M, 100M (default: unlimited)')
    parser.add_argument('--token', type=str, default=None,
                        help='Earthdata bearer token (from urs.earthdata.nasa.gov)')
    args = parser.parse_args()

    username, password = get_credentials()
    print(f"Earthdata user: {username}")

    # Determine which dates to download
    if args.start and args.end:
        dates = dates_from_range(args.start, args.end)
        print(f"Date range: {args.start} → {args.end} ({len(dates)} days)")
    else:
        dates = dates_with_chips()
        print(f"Found {len(dates)} dates with existing GRD ship chips")

    if not dates:
        print("No dates found. Use --start/--end to specify a range.")
        sys.exit(1)

    scenes = search_slc_scenes(dates, max_scenes=args.max_scenes)

    if not scenes:
        print("No SLC scenes found matching criteria.")
        sys.exit(1)

    download_scenes(scenes, username, password,
                    dry_run=args.dry_run, unzip=args.unzip,
                    limit_rate=args.limit_rate, token=args.token)


if __name__ == '__main__':
    main()
