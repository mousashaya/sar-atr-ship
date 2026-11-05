"""
Search ASF for Sentinel-1 IW GRD scenes over high-traffic maritime regions.
Dry-run only — prints scene counts and estimated download size, no downloads.

Usage:
    venv/bin/python scripts/search_new_regions.py
    venv/bin/python scripts/search_new_regions.py --start 2024-01-01 --end 2024-12-31
"""

import argparse
import datetime
from pathlib import Path

import asf_search as asf

REGIONS = {
    'LA/Long Beach':   dict(lat_min=33.0, lat_max=34.5, lon_min=-119.0, lon_max=-117.0),
    'NY/NJ Harbor':    dict(lat_min=40.3, lat_max=41.0, lon_min=-74.5,  lon_max=-73.5),
    'Chesapeake Bay':  dict(lat_min=36.8, lat_max=39.5, lon_min=-77.0,  lon_max=-75.5),
    'Puget Sound':     dict(lat_min=47.0, lat_max=48.8, lon_min=-123.0, lon_max=-122.0),
    'Norwegian Coast': dict(lat_min=58.0, lat_max=62.0, lon_min=4.0,    lon_max=8.0),
}

GRD_SCENE_SIZE_GB = 0.8  # approximate per scene


def bbox_wkt(r):
    return (f"POLYGON(({r['lon_min']} {r['lat_min']},"
            f"{r['lon_max']} {r['lat_min']},"
            f"{r['lon_max']} {r['lat_max']},"
            f"{r['lon_min']} {r['lat_max']},"
            f"{r['lon_min']} {r['lat_min']}))")


def search_region(name, bbox, start, end):
    print(f"  Searching {name}...", flush=True)
    try:
        results = asf.search(
            platform        = [asf.PLATFORM.SENTINEL1],
            processingLevel = [asf.PRODUCT_TYPE.GRD_HD],
            beamMode        = ['IW'],
            polarization    = ['VV+VH'],
            start           = start,
            end             = end,
            intersectsWith  = bbox_wkt(bbox),
        )
        scenes = list(results)
        # Deduplicate by scene name
        seen = set()
        unique = []
        for r in scenes:
            n = r.properties['sceneName']
            if n not in seen:
                seen.add(n)
                unique.append(r)
        return unique
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-01-01',
                        help='Start date YYYY-MM-DD (default: 2024-01-01)')
    parser.add_argument('--end',   default='2024-12-31',
                        help='End date YYYY-MM-DD (default: 2024-12-31)')
    args = parser.parse_args()

    start = f"{args.start}T00:00:00Z"
    end   = f"{args.end}T23:59:59Z"

    print(f"\nASF Scene Search — Sentinel-1 IW GRD")
    print(f"Date range: {args.start} → {args.end}")
    print(f"{'='*60}")

    total_scenes = 0
    for name, bbox in REGIONS.items():
        scenes = search_region(name, bbox, start, end)
        size_gb = len(scenes) * GRD_SCENE_SIZE_GB
        print(f"  {name:<20s}: {len(scenes):4d} scenes  (~{size_gb:.0f} GB)")
        total_scenes += len(scenes)

    print(f"{'='*60}")
    print(f"  {'TOTAL':<20s}: {total_scenes:4d} scenes  (~{total_scenes * GRD_SCENE_SIZE_GB:.0f} GB)")
    print(f"\nNote: AIS data for all US regions already on disk (full year 2024).")
    print(f"      Norwegian coast requires BarentsWatch AIS download.")
    print(f"      Drive free space: ~2.1 TB available.")

if __name__ == '__main__':
    main()
