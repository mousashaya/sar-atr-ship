"""
Download Sentinel-1 IW GRD scenes for high-traffic maritime regions.
Uses ASF search API + streaming download with resume support.

Regions: LA/Long Beach, NY/NJ Harbor, Chesapeake Bay, Puget Sound, Norwegian Coast

Usage:
    venv/bin/python scripts/download_new_regions.py --dry-run   # check counts
    venv/bin/python scripts/download_new_regions.py             # download all
    venv/bin/python scripts/download_new_regions.py --region LA # single region
"""

import argparse
import netrc
import sys
import urllib3
from pathlib import Path

import asf_search as asf
import time
asf.constants.INTERNAL.CMR_TIMEOUT = 300
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGIONS = {
    'LA':       dict(name='LA/Long Beach',  lat_min=33.0, lat_max=34.5, lon_min=-119.0, lon_max=-117.0),
    'NY':       dict(name='NY/NJ Harbor',   lat_min=40.3, lat_max=41.0, lon_min=-74.5,  lon_max=-73.5),
    'Chesapeake': dict(name='Chesapeake Bay', lat_min=36.8, lat_max=39.5, lon_min=-77.0, lon_max=-75.5),
    'Puget':    dict(name='Puget Sound',    lat_min=47.0, lat_max=48.8, lon_min=-123.0, lon_max=-122.0),
    'NewEngland': dict(name='New England', lat_min=40.5, lat_max=43.0, lon_min=-71.5, lon_max=-68.5),
    'GulfEast': dict(name='Central Gulf LA/MS', lat_min=28.5, lat_max=30.5, lon_min=-91.5, lon_max=-88.0),
    'Oregon': dict(name='Oregon/Columbia River', lat_min=45.0, lat_max=47.0, lon_min=-125.0, lon_max=-123.5),
    'Hawaii': dict(name='Hawaii', lat_min=20.0, lat_max=22.5, lon_min=-159.0, lon_max=-156.0),
    'GulfOpen': dict(name='Gulf open water (paper box)', lat_min=26.0, lat_max=28.5, lon_min=-96.5, lon_max=-88.0),
    'Norway':   dict(name='Norwegian Coast', lat_min=58.0, lat_max=62.0, lon_min=4.0,   lon_max=8.0),
}

START = '2024-01-01T00:00:00Z'
END   = '2024-12-31T23:59:59Z'
OUT_BASE = Path('/Volumes/Machine Learning')


def bbox_wkt(r):
    return (f"POLYGON(({r['lon_min']} {r['lat_min']},"
            f"{r['lon_max']} {r['lat_min']},"
            f"{r['lon_max']} {r['lat_max']},"
            f"{r['lon_min']} {r['lat_max']},"
            f"{r['lon_min']} {r['lat_min']}))")


def get_credentials():
    try:
        n = netrc.netrc()
        auth = n.authenticators('urs.earthdata.nasa.gov')
        if auth:
            return auth[0], auth[2]
    except Exception:
        pass
    raise RuntimeError("No Earthdata credentials in ~/.netrc")


def search_region(key, r):
    print(f"  Searching {r['name']}...", flush=True)
    for attempt in range(5):
        try:
            results = _search(r)
            break
        except Exception as e:
            print(f"    search failed ({e.__class__.__name__}), retry {attempt+1}/5 in 60 s", flush=True)
            time.sleep(60)
    else:
        raise RuntimeError('ASF search failed 5 times')
    return _dedupe(results)


def _search(r):
    return asf.search(
        platform        = [asf.PLATFORM.SENTINEL1],
        processingLevel = [asf.PRODUCT_TYPE.GRD_HD],
        beamMode        = ['IW'],
        polarization    = ['VV+VH'],
        start           = START,
        end             = END,
        intersectsWith  = bbox_wkt(r),
    )


def _dedupe(results):
    seen = set()
    unique = []
    for s in results:
        n = s.properties['sceneName']
        if n not in seen:
            seen.add(n)
            unique.append(s)
    return unique


def download_scenes(scenes, out_dir, username, password, dry_run=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    total_gb = len(scenes) * 0.8
    print(f"  {len(scenes)} scenes  (~{total_gb:.0f} GB)  → {out_dir}")
    if dry_run:
        return

    session = requests.Session()
    session.auth = (username, password)
    session.verify = False

    for i, r in enumerate(scenes, 1):
        name = r.properties['sceneName']
        url  = r.properties['url']
        out_zip = out_dir / f"{name}.zip"
        out_safe = out_dir / f"{name}.SAFE"

        if out_safe.exists() or out_zip.exists():
            print(f"  [{i:3d}/{len(scenes)}] Already exists: {name}")
            continue

        date = r.properties['startTime'][:10]
        size = r.properties.get('bytes', 0) / 1e9
        print(f"  [{i:3d}/{len(scenes)}] {date}  ({size:.1f}GB)  {name}", flush=True)

        for attempt in range(1, 6):
            try:
                with session.get(url, stream=True, timeout=300, verify=False) as resp:
                    resp.raise_for_status()
                    with open(out_zip, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                            f.write(chunk)
                # unzip in place
                import subprocess
                subprocess.run(['unzip', '-q', '-n', str(out_zip), '-d', str(out_dir)],
                               check=True)
                out_zip.unlink()
                break
            except Exception as e:
                if out_zip.exists():
                    out_zip.unlink()
                if attempt < 5:
                    print(f"    Attempt {attempt} failed ({e}) — retrying...")
                    session = requests.Session()
                    session.auth = (username, password)
                    session.verify = False
                else:
                    print(f"    WARNING: failed after 5 attempts — {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Count scenes without downloading')
    parser.add_argument('--region',  choices=list(REGIONS.keys()), default=None,
                        help='Download one region only')
    parser.add_argument('--start', default=None, help='e.g. 2024-02-01 (default: all of 2024)')
    parser.add_argument('--end',   default=None, help='e.g. 2024-08-31')
    parser.add_argument('--out-dir', default=None, help='override scene folder name under the ML volume')
    args = parser.parse_args()
    global START, END
    if args.start: START = f'{args.start}T00:00:00Z'
    if args.end:   END   = f'{args.end}T23:59:59Z'

    username, password = get_credentials()
    print(f"Earthdata user: {username}")

    targets = {args.region: REGIONS[args.region]} if args.region else REGIONS

    print(f"\nSentinel-1 IW GRD Download — 2024")
    print(f"{'='*60}")

    total = 0
    for key, r in targets.items():
        scenes = search_region(key, r)
        out_dir = OUT_BASE / (args.out_dir or f"sentinel1_{key.lower()}")
        download_scenes(scenes, out_dir, username, password, dry_run=args.dry_run)
        total += len(scenes)
        print()

    print(f"{'='*60}")
    if args.dry_run:
        print(f"Total: {total} scenes (~{total * 0.8:.0f} GB) — dry run, nothing downloaded")
    else:
        print(f"All downloads complete.")


if __name__ == '__main__':
    main()
