"""
Search Umbra Open Data STAC catalog for SAR scenes over maritime regions.
No authentication needed — open data is public on S3.

Finds scenes over LA/Long Beach, NY/NJ Harbor, Norwegian Coast, and Gulf.
Downloads GEC.tif files via sar-data/task-data/{task_id}/{collect}/_GEC.tif.

Usage:
    venv/bin/python scripts/search_umbra_opendata.py           # search only
    venv/bin/python scripts/search_umbra_opendata.py --download # download GEC images
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pystac
import requests

CATALOG_URL = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/stac/catalog.json"
BUCKET_URL  = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog"
OUT_BASE    = Path("/Volumes/Machine Learning/umbra_open_data")

REGIONS = {
    'LA':         dict(name='LA/Long Beach',   lon_min=-119.0, lat_min=33.0,  lon_max=-117.0, lat_max=34.5),
    'NY':         dict(name='NY/NJ Harbor',    lon_min=-74.5,  lat_min=40.3,  lon_max=-73.5,  lat_max=41.0),
    'Gulf':       dict(name='Gulf/Houston',    lon_min=-98.0,  lat_min=28.0,  lon_max=-93.5,  lat_max=32.0),
    'Houston':    dict(name='Houston Ship Channel', lon_min=-95.4, lat_min=29.5, lon_max=-94.7, lat_max=30.0),
    'Norway':     dict(name='Norwegian Coast', lon_min=4.0,    lat_min=58.0,  lon_max=8.0,    lat_max=62.0),
    'Chesapeake': dict(name='Chesapeake Bay',  lon_min=-77.0,  lat_min=36.8,  lon_max=-75.5,  lat_max=39.5),
    'Puget':      dict(name='Puget Sound',     lon_min=-123.0, lat_min=47.0,  lon_max=-122.0, lat_max=48.8),
    'SFBay':      dict(name='SF Bay/Oakland',  lon_min=-122.5, lat_min=37.5,  lon_max=-121.5, lat_max=38.0),
    'Savannah':   dict(name='Savannah/Brunswick', lon_min=-81.5, lat_min=30.8, lon_max=-80.5, lat_max=32.2),
    'Miami':      dict(name='Miami/Port Everglades', lon_min=-80.3, lat_min=25.6, lon_max=-80.0, lat_max=26.0),
}

S3_NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def bbox_intersects(item_bbox, region):
    """Check if item bbox overlaps with region bbox."""
    i_lon_min, i_lat_min, i_lon_max, i_lat_max = item_bbox
    return not (i_lon_max < region['lon_min'] or i_lon_min > region['lon_max'] or
                i_lat_max < region['lat_min'] or i_lat_min > region['lat_max'])


def find_gec_url(task_id):
    """
    Look up GEC.tif URL for a task via S3 listing.
    Data lives at: sar-data/task-data/{task_id}/{collect}/{collect}_GEC.tif
    """
    list_url = (f"{BUCKET_URL}/?list-type=2"
                f"&prefix=sar-data/task-data/{task_id}/"
                f"&max-keys=50")
    try:
        r = requests.get(list_url, timeout=30)
        r.raise_for_status()
        tree = ET.fromstring(r.content)
        keys = [c.text for c in tree.findall('.//s3:Contents/s3:Key', S3_NS)]
        gec_keys = [k for k in keys if k.endswith('_GEC.tif')]
        if gec_keys:
            return f"{BUCKET_URL}/{gec_keys[0]}"
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download GEC files for matched scenes')
    parser.add_argument('--region', choices=list(REGIONS.keys()), default=None,
                        help='Search one region only')
    args = parser.parse_args()

    active_regions = {args.region: REGIONS[args.region]} if args.region else REGIONS

    print("Loading Umbra Open Data static catalog (walking S3 tree — takes a few minutes)...")
    try:
        catalog = pystac.Catalog.from_file(CATALOG_URL)
    except Exception as e:
        print(f"Failed to open catalog: {e}")
        return

    matches = {key: [] for key in active_regions}
    total_scanned = 0

    print("Scanning items...")
    for item in catalog.get_all_items():
        total_scanned += 1
        if total_scanned % 100 == 0:
            print(f"  {total_scanned} items scanned...", flush=True)

        if not item.bbox:
            continue

        props = item.properties
        date    = props.get('datetime', '')[:10]
        gsd     = props.get('gsd') or props.get('resolution')
        task_id = props.get('umbra:task_id')

        for key, region in active_regions.items():
            if bbox_intersects(item.bbox, region):
                matches[key].append({
                    'id':      item.id,
                    'date':    date,
                    'gsd':     gsd,
                    'bbox':    item.bbox,
                    'task_id': task_id,
                    'assets':  list(item.assets.keys()),
                })

    print(f"Total items scanned: {total_scanned}\n")

    # Report
    total = 0
    for key, region in active_regions.items():
        found = matches[key]
        total += len(found)
        print(f"{region['name']}: {len(found)} scenes")
        for m in found:
            print(f"  {m['date']}  GSD={m['gsd']}m  {m['id']}  task={m['task_id']}")
            print(f"    Assets: {m['assets']}")

    print(f"\nTotal matched: {total} scenes across {len(active_regions)} regions")

    if not args.download or total == 0:
        return

    # Download GEC files
    print("\nDownloading GEC.tif files (resolving via sar-data/task-data/)...")
    for key, region in active_regions.items():
        for m in matches[key]:
            if not m['task_id']:
                print(f"  No task_id for {m['id']} — skipping")
                continue
            out_dir = OUT_BASE / key
            out_dir.mkdir(parents=True, exist_ok=True)

            gec_url = find_gec_url(m['task_id'])
            if not gec_url:
                print(f"  No GEC for {m['id']} (task {m['task_id']}) — skipping")
                continue

            fname = gec_url.split('/')[-1]
            out_path = out_dir / fname
            if out_path.exists():
                print(f"  Already exists: {fname}")
                continue
            print(f"  Downloading {fname}...", end=' ', flush=True)
            try:
                r = requests.get(gec_url, stream=True, timeout=300)
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                size = out_path.stat().st_size / 1e9
                print(f"done ({size:.2f} GB)")
            except Exception as e:
                print(f"ERROR — {e}")


if __name__ == '__main__':
    main()
