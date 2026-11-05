"""
Download ALL Umbra open data GEC.tif scenes — no region filter.
Output: /Volumes/Machine Learning/umbra_open_data/global/

Skips files already on disk. Safe to interrupt and resume.
Expected: ~1,558 scenes, ~400 GB, ~12-20 hours depending on bandwidth.

Usage:
    venv/bin/python scripts/download_umbra_global.py
    venv/bin/python scripts/download_umbra_global.py --dry-run   # count only
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pystac
import requests

CATALOG_URL = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/stac/catalog.json"
BUCKET_URL  = "https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog"
OUT_DIR     = Path("/Volumes/Machine Learning/umbra_open_data/global")

S3_NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def find_gec_url(task_id):
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
    parser.add_argument('--dry-run', action='store_true', help='Count scenes without downloading')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Umbra catalog...", flush=True)
    catalog = pystac.Catalog.from_file(CATALOG_URL)

    items = []
    for i, item in enumerate(catalog.get_all_items(), 1):
        if i % 200 == 0:
            print(f"  {i} items scanned...", flush=True)
        task_id = item.properties.get('umbra:task_id')
        date    = item.properties.get('datetime', '')[:10]
        if task_id:
            items.append((task_id, date, item.id))

    print(f"\nTotal scenes in catalog: {len(items)}")

    already = sum(1 for f in OUT_DIR.glob('*_GEC.tif'))
    print(f"Already downloaded:      {already}")
    todo = len(items) - already
    print(f"To download:             {todo}")
    print(f"Estimated storage:       {todo * 0.256:.0f} GB\n")

    if args.dry_run:
        return

    ok = skipped = errors = 0
    for idx, (task_id, date, item_id) in enumerate(items, 1):
        gec_url = find_gec_url(task_id)
        if not gec_url:
            print(f"[{idx}/{len(items)}] No GEC for {item_id} — skip")
            errors += 1
            continue

        fname = gec_url.split('/')[-1]
        out_path = OUT_DIR / fname

        if out_path.exists():
            skipped += 1
            continue

        print(f"[{idx}/{len(items)}] {fname} ({date})...", end=' ', flush=True)
        try:
            r = requests.get(gec_url, stream=True, timeout=600)
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            size_mb = out_path.stat().st_size / 1e6
            print(f"done ({size_mb:.0f} MB)", flush=True)
            ok += 1
        except Exception as e:
            print(f"ERROR — {e}", flush=True)
            if out_path.exists():
                out_path.unlink()
            errors += 1

    print(f"\nDone. Downloaded: {ok}  Skipped: {skipped}  Errors: {errors}")


if __name__ == '__main__':
    main()
