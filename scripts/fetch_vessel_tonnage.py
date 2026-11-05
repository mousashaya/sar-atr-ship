"""
Fetch DWT and gross tonnage for all vessels in GFW data via VesselAPI.
Saves results to data/vessel_tonnage.json.

Usage:
    export VESSEL_API_KEY=your_key_here
    venv/bin/python scripts/fetch_vessel_tonnage.py
    venv/bin/python scripts/fetch_vessel_tonnage.py --dry-run   # show counts only

VesselAPI free tier: https://vesselapi.com
Endpoint: GET https://api.vesselapi.com/v1/vessel/{mmsi}?filter.idType=mmsi
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

GFW_PATH    = 'data/gfw_vessel_types.json'
OUT_PATH    = 'data/vessel_tonnage.json'
BASE_URL    = 'https://api.vesselapi.com/v1'
DELAY_S     = 1.5   # seconds between requests — stay within free tier limits


def get_api_key():
    key = os.environ.get('VESSEL_API_KEY', '')
    if not key:
        raise RuntimeError("Set VESSEL_API_KEY environment variable before running")
    return key


def fetch_tonnage(mmsi: str, session: requests.Session) -> dict:
    url = f"{BASE_URL}/vessel/{mmsi}"
    resp = session.get(url, params={'filter.idType': 'mmsi'}, timeout=30)
    if resp.status_code == 404:
        return {'mmsi': mmsi, 'found': False}
    resp.raise_for_status()
    data = resp.json()
    return {
        'mmsi':               mmsi,
        'found':              True,
        'deadweight_tonnage': data.get('deadweight_tonnage'),
        'gross_tonnage':      data.get('gross_tonnage'),
        'vessel_type':        data.get('vessel_type'),
        'name':               data.get('name'),
        'imo':                data.get('imo'),
        'length':             data.get('length'),
        'year_built':         data.get('year_built'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(GFW_PATH) as f:
        gfw = json.load(f)

    mmsis = [k for k, v in gfw.items() if v is not None]
    print(f"Vessels to query: {len(mmsis)}")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    # Load existing results for resume support
    out = {}
    if Path(OUT_PATH).exists():
        with open(OUT_PATH) as f:
            out = json.load(f)
        print(f"Resuming — {len(out)} already fetched, {len(mmsis) - len(out)} remaining")

    api_key = get_api_key()
    session = requests.Session()
    session.headers['Authorization'] = f'Bearer {api_key}'

    fetched = 0
    errors  = 0

    for i, mmsi in enumerate(mmsis):
        if mmsi in out:
            continue

        try:
            result = fetch_tonnage(mmsi, session)
            out[mmsi] = result
            fetched += 1

            if fetched % 50 == 0:
                with open(OUT_PATH, 'w') as f:
                    json.dump(out, f, indent=2)
                found = sum(1 for v in out.values() if v.get('found'))
                print(f"  [{fetched} fetched]  found: {found}  errors: {errors}")

            time.sleep(DELAY_S)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 30
                print(f"  Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                try:
                    result = fetch_tonnage(mmsi, session)
                    out[mmsi] = result
                    fetched += 1
                    time.sleep(DELAY_S)
                    continue
                except Exception as e2:
                    errors += 1
                    print(f"  WARNING: MMSI {mmsi} failed after retry — {e2}")
                    out[mmsi] = {'mmsi': mmsi, 'found': False, 'error': str(e2)}
            else:
                errors += 1
                print(f"  WARNING: MMSI {mmsi} failed — {e}")
                out[mmsi] = {'mmsi': mmsi, 'found': False, 'error': str(e)}
                time.sleep(DELAY_S * 2)
        except Exception as e:
            errors += 1
            print(f"  WARNING: MMSI {mmsi} failed — {e}")
            out[mmsi] = {'mmsi': mmsi, 'found': False, 'error': str(e)}
            time.sleep(DELAY_S * 2)

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    found     = sum(1 for v in out.values() if v.get('found'))
    has_dwt   = sum(1 for v in out.values() if v.get('deadweight_tonnage'))
    has_gt    = sum(1 for v in out.values() if v.get('gross_tonnage'))

    print(f"\n{'='*50}")
    print(f"  Total queried:  {len(out)}")
    print(f"  Found:          {found} ({found/len(out)*100:.1f}%)")
    print(f"  Has DWT:        {has_dwt} ({has_dwt/len(out)*100:.1f}%)")
    print(f"  Has gross ton:  {has_gt} ({has_gt/len(out)*100:.1f}%)")
    print(f"  Errors:         {errors}")
    print(f"  Saved to:       {OUT_PATH}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
