"""
Look up vessel types for all unique MMSIs in SLC chips via Global Fishing Watch API.
Saves results to data/gfw_vessel_types.json — keyed by MMSI string.

GFW vessel type categories:
  fishing, carrier, support, passenger, cargo, tanker, other

Usage:
    venv/bin/python scripts/lookup_gfw_vessel_types.py
"""

import glob
import json
import os
import time
from collections import Counter

import numpy as np
import requests

import sys
# extra chip dirs can be passed as arguments (e.g. other regions); default = SLC Gulf
CHIP_DIRS = sys.argv[1:] or ['/Volumes/Machine Learning/sar_ship_chips_slc_gulf']
OUT_PATH  = 'data/gfw_vessel_types.json'
API_KEY   = os.environ.get('GFW_API_KEY', '')
BASE_URL  = 'https://gateway.api.globalfishingwatch.org/v3/vessels/search'
RATE_LIMIT = 10   # requests per second (free tier)

if not API_KEY:
    raise RuntimeError("GFW_API_KEY not set. Run: source ~/.zshrc")

os.makedirs('data', exist_ok=True)

# Load existing results to allow resume
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} existing results — will skip those MMSIs")
else:
    results = {}

# Collect unique MMSIs
print("Scanning chips for unique MMSIs...")
mmsis = set()
for npz in (f for d in CHIP_DIRS for f in glob.glob(f'{d}/**/*.npz', recursive=True)):
    try:
        mmsis.add(int(np.load(npz, allow_pickle=True)['mmsi']))
    except Exception:
        continue

mmsis_to_query = sorted(mmsis - {int(k) for k in results.keys()})
print(f"Total unique MMSIs: {len(mmsis)}")
print(f"Already cached:     {len(results)}")
print(f"To query:           {len(mmsis_to_query)}\n")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}

def query_mmsi(mmsi):
    """Query GFW vessel search API for a single MMSI. Returns vessel info dict or None."""
    params = {
        'query': str(mmsi),
        'datasets[0]': 'public-global-vessel-identity:latest',
        'limit': 1,
    }
    try:
        r = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            entries = data.get('entries', [])
            if entries:
                v = entries[0]
                sr   = (v.get('selfReportedInfo') or [{}])[0]
                comb = (v.get('combinedSourcesInfo') or [{}])[0]
                reg  = (v.get('registryInfo') or [{}])[0]
                ship_type = ((comb.get('shiptypes') or [{}])[0]).get('name', '')
                gear_type = ((comb.get('geartypes') or [{}])[0]).get('name', '')
                return {
                    'mmsi':      mmsi,
                    'ship_type': ship_type,   # GFW inferred: TANKER, CARGO, FISHING, etc.
                    'gear_type': gear_type,
                    'name':      sr.get('shipname') or reg.get('shipname', ''),
                    'flag':      sr.get('flag')     or reg.get('flag', ''),
                    'imo':       sr.get('imo')      or reg.get('imo', ''),
                    'length':    reg.get('length', None),
                    'gfw_id':    sr.get('id', ''),
                }
        elif r.status_code == 429:
            time.sleep(2)
            return 'rate_limited'
        return None
    except Exception:
        return None


found = not_found = errors = 0
delay = 1.0 / RATE_LIMIT

for i, mmsi in enumerate(mmsis_to_query):
    result = query_mmsi(mmsi)

    if result == 'rate_limited':
        print(f"  Rate limited at {i} — waiting 5s...")
        time.sleep(5)
        result = query_mmsi(mmsi)

    if result and result != 'rate_limited':
        results[str(mmsi)] = result
        found += 1
    else:
        results[str(mmsi)] = None
        not_found += 1

    time.sleep(delay)

    if (i + 1) % 100 == 0 or (i + 1) == len(mmsis_to_query):
        with open(OUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  {i+1}/{len(mmsis_to_query)} — found: {found}  not found: {not_found}", flush=True)

# Summary
print(f"\nDone. Results saved to {OUT_PATH}")
print(f"Found: {found}  Not found: {not_found}")

types = Counter()
for v in results.values():
    if v:
        types[v.get('vessel_type', 'unknown')] += 1
    else:
        types['not_found'] += 1

print("\nGFW vessel type distribution:")
for t, n in types.most_common():
    print(f"  {t:20s}: {n}")
