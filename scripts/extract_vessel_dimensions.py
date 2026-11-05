"""
Extract vessel dimensions (length, width) from NOAA AIS data for all MMSIs in GFW data.
Takes the max observed length per MMSI (most reliable — AIS occasionally reports 0).

Output: data/vessel_dimensions.json

Usage:
    venv/bin/python scripts/extract_vessel_dimensions.py
    venv/bin/python scripts/extract_vessel_dimensions.py --dry-run
"""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import zstandard as zstd

AIS_DIR  = Path('/Volumes/Machine Learning/ais_noaa')
GFW_PATH = Path('data/gfw_vessel_types.json')
OUT_PATH = Path('data/vessel_dimensions.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    gfw = json.loads(GFW_PATH.read_text())
    target_mmsis = {str(k) for k, v in gfw.items() if v is not None}
    print(f"Target MMSIs: {len(target_mmsis)}")

    files = sorted(AIS_DIR.glob('ais-*.csv.zst'))
    print(f"AIS files:    {len(files)}")

    if args.dry_run:
        return

    # mmsi -> {'length': max, 'width': max, 'name': str, 'imo': str, 'vessel_type': mode}
    best = defaultdict(lambda: {'length': 0.0, 'width': 0.0, 'name': '', 'imo': '', 'vessel_type': None})

    dctx = zstd.ZstdDecompressor()
    for i, path in enumerate(files, 1):
        print(f"  [{i:3d}/{len(files)}] {path.name}", flush=True)
        try:
            with open(path, 'rb') as f:
                raw = dctx.decompress(f.read())
            df = pd.read_csv(io.BytesIO(raw), dtype={'mmsi': str, 'imo': str},
                             usecols=['mmsi', 'vessel_name', 'imo', 'vessel_type', 'length', 'width'])
        except Exception as e:
            print(f"    WARNING: skipping {path.name} — {e}")
            continue

        df = df[df['mmsi'].isin(target_mmsis)]
        if df.empty:
            continue

        for mmsi, grp in df.groupby('mmsi'):
            rec = best[mmsi]

            max_len = grp['length'].max()
            max_wid = grp['width'].max()
            if max_len > rec['length']:
                rec['length'] = float(max_len)
            if max_wid > rec['width']:
                rec['width'] = float(max_wid)

            if not rec['name']:
                names = grp['vessel_name'].dropna()
                if not names.empty:
                    rec['name'] = str(names.iloc[0])

            if not rec['imo']:
                imos = grp['imo'].dropna()
                if not imos.empty:
                    rec['imo'] = str(imos.iloc[0])

            if rec['vessel_type'] is None:
                types = grp['vessel_type'].dropna()
                if not types.empty:
                    rec['vessel_type'] = int(types.mode().iloc[0])

    # Finalise
    out = {}
    for mmsi in target_mmsis:
        rec = best.get(mmsi)
        if rec and rec['length'] > 0:
            out[mmsi] = {
                'mmsi':        mmsi,
                'found':       True,
                'length':      rec['length'],
                'width':       rec['width'],
                'name':        rec['name'],
                'imo':         rec['imo'],
                'vessel_type': rec['vessel_type'],
            }
        else:
            out[mmsi] = {'mmsi': mmsi, 'found': False}

    OUT_PATH.write_text(json.dumps(out, indent=2))

    found    = sum(1 for v in out.values() if v['found'])
    has_len  = sum(1 for v in out.values() if v.get('length', 0) > 0)
    print(f"\n{'='*50}")
    print(f"  Total queried: {len(out)}")
    print(f"  Found:         {found} ({found/len(out)*100:.1f}%)")
    print(f"  Has length:    {has_len} ({has_len/len(out)*100:.1f}%)")
    print(f"  Saved to:      {OUT_PATH}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
