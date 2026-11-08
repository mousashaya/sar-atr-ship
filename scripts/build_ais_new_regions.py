"""
Build per-region daily AIS files for the 2026-11 expansion regions (NewEngland, GulfEast, Oregon, Hawaii), 2024.
Uses the national daily files already on disk in /Volumes/Machine Learning/ais_noaa where they are national
(Feb-Aug 2024, >200 MB); other days are streamed from NOAA's current host (2026-11: Azure blob, csv.zst,
https://noaaocm.blob.core.windows.net/ais/csv2/csv{YYYY}/ais-{YYYY-MM-DD}.csv.zst) and discarded after filtering.

Usage:
    venv/bin/python scripts/build_ais_new_regions.py [--start 2024-01-01 --end 2024-12-31]
"""
import argparse
import datetime
import io
import sys
from pathlib import Path

import pandas as pd
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
import requests
from download_ais_regions import REGIONS, ML_BASE, filter_region, save_zst

NEW_KEYS = ['NewEngland', 'GulfEast', 'Oregon', 'Hawaii']
BLOB = 'https://noaaocm.blob.core.windows.net/ais/csv2/csv{y}/ais-{d}.csv.zst'
LOCAL = ML_BASE / 'ais_noaa'
NATIONAL_MIN_BYTES = 200e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2024-01-01')
    ap.add_argument('--end', default='2024-12-31')
    ap.add_argument('--regions', nargs='+', default=NEW_KEYS, choices=list(REGIONS))
    a = ap.parse_args()
    NEW = {k: REGIONS[k] for k in a.regions}
    d, end = datetime.date.fromisoformat(a.start), datetime.date.fromisoformat(a.end)
    while d <= end:
        ds = d.isoformat()
        todo = {k: r for k, r in NEW.items() if not (ML_BASE / r['out_dir'] / f'ais-{ds}.csv.zst').exists()}
        if todo:
            loc = LOCAL / f'ais-{ds}.csv.zst'
            try:
                if loc.exists() and loc.stat().st_size > NATIONAL_MIN_BYTES:
                    with open(loc, 'rb') as f:
                        df = pd.read_csv(io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(f)), low_memory=False)
                    src = 'local'
                else:
                    tmp = ML_BASE / f'ais_tmp_{ds}.csv.zst'          # stream to disk, then stream-decompress
                    with requests.get(BLOB.format(y=d.year, d=ds), stream=True, timeout=600) as resp:
                        if resp.status_code == 404:
                            print(f'{ds} not available', flush=True); d += datetime.timedelta(days=1); continue
                        resp.raise_for_status()
                        with open(tmp, 'wb') as f:
                            for chunk in resp.iter_content(8 << 20): f.write(chunk)
                    with open(tmp, 'rb') as f:
                        df = pd.read_csv(io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(f)), low_memory=False)
                    tmp.unlink()
                    src = 'NOAA'
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                for k, r in todo.items():
                    save_zst(filter_region(df, r), ML_BASE / r['out_dir'] / f'ais-{ds}.csv.zst')
                print(f'{ds} {src:5s}  {list(todo)}', flush=True)
            except Exception as e:
                print(f'{ds} ERROR {e}', flush=True)
        d += datetime.timedelta(days=1)


if __name__ == '__main__':
    main()
