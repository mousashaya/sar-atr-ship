"""
Download AIS vessel tracks from Global Fishing Watch API
for the English Channel corridor, saved to CSV for AIS-SAR matching.

Usage:
    python scripts/download_ais_gfw.py --start 2024-01-01 --end 2024-02-01
    python scripts/download_ais_gfw.py --start 2024-01-01 --end 2024-02-01 --dry-run
"""

import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GFW_TOKEN = os.getenv("GFW_TOKEN")
GFW_BASE  = "https://gateway.api.globalfishingwatch.org/v3"

# English Channel bounding box: lon_min, lat_min, lon_max, lat_max
ENGLISH_CHANNEL = {
    "lon_min": -5.0,
    "lat_min":  49.0,
    "lon_max":   2.0,
    "lat_max":  52.0,
}

# AIS ship type codes → clean labels (IMO standard)
AIS_TYPE_MAP = {
    range(70, 80): "Cargo",
    range(80, 90): "Tanker",
    range(30, 40): "Fishing",
    range(50, 60): "Service",   # tugs, pilots, etc.
    range(60, 70): "Passenger",
    range(35, 36): "Fishing",   # fishing (alternate)
}

def map_ship_type(type_code):
    if type_code is None:
        return "Unknown"
    for r, label in AIS_TYPE_MAP.items():
        if type_code in r:
            return label
    return "Other"

def get_headers():
    return {
        "Authorization": f"Bearer {GFW_TOKEN}",
        "Content-Type": "application/json",
    }

def download_vessels(bbox, start_date, end_date, dry_run=False):
    """Download vessel positions from GFW for the given bbox and date range."""

    print(f"Querying GFW API: {start_date} → {end_date}")
    print(f"Bounding box: {bbox}")

    params = {
        "datasets[0]": "public-global-vessels-events:latest",
        "start-date": start_date,
        "end-date":   end_date,
        "bbox":       f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}",
        "limit":      1000,
        "offset":     0,
    }

    if dry_run:
        print("[DRY RUN] Would query:", f"{GFW_BASE}/events")
        print("[DRY RUN] Params:", params)
        return []

    url = f"{GFW_BASE}/events"
    all_records = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = requests.get(url, headers=get_headers(), params=params, timeout=30)

        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        entries = data.get("entries", [])
        total   = data.get("total", 0)

        print(f"  Fetched {offset + len(entries)} / {total} records")
        all_records.extend(entries)

        if offset + len(entries) >= total or len(entries) == 0:
            break
        offset += len(entries)

    return all_records

def parse_records(records):
    """Flatten GFW event records into a dataframe."""
    rows = []
    for r in records:
        vessel = r.get("vessel", {})
        pos    = r.get("position", {})
        rows.append({
            "mmsi":        vessel.get("mmsi"),
            "vessel_id":   vessel.get("id"),
            "ship_name":   vessel.get("name"),
            "ais_type":    vessel.get("type"),
            "ship_type":   map_ship_type(vessel.get("shiptype")),
            "lat":         pos.get("lat"),
            "lon":         pos.get("lon"),
            "timestamp":   r.get("start"),
            "speed_knots": r.get("speed"),
            "course_deg":  r.get("course"),
            "length_m":    vessel.get("length"),
            "width_m":     vessel.get("width"),
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Download AIS data from Global Fishing Watch")
    parser.add_argument("--start",   default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2024-02-01", help="End date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="data/ais",   help="Output directory for CSVs")
    parser.add_argument("--dry-run", action="store_true",  help="Test without API calls")
    args = parser.parse_args()

    if not GFW_TOKEN:
        print("ERROR: GFW_TOKEN not found in .env file")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = download_vessels(ENGLISH_CHANNEL, args.start, args.end, dry_run=args.dry_run)

    if args.dry_run:
        print("[DRY RUN] Done — no data downloaded")
        return

    if not records:
        print("No records returned")
        return

    df = parse_records(records)
    print(f"\nParsed {len(df)} vessel position records")
    print(f"Ship types: {df['ship_type'].value_counts().to_dict()}")

    out_file = out_dir / f"ais_english_channel_{args.start}_{args.end}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved → {out_file}")

if __name__ == "__main__":
    main()
