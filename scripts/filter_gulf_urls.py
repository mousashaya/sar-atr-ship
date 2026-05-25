"""
Filter the 250 Gulf ASF download URLs to a manageable sample.

Usage:
    python scripts/filter_gulf_urls.py gulf_urls.txt

Reads a file with one URL per line. Writes two output files:
  gulf_urls_may_aug.txt   — May-Aug 2024 only (AIS data available)
  gulf_urls_all.txt       — all 250, deduplicated

Paste your 250 URLs into a plain text file (one per line) and pass it as the argument.
"""

import sys
import re
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python filter_gulf_urls.py <url_file>")
    sys.exit(1)

urls = Path(sys.argv[1]).read_text().strip().splitlines()
urls = [u.strip() for u in urls if u.strip()]
print(f"Total URLs: {len(urls)}")

# Extract YYYYMMDD from the filename embedded in the URL
date_re = re.compile(r'_(\d{8})T\d{6}_')

may_aug = []
all_deduped = []
seen = set()

for url in urls:
    m = date_re.search(url)
    if not m:
        print(f"  WARNING: cannot parse date from {url[:80]}")
        continue
    date_str = m.group(1)  # e.g. 20240519
    if url not in seen:
        seen.add(url)
        all_deduped.append(url)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        if year == 2024 and 5 <= month <= 8:
            may_aug.append(url)

Path("gulf_urls_may_aug.txt").write_text("\n".join(may_aug) + "\n")
Path("gulf_urls_all.txt").write_text("\n".join(all_deduped) + "\n")

print(f"May-Aug 2024 subset: {len(may_aug)} scenes  → gulf_urls_may_aug.txt")
print(f"Full deduplicated:   {len(all_deduped)} scenes  → gulf_urls_all.txt")
print()
print("Next steps:")
print("  1. Download the sample:  aria2c -i gulf_urls_may_aug.txt -d /Volumes/Machine\\ Learning/sentinel1_gulf_zips -x 4 -j 4")
print("  2. Unzip:                for f in /Volumes/Machine\\ Learning/sentinel1_gulf_zips/*.zip; do unzip -q -d /Volumes/Machine\\ Learning/sentinel1_gulf \"$f\"; done")
print("  3. Run chipping:         python scripts/chip_ships_gulf.py")
