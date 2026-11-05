"""
AIS-only vessel label resolver — ablation baseline.
Uses only the raw AIS type code, no secondary signals.
This is the naive baseline that most prior work implicitly uses.

Output: data/resolved_labels_ais_only.json
"""

import glob
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

CHIPS_DIR = '/Volumes/Machine Learning/sar_ship_chips_gulf'
OUT_PATH  = 'data/resolved_labels_ais_only.json'

TANKER_AIS  = set(range(80, 90))
CARGO_AIS   = set(range(70, 80))
FISHING_AIS = {30}
TUG_AIS     = {31, 32} | set(range(50, 60))

def ais_vote(code):
    if code in TANKER_AIS:  return 'Tanker'
    if code in CARGO_AIS:   return 'Cargo'
    if code in FISHING_AIS: return 'Fishing'
    if code in TUG_AIS:     return 'Tug/Special'
    return None

print("Loading chip metadata...")
by_mmsi = defaultdict(list)
for npz in sorted(glob.glob(f'{CHIPS_DIR}/**/*.npz', recursive=True)):
    try:
        f = np.load(npz, allow_pickle=True)
        mmsi = str(int(f['mmsi']))
        code = int(f['vessel_type'])
        by_mmsi[mmsi].append(code)
    except Exception:
        continue

print(f"Loaded {len(by_mmsi)} unique vessels")

results = {}
class_counts = Counter()

for mmsi, codes in by_mmsi.items():
    ais_code = Counter(codes).most_common(1)[0][0]
    label = ais_vote(ais_code)
    if label is None:
        label = 'Unresolved'

    results[mmsi] = {
        'mmsi':       mmsi,
        'label':      label,
        'confidence': 'AIS_ONLY',
        'sources':    ['ais'],
        'ais_code':   ais_code,
    }
    class_counts[label] += 1

total    = len(results)
resolved = sum(1 for v in results.values() if v['label'] != 'Unresolved')

print(f"\n{'='*55}")
print(f"  AIS-Only Label Resolution")
print(f"  Total vessels:  {total}")
print(f"  Resolved:       {resolved} ({resolved/total*100:.1f}%)")
print(f"  Unresolved:     {total-resolved} ({(total-resolved)/total*100:.1f}%)")
print(f"\n  By class:")
for cls, n in class_counts.most_common():
    print(f"    {cls:15s}: {n:4d} ({n/total*100:.1f}%)")
print(f"{'='*55}")

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
