"""
Multi-signal vessel type resolver for SLC chips.
Combines AIS type code, ship length, GFW classification, and vessel name
into a confident label or marks the vessel as unresolvable.

Confidence voting: each signal casts a vote for a class.
  - 3+ agreeing votes → HIGH confidence
  - 2 agreeing votes  → MEDIUM confidence
  - 1 vote or conflict → UNRESOLVED

Output: data/resolved_labels.json  keyed by MMSI string
        data/resolved_labels_summary.txt

Classes: Tanker, Cargo, Fishing, Tug/Special, Unresolved
"""

import glob
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

CHIPS_DIR = '/Volumes/Machine Learning/sar_ship_chips_slc_gulf'
GFW_PATH  = 'data/gfw_vessel_types.json'
OUT_PATH  = 'data/resolved_labels.json'

with open(GFW_PATH) as f:
    gfw = json.load(f)

# ── Signal definitions ─────────────────────────────────────────────────────────

TANKER_AIS   = set(range(80, 90))
CARGO_AIS    = set(range(70, 80))
FISHING_AIS  = {30}
TUG_AIS      = {31, 32} | set(range(50, 60))

TANKER_KW  = ['TANKER','CRUDE','LNG','LPG','VLCC','ULCC','FPSO',
               'CHEMICAL','BITUMEN','ASPHALT','GAS CARRIER']
CARGO_KW   = ['BULK CARRIER','CONTAINER','CARGO','BULKER','CEMENT',
               'GRAIN','COAL CARRIER','GENERAL CARGO']
TUG_KW     = ['TUG','AHTS','PSV','OSV','OFFSHORE SUPPLY','ANCHOR HANDLING']
FISHING_KW = ['FISHING','TRAWL','SEINER']

def ais_vote(code):
    if code in TANKER_AIS:  return 'Tanker'
    if code in CARGO_AIS:   return 'Cargo'
    if code in FISHING_AIS: return 'Fishing'
    if code in TUG_AIS:     return 'Tug/Special'
    return None

def length_vote(length):
    """Length alone is only a strong signal at the extremes."""
    if length >= 200:  return 'Tanker'   # almost nothing else this big in Gulf
    if length < 50:    return 'Tug/Special'
    if length < 30:    return 'Fishing'
    return None  # 50-200m is ambiguous — no vote

def gfw_vote(ship_type):
    if not ship_type: return None
    t = ship_type.upper()
    if t in ('TANKER', 'BUNKER'):  return 'Tanker'
    if t == 'CARGO':               return 'Cargo'
    if t == 'FISHING':             return 'Fishing'
    if t in ('CARRIER', 'SEISMIC_VESSEL'): return 'Cargo'
    return None  # OTHER, NA → no vote

def name_vote(name):
    if not name: return None
    n = name.upper()
    if any(k in n for k in TANKER_KW):  return 'Tanker'
    if any(k in n for k in CARGO_KW):   return 'Cargo'
    if any(k in n for k in TUG_KW):     return 'Tug/Special'
    if any(k in n for k in FISHING_KW): return 'Fishing'
    return None

def resolve(ais_code, length, gfw_type, name):
    votes = []
    v = ais_vote(ais_code);  votes.append(('ais', v))
    v = length_vote(length); votes.append(('length', v))
    v = gfw_vote(gfw_type);  votes.append(('gfw', v))
    v = name_vote(name);     votes.append(('name', v))

    # Count non-None votes
    cast = [(src, cls) for src, cls in votes if cls is not None]
    if not cast:
        return 'Unresolved', 0, []

    class_counts = Counter(cls for _, cls in cast)
    top_class, top_count = class_counts.most_common(1)[0]
    sources = [src for src, cls in cast if cls == top_class]

    # Check for conflict: another class got votes too
    conflict = len(class_counts) > 1

    if top_count >= 3:
        confidence = 'HIGH'
    elif top_count == 2 and not conflict:
        confidence = 'HIGH'
    elif top_count == 2 and conflict:
        confidence = 'MEDIUM'
    elif top_count == 1 and not conflict:
        confidence = 'MEDIUM'
    else:
        return 'Unresolved', 0, sources

    return top_class, confidence, sources


# ── Load chip metadata per MMSI ───────────────────────────────────────────────

print("Loading chip metadata...")
by_mmsi = defaultdict(lambda: {'ais_codes': [], 'lengths': [], 'names': []})

for npz in sorted(glob.glob(f'{CHIPS_DIR}/**/*.npz', recursive=True)):
    try:
        f = np.load(npz, allow_pickle=True)
        mmsi   = str(int(f['mmsi']))
        code   = int(f['vessel_type'])
        length = float(f['length_m'])
        name   = str(f.get('vessel_name', '')) if 'vessel_name' in f else ''
        by_mmsi[mmsi]['ais_codes'].append(code)
        by_mmsi[mmsi]['lengths'].append(length)
        by_mmsi[mmsi]['names'].append(name)
    except Exception:
        continue

print(f"Loaded {len(by_mmsi)} unique vessels\n")

# ── Resolve per vessel ─────────────────────────────────────────────────────────

results = {}
conf_counts = Counter()
class_counts = Counter()

for mmsi, data in by_mmsi.items():
    # Use modal AIS code, median length, any non-empty name
    ais_code = Counter(data['ais_codes']).most_common(1)[0][0]
    length   = float(np.median(data['lengths']))
    name     = next((n for n in data['names'] if n and n not in ('', 'nan', 'None')), '')
    gfw_info = gfw.get(mmsi)
    gfw_type = gfw_info.get('ship_type', '') if gfw_info else ''
    gfw_name = gfw_info.get('name', '') if gfw_info else ''
    name     = name or gfw_name   # prefer AIS name, fallback to GFW name

    label, confidence, sources = resolve(ais_code, length, gfw_type, name)

    results[mmsi] = {
        'mmsi':       mmsi,
        'label':      label,
        'confidence': confidence,
        'sources':    sources,
        'ais_code':   ais_code,
        'length_m':   length,
        'gfw_type':   gfw_type,
        'name':       name,
        'n_chips':    len(data['ais_codes']),
    }
    conf_counts[confidence if label != 'Unresolved' else 'Unresolved'] += 1
    class_counts[label] += 1

# ── Summary ────────────────────────────────────────────────────────────────────

total = len(results)
resolved = sum(1 for v in results.values() if v['label'] != 'Unresolved')

print(f"{'='*55}")
print(f"  Vessel Label Resolution Summary")
print(f"  Total vessels:    {total}")
print(f"  Resolved:         {resolved} ({resolved/total*100:.1f}%)")
print(f"  Unresolved:       {total-resolved} ({(total-resolved)/total*100:.1f}%)")
print(f"\n  By class:")
for cls, n in class_counts.most_common():
    print(f"    {cls:15s}: {n:4d} ({n/total*100:.1f}%)")
print(f"\n  By confidence:")
for conf, n in conf_counts.most_common():
    print(f"    {conf:12s}: {n:4d}")
print(f"{'='*55}")

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_PATH}")

# ── Show conflict examples ────────────────────────────────────────────────────
print("\nSample UNRESOLVED vessels (first 10):")
for mmsi, v in list(results.items())[:100]:
    if v['label'] == 'Unresolved':
        print(f"  MMSI {mmsi}: AIS={v['ais_code']} len={v['length_m']:.0f}m "
              f"GFW={v['gfw_type']} name={v['name'][:30]}")
