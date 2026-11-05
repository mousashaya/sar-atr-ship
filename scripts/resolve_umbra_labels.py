#!/usr/bin/env python3
"""
Apply the multi-signal AIS resolver to Umbra ship chips.

Uses the same resolver logic as the Sentinel-1 paper (ais_code, length_m,
vessel_name keywords). GFW signal is recycled from existing resolved_labels.json
for vessels that appear in both datasets; new vessels get 3-signal resolution.

Output: data/resolved_umbra_labels.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Signal maps (identical to paper resolver) ─────────────────────────────────
TANKER_AIS  = set(range(80, 90))
CARGO_AIS   = set(range(70, 80))
FISHING_AIS = {30}
TUG_AIS     = {31, 32, 33, 34} | set(range(50, 60))  # 33=dredging, 34=diving ops

TANKER_KW  = {'TANKER','CRUDE','LNG','VLCC','FPSO','CHEMICAL','BITUMEN','PRODUCT'}
CARGO_KW   = {'BULK','CONTAINER','CARGO','CEMENT','GRAIN','REEFER','LIVESTOCK'}
TUG_KW     = {'TUG','AHTS','PSV','SUPPLY','OFFSHORE','ANCHOR'}
FISHING_KW = {'FISHING','TRAWL','SEINER','DREDGE'}

GFW_MAP = {'TANKER':'Tanker','OIL_TANKER':'Tanker','CHEMICAL_TANKER':'Tanker',
           'LNG_TANKER':'Tanker','LPG_TANKER':'Tanker',
           'CARGO':'Cargo','CONTAINER_SHIP':'Cargo','BULK_CARRIER':'Cargo',
           'REEFER':'Cargo','GENERAL_CARGO':'Cargo',
           'FISHING':'Fishing','TRAWLERS':'Fishing','DRIFTING_LONGLINES':'Fishing',
           'TRAWLER':'Fishing',
           'TUG':'Tug/Special','SUPPLY':'Tug/Special','SPECIALIZED':'Tug/Special',
           'PASSENGER':'Tug/Special'}


def ais_vote(code):
    if code is None: return None
    c = int(code)
    if c in TANKER_AIS:  return 'Tanker'
    if c in CARGO_AIS:   return 'Cargo'
    if c in FISHING_AIS: return 'Fishing'
    if c in TUG_AIS:     return 'Tug/Special'
    return None

def length_vote(length_m):
    if not length_m: return None
    l = float(length_m)
    if l >= 300: return 'Tanker'   # raised from 200m: 300m+ is almost certainly tanker/bulk
    if l < 50:   return 'Tug/Special'
    return None

def gfw_vote(gfw_type):
    if not gfw_type: return None
    return GFW_MAP.get(str(gfw_type).upper())

def name_vote(name):
    if not name: return None
    n = str(name).upper()
    for kw in TANKER_KW:
        if kw in n: return 'Tanker'
    for kw in CARGO_KW:
        if kw in n: return 'Cargo'
    for kw in TUG_KW:
        if kw in n: return 'Tug/Special'
    for kw in FISHING_KW:
        if kw in n: return 'Fishing'
    return None

def resolve_votes(votes):
    if not votes:
        return 'Unresolved', None
    tally = defaultdict(int)
    for cls in votes:
        tally[cls] += 1
    ranked = sorted(tally.items(), key=lambda x: -x[1])
    top_cls, top_n = ranked[0]
    other_n = sum(n for _, n in ranked[1:])
    if top_n >= 3:                  return 'HIGH',   top_cls
    if top_n == 2 and other_n == 0: return 'HIGH',   top_cls
    if top_n >= 2 and other_n >= 1: return 'MEDIUM', top_cls
    if top_n == 1 and other_n == 0: return 'MEDIUM', top_cls
    # Conflicting single votes: trust the plurality, mark MEDIUM
    return 'MEDIUM', top_cls


# ── Load existing resolved labels for GFW recycling ──────────────────────────
existing_path = Path('data/resolved_labels.json')
existing = {}
if existing_path.exists():
    with open(existing_path) as f:
        existing = json.load(f)
print(f"Existing resolved labels (Sentinel-1): {len(existing)}")


# ── Collect unique vessels from Umbra chips ───────────────────────────────────
CHIP_DIRS = {
    'la':         Path('/Volumes/Machine Learning/umbra_ship_chips_la'),
    'ny':         Path('/Volumes/Machine Learning/umbra_ship_chips_ny'),
    'savannah':   Path('/Volumes/Machine Learning/umbra_ship_chips_savannah'),
    'chesapeake': Path('/Volumes/Machine Learning/umbra_ship_chips_chesapeake'),
    'puget':      Path('/Volumes/Machine Learning/umbra_ship_chips_puget'),
}

vessels = {}
for region, chip_dir in CHIP_DIRS.items():
    if not chip_dir.exists():
        continue
    for npz in sorted(chip_dir.glob('**/*.npz')):
        try:
            d = np.load(npz, allow_pickle=True)
            mmsi = str(int(d['mmsi']))
            if mmsi not in vessels:
                vessels[mmsi] = {
                    'mmsi':        mmsi,
                    'vessel_name': str(d['vessel_name']),
                    'ais_code':    int(d['vessel_type']),
                    'length_m':    float(d['length_m']),
                    'regions':     [],
                    'n_chips':     0,
                    # GFW recycled from existing resolved labels if available
                    'gfw_type':    existing.get(mmsi, {}).get('gfw_type', ''),
                }
            vessels[mmsi]['n_chips'] += 1
            if region not in vessels[mmsi]['regions']:
                vessels[mmsi]['regions'].append(region)
        except Exception:
            pass

print(f"Unique Umbra vessels: {len(vessels)}")
gfw_recycled = sum(1 for v in vessels.values() if v['gfw_type'])
print(f"GFW recycled from Sentinel-1 overlap: {gfw_recycled}")


# ── Resolve ───────────────────────────────────────────────────────────────────
resolved = {}
for mmsi, v in vessels.items():
    votes = []
    av = ais_vote(v['ais_code'])
    lv = length_vote(v['length_m'])
    gv = gfw_vote(v['gfw_type'])
    nv = name_vote(v['vessel_name'])
    if av: votes.append(av)
    if lv: votes.append(lv)
    if gv: votes.append(gv)
    if nv: votes.append(nv)

    confidence, label = resolve_votes(votes)

    resolved[mmsi] = {
        'label':       label if label else 'Unresolved',
        'confidence':  confidence,
        'ais_code':    v['ais_code'],
        'length_m':    v['length_m'],
        'gfw_type':    v['gfw_type'],
        'name':        v['vessel_name'],
        'regions':     v['regions'],
        'n_chips':     v['n_chips'],
        'votes':       {'ais': av, 'length': lv, 'gfw': gv, 'name': nv},
    }


# ── Report ────────────────────────────────────────────────────────────────────
high   = {m: v for m, v in resolved.items() if v['confidence'] == 'HIGH'}
medium = {m: v for m, v in resolved.items() if v['confidence'] == 'MEDIUM'}
unresl = {m: v for m, v in resolved.items() if v['confidence'] == 'Unresolved'}

print(f"\nResolution results:")
print(f"  HIGH:       {len(high):4d}  ({len(high)/len(resolved)*100:.1f}%)")
print(f"  MEDIUM:     {len(medium):4d}  ({len(medium)/len(resolved)*100:.1f}%)")
print(f"  Unresolved: {len(unresl):4d}  ({len(unresl)/len(resolved)*100:.1f}%)")
print(f"  Total resolved: {(len(high)+len(medium))/len(resolved)*100:.1f}%")

CLASSES = ['Tanker', 'Cargo', 'Fishing', 'Tug/Special', 'Unresolved']
print(f"\nClass distribution (HIGH confidence):")
for cls in CLASSES:
    n = sum(1 for v in high.values() if v['label'] == cls)
    chips = sum(v['n_chips'] for v in high.values() if v['label'] == cls)
    if n > 0:
        print(f"  {cls:<12}: {n:3d} vessels, {chips:4d} chips")

print(f"\nClass distribution (MEDIUM confidence):")
for cls in CLASSES:
    n = sum(1 for v in medium.values() if v['label'] == cls)
    chips = sum(v['n_chips'] for v in medium.values() if v['label'] == cls)
    if n > 0:
        print(f"  {cls:<12}: {n:3d} vessels, {chips:4d} chips")

print(f"\nUnresolved vessels (AIS code not in known classes):")
for mmsi, v in sorted(unresl.items(), key=lambda x: -x[1]['n_chips'])[:10]:
    print(f"  MMSI {mmsi}  code={v['ais_code']}  len={v['length_m']:.0f}m  "
          f"name={v['name']!r}  chips={v['n_chips']}  votes={v['votes']}")

out_path = Path('data/resolved_umbra_labels.json')
with open(out_path, 'w') as f:
    json.dump(resolved, f, indent=2)
print(f"\nSaved → {out_path}")
