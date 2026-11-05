#!/usr/bin/env python3
"""
Resolver signal ablation: remove each signal one at a time and report
change in resolution rate. Uses raw signal values stored in resolved_labels.json.
"""

import json
from collections import defaultdict
from pathlib import Path

LABELS_PATH = Path('data/resolved_labels.json')
with open(LABELS_PATH) as f:
    resolved = json.load(f)

# ── Signal mapping functions (same as main resolver) ──────────────────────
TANKER_AIS  = set(range(80, 90))
CARGO_AIS   = set(range(70, 80))
FISHING_AIS = {30}
TUG_AIS     = {31, 32} | set(range(50, 60))

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
    if length_m is None: return None
    l = float(length_m)
    if l >= 200: return 'Tanker'
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
    if top_n >= 3:                        return 'HIGH',       top_cls
    if top_n == 2 and other_n == 0:       return 'HIGH',       top_cls
    if top_n >= 2 and other_n >= 1:       return 'MEDIUM',     top_cls
    if top_n == 1 and other_n == 0:       return 'MEDIUM',     top_cls
    return 'Unresolved', None

# ── Ablation ──────────────────────────────────────────────────────────────
ABLATIONS = [
    (None,        'Full resolver'),
    ('ais_type',  'Remove AIS type'),
    ('length',    'Remove length'),
    ('gfw',       'Remove GFW'),
    ('name',      'Remove name kw.'),
]

full_resolved = sum(1 for v in resolved.values() if v['label'] != 'Unresolved')
print(f"Vessels: {len(resolved)}\n")
print(f"{'Condition':<22}  {'HIGH':>5}  {'MEDIUM':>6}  {'Unres.':>6}  {'Resolved%':>9}  {'Drop':>5}")
print("-" * 60)

ablation_rows = []
for ablate_sig, label in ABLATIONS:
    high = medium = unresolved = 0
    for entry in resolved.values():
        votes = []
        if ablate_sig != 'ais_type':
            v = ais_vote(entry.get('ais_code'))
            if v: votes.append(v)
        if ablate_sig != 'length':
            v = length_vote(entry.get('length_m'))
            if v: votes.append(v)
        if ablate_sig != 'gfw':
            v = gfw_vote(entry.get('gfw_type'))
            if v: votes.append(v)
        if ablate_sig != 'name':
            v = name_vote(entry.get('name'))
            if v: votes.append(v)
        conf, _ = resolve_votes(votes)
        if conf == 'HIGH':     high += 1
        elif conf == 'MEDIUM': medium += 1
        else:                  unresolved += 1

    total = len(resolved)
    pct   = (high + medium) / total * 100
    drop  = full_resolved - (high + medium)
    drop_str = f'-{drop}' if ablate_sig else '—'
    print(f"{label:<22}  {high:>5}  {medium:>6}  {unresolved:>6}  {pct:>8.1f}%  {drop_str:>5}")
    ablation_rows.append((label, high, medium, unresolved, pct, drop if ablate_sig else 0))

json.dump(ablation_rows, open('data/resolver_ablation.json', 'w'), indent=2)
print("\nSaved → data/resolver_ablation.json")
