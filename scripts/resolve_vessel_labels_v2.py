"""
Region-neutral multi-signal vessel type resolver (v2, 2026-11-08).

Differences from resolve_vessel_labels.py (v1, used for the JSTARS/GRSL Gulf results):
  - NO length vote. v1 voted 'Tanker' for every ship >= 200 m ("almost nothing else this big in Gulf"),
    which is wrong in LA/NY/Chesapeake where most >= 200 m ships are container/bulk carriers.
    v1 also had an unreachable 'Fishing if < 30 m' branch (shadowed by '< 50 m -> Tug/Special').
  - Votes come only from AIS type code, Global Fishing Watch type and vessel-name keywords.
  - Length is kept only as a sanity veto: Fishing/Tug/Special labels on ships > 150 m are marked Unresolved.
  - Runs over several chip directories at once and records which regions each vessel was seen in.

Confidence rule unchanged from v1:
  top class >= 3 votes, or 2 votes with no conflict -> HIGH
  2 votes with conflict, or 1 vote with no conflict -> MEDIUM
  otherwise -> Unresolved

Output: data/resolved_labels_v2.json (keyed by MMSI). v1 file is NOT modified.

Usage:
    venv/bin/python scripts/resolve_vessel_labels_v2.py
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = '/Volumes/Machine Learning'
REGIONS = {
    'gulf':       [f'{ROOT}/sar_ship_chips_gulf', f'{ROOT}/sar_ship_chips_slc_gulf'],
    'chesapeake': [f'{ROOT}/sar_ship_chips_chesapeake'],
    'la':         [f'{ROOT}/sar_ship_chips_la'],
    'ny':         [f'{ROOT}/sar_ship_chips_ny'],
    'puget':      [f'{ROOT}/sar_ship_chips_puget'],
}
GFW_PATH = 'data/gfw_vessel_types.json'
OUT_PATH = 'data/resolved_labels_v2.json'
V1_PATH = 'data/resolved_labels.json'

TANKER_AIS, CARGO_AIS, FISHING_AIS = set(range(80, 90)), set(range(70, 80)), {30}
TUG_AIS = {31, 32} | set(range(50, 60))
TANKER_KW = ['TANKER', 'CRUDE', 'LNG', 'LPG', 'VLCC', 'ULCC', 'FPSO', 'CHEMICAL', 'BITUMEN', 'ASPHALT', 'GAS CARRIER']
CARGO_KW = ['BULK CARRIER', 'CONTAINER', 'CARGO', 'BULKER', 'CEMENT', 'GRAIN', 'COAL CARRIER', 'GENERAL CARGO']
TUG_KW = ['TUG', 'AHTS', 'PSV', 'OSV', 'OFFSHORE SUPPLY', 'ANCHOR HANDLING']
FISHING_KW = ['FISHING', 'TRAWL', 'SEINER']
MAX_SMALL_CLASS_LEN = 150.0


def ais_vote(c):
    if c in TANKER_AIS: return 'Tanker'
    if c in CARGO_AIS: return 'Cargo'
    if c in FISHING_AIS: return 'Fishing'
    if c in TUG_AIS: return 'Tug/Special'
    return None


def gfw_vote(t):
    t = (t or '').upper()
    if t in ('TANKER', 'BUNKER'): return 'Tanker'
    if t in ('CARGO', 'CARRIER', 'SEISMIC_VESSEL'): return 'Cargo'
    if t == 'FISHING': return 'Fishing'
    return None


def name_vote(n):
    n = (n or '').upper()
    for kws, c in ((TANKER_KW, 'Tanker'), (CARGO_KW, 'Cargo'), (TUG_KW, 'Tug/Special'), (FISHING_KW, 'Fishing')):
        if any(k in n for k in kws):
            return c
    return None


def resolve(ais_code, length, gfw_type, name):
    cast = [(s, v) for s, v in (('ais', ais_vote(ais_code)), ('gfw', gfw_vote(gfw_type)), ('name', name_vote(name))) if v]
    if not cast:
        return 'Unresolved', 0, []
    counts = Counter(v for _, v in cast)
    top, k = counts.most_common(1)[0]
    conflict = len(counts) > 1
    srcs = [s for s, v in cast if v == top]
    if k >= 3 or (k == 2 and not conflict): conf = 'HIGH'
    elif k == 2 or (k == 1 and not conflict): conf = 'MEDIUM'
    else: return 'Unresolved', 0, srcs
    if top in ('Fishing', 'Tug/Special') and length and length > MAX_SMALL_CLASS_LEN:
        return 'Unresolved', 0, srcs + ['length_veto']
    return top, conf, srcs


def main():
    gfw = json.load(open(GFW_PATH))
    by = defaultdict(lambda: {'codes': [], 'lens': [], 'names': [], 'regions': Counter()})
    for region, dirs in REGIONS.items():
        n = 0
        for d in dirs:
            for p in glob.glob(f'{d}/**/*.npz', recursive=True):
                try:
                    f = np.load(p, allow_pickle=True)
                    m = str(int(f['mmsi']))
                    by[m]['codes'].append(int(float(f['vessel_type'])))
                    by[m]['lens'].append(float(f['length_m']))
                    by[m]['names'].append(str(f['vessel_name']) if 'vessel_name' in f.files else '')
                    by[m]['regions'][region] += 1
                    n += 1
                except Exception:
                    continue
        print(f'{region:11s} {n:6d} chips', flush=True)

    out = {}
    for m, d in by.items():
        code = Counter(d['codes']).most_common(1)[0][0]
        lens = [x for x in d['lens'] if x and x > 0]
        length = float(np.median(lens)) if lens else 0.0
        g = gfw.get(m) or {}
        name = next((x for x in d['names'] if x and x not in ('nan', 'None')), '') or g.get('name', '')
        label, conf, srcs = resolve(code, length, g.get('ship_type', ''), name)
        out[m] = dict(mmsi=m, label=label, confidence=conf, sources=srcs, ais_code=code, length_m=length,
                      gfw_type=g.get('ship_type', ''), name=name, n_chips=len(d['codes']), regions=dict(d['regions']))
    json.dump(out, open(OUT_PATH, 'w'), indent=2)
    print(f'\nSaved {len(out)} vessels -> {OUT_PATH}')

    # vessels per class/confidence, by region (a vessel counts in every region it appears in)
    classes = ['Tanker', 'Cargo', 'Fishing', 'Tug/Special']
    print(f"\n{'region':11s} " + ' '.join(f'{c[:7]:>13s}' for c in classes) + '   Unres   (HIGH/MED vessels)')
    for region in REGIONS:
        vs = [v for v in out.values() if region in v['regions']]
        cells = [f"{sum(v['label']==c and v['confidence']=='HIGH' for v in vs):5d}/{sum(v['label']==c and v['confidence']=='MEDIUM' for v in vs):<5d}" for c in classes]
        print(f'{region:11s} ' + ' '.join(f'{x:>13s}' for x in cells) + f"   {sum(v['label']=='Unresolved' for v in vs):5d}")

    # how much do Gulf labels change vs v1 (the paper's labels)?
    v1 = json.load(open(V1_PATH))
    common = [m for m in v1 if m in out]
    same = sum(v1[m]['label'] == out[m]['label'] for m in common)
    print(f'\nGulf v1 vs v2: {len(common)} vessels in both, same label {same} ({same/len(common)*100:.1f}%)')
    trans = Counter((f"{v1[m]['label']}/{v1[m]['confidence']}", f"{out[m]['label']}/{out[m]['confidence']}")
                    for m in common if (v1[m]['label'], v1[m]['confidence']) != (out[m]['label'], out[m]['confidence']))
    for (a, b), n in trans.most_common(12):
        print(f'   {a:22s} -> {b:22s} {n}')


if __name__ == '__main__':
    main()
