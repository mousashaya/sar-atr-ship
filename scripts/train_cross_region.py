"""
scripts/train_cross_region.py
=============================
Cross-region test: fine-tune CROMA on Gulf of Mexico Sentinel-1 GRD chips, test on other US regions.

Labels: data/resolved_labels_v2.json (region-neutral resolver), HIGH+MEDIUM ("uncontradicted") by default.
Train/val: Gulf chips, split by vessel (MMSI). A held-out set of Gulf vessels gives the in-region score.
Test:      Chesapeake, LA, NY, Puget chips of vessels NEVER seen in the Gulf data.
           Chips duplicated across region folders (overlapping scenes, e.g. NY/Chesapeake) are counted once.
Same CROMA recipe as train_ship_classify.py (30 epochs, AdamW 1e-4, cosine, best-val checkpoint).

Metrics per test set: OA, balanced accuracy (mean per-class recall), true macro-F1,
majority-class OA baseline, confusion matrix, number of vessels.

Usage:
    venv/bin/python scripts/train_cross_region.py                 # 3-class Tanker/Cargo/Fishing
    venv/bin/python scripts/train_cross_region.py --four-class
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'models')
from croma import build_model, chip_to_tensor

ap = argparse.ArgumentParser()
ap.add_argument('--four-class', action='store_true')
ap.add_argument('--high', action='store_true', help='HIGH-confidence labels only')
ap.add_argument('--cap', type=int, default=1000, help='max Gulf chips per class (train+val+in-region test)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--epochs', type=int, default=30)
args = ap.parse_args()

ROOT = Path('/Volumes/Machine Learning')
GULF = [ROOT / 'sar_ship_chips_gulf']
OTHER = {r: ROOT / f'sar_ship_chips_{r}' for r in ['chesapeake', 'la', 'ny', 'puget']}
CLASSES = ['Tanker', 'Cargo', 'Tug/Special', 'Fishing'] if args.four_class else ['Tanker', 'Cargo', 'Fishing']
TAG = f"xregion_{len(CLASSES)}cls_{'high' if args.high else 'medium'}_cap{args.cap}_s{args.seed}"
OUT = Path(f'logs/{TAG}'); OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(f'models/saved/{TAG}.pt')

LAB = json.load(open('data/resolved_labels_v2.json'))
OK = {'HIGH'} if args.high else {'HIGH', 'MEDIUM'}
TRUST = {m: v['label'] for m, v in LAB.items() if v['label'] in CLASSES and v['confidence'] in OK}


def scan(dirs):
    """-> list of (path, mmsi, class_idx); dedupe identical (scene, mmsi, pixel) chips across folders."""
    seen, recs = set(), []
    for d in dirs:
        for p in sorted(d.glob('**/*.npz')):
            try:
                f = np.load(p, allow_pickle=True)
                m = str(int(f['mmsi']))
                if m not in TRUST:
                    continue
                key = (str(f['scene_id']), m, int(f['pixel_row']), int(f['pixel_col']))
                if key in seen:
                    continue
                seen.add(key)
                recs.append((str(p), m, CLASSES.index(TRUST[m])))
            except Exception:
                continue
    return recs


class DS(Dataset):
    def __init__(self, recs, aug=False): self.r, self.aug = recs, aug
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        p, _, y = self.r[i]
        f = np.load(p, allow_pickle=True)
        return chip_to_tensor(f['chip_vv'], f['chip_vh'] if 'chip_vh' in f.files else None, augment=self.aug), y


def gulf_split(recs):
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r[2]][r[1]].append(r)
    tr, va, te = [], [], []
    for c in sorted(by):
        mm = sorted(by[c]); random.shuffle(mm)
        chips = [r for m in mm for r in by[c][m]]
        keep, n = [], 0                               # cap by whole vessels
        for m in mm:
            if n >= args.cap: break
            keep.append(m); n += len(by[c][m])
        tgt, got = 0.15 * n, [0, 0]
        for m in keep:
            k = 0 if got[0] < tgt else (1 if got[1] < tgt else 2)
            [va, te, tr][k].extend(by[c][m])
            if k < 2: got[k] += len(by[c][m])
        print(f'  Gulf {CLASSES[c]:12s} {len(mm):5d} vessels ({len(chips)} chips) -> used {len(keep)} vessels / {n} chips; '
              f'val {got[0]} test {got[1]}', flush=True)
    return tr, va, te


def predict(model, recs, dev):
    model.eval(); out = []
    with torch.no_grad():
        for x, _ in DataLoader(DS(recs), batch_size=64, num_workers=0):
            out.append(model(x.to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.array([], int)


def metrics(y, p):
    k = len(CLASSES)
    cm = np.zeros((k, k), int)
    for a, b in zip(y, p): cm[a, b] += 1
    present = [i for i in range(k) if cm[i].sum() > 0]
    rec = [cm[i, i] / cm[i].sum() for i in present]
    f1 = [2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum()) if cm[i, i] else 0.0 for i in present]
    return dict(n=int(len(y)), oa=float((y == p).mean()), bal_acc=float(np.mean(rec)), macro_f1=float(np.mean(f1)),
                majority_oa=float(cm.sum(1).max() / cm.sum()), classes_present=[CLASSES[i] for i in present],
                confusion=cm.tolist())


def main():
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'{TAG}  classes {CLASSES}  device {dev}', flush=True)

    gulf = scan(GULF)
    gulf_mmsi = {r[1] for r in gulf}
    tr, va, te_in = gulf_split(gulf)
    tests = {'gulf_heldout': te_in}
    for reg, d in OTHER.items():
        tests[reg] = [r for r in scan([d]) if r[1] not in gulf_mmsi]
    allo = {}
    for reg in OTHER:                                          # union of other regions, deduped by path key
        for r in tests[reg]: allo.setdefault((Path(r[0]).name, r[1]), r)
    tests['all_other'] = list(allo.values())
    for k, v in tests.items():
        cnt = np.bincount([r[2] for r in v], minlength=len(CLASSES))
        print(f'  test {k:13s} {len(v):6d} chips  {len({r[1] for r in v}):5d} vessels  per class {dict(zip(CLASSES, cnt.tolist()))}', flush=True)
    print(f'train {len(tr)}  val {len(va)}', flush=True)

    model = build_model(num_classes=len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    yva = np.array([r[2] for r in va]); best = -1
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); ls = []
        for x, y in DataLoader(DS(tr, aug=True), batch_size=64, shuffle=True, num_workers=0):
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.item())
        sch.step()
        m = metrics(yva, predict(model, va, dev))
        flag = ''
        if m['oa'] > best:
            best = m['oa']; torch.save(model.state_dict(), MODEL_PATH); flag = ' *'
        print(f'  epoch {ep:2d} loss {np.mean(ls):.4f} val OA {m["oa"]*100:.1f} balAcc {m["bal_acc"]*100:.1f} '
              f'F1 {m["macro_f1"]*100:.1f} ({time.time()-t0:.0f}s){flag}', flush=True)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    res = {'args': vars(args), 'classes': CLASSES, 'n_train': len(tr), 'n_val': len(va), 'tests': {}}
    for k, v in tests.items():
        if not v: continue
        y = np.array([r[2] for r in v]); m = metrics(y, predict(model, v, dev))
        m['n_vessels'] = len({r[1] for r in v}); res['tests'][k] = m
        print(f'TEST {k:13s} n={m["n"]:6d} vessels={m["n_vessels"]:5d}  OA {m["oa"]*100:5.1f} (majority {m["majority_oa"]*100:5.1f})  '
              f'balAcc {m["bal_acc"]*100:5.1f}  macroF1 {m["macro_f1"]*100:5.1f}  cm {m["confusion"]}', flush=True)
        json.dump(res, open(OUT / 'results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
