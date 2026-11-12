"""
Depression-angle diversity, done properly (paper 2). Replaces train_angle_ablation.py, whose baseline and ablation
rows were scored on DIFFERENT test sets (EOC 30/45/60 vs held-out 15 deg).

Design: fix a test angle T and keep it out of training. Train ConvNeXt-Small on angle set A (T not in A) with the
SAME number of chips per class in every condition (--per-class, split equally over the angles in A), so that
'more angles' never means 'more data'. Test = ATRNet-STAR SOC_40classes official test split at angle T.

Usage:
    venv/bin/python scripts/train_angle_heldout.py --test-angle 45 --train-angles 15 60 --seed 0
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

ap = argparse.ArgumentParser()
ap.add_argument('--test-angle', type=float, required=True)
ap.add_argument('--train-angles', type=float, nargs='+', required=True)
ap.add_argument('--per-class', type=int, default=360)
ap.add_argument('--epochs', type=int, default=15)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
assert a.test_angle not in a.train_angles
TAG = f"angle_test{int(a.test_angle)}_train{'+'.join(str(int(x)) for x in sorted(a.train_angles))}_n{a.per_class}_s{a.seed}"
OUT = Path('logs/angle_heldout'); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def rows(split, angles):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT local_path, target_type, depression_angle FROM defense.atrnet_images
                       WHERE experiment_setting='SOC_40classes' AND split=%s AND depression_angle = ANY(%s)
                       AND target_type IS NOT NULL AND target_type != '' ORDER BY id""", [split, list(angles)])
        r = [dict(x) for x in cur.fetchall()]
    conn.close()
    return r


def load(p):
    return np.log10(np.array(Image.open(p), np.float32) + 1.0)


tr_all = rows('train', a.train_angles)
classes = sorted({r['target_type'] for r in rows('train', [15.0, 30.0, 45.0, 60.0])})
by = defaultdict(list)
for r in tr_all: by[(r['target_type'], float(r['depression_angle']))].append(r)
per_angle = a.per_class // len(a.train_angles)
train = []
for c in classes:
    for ang in a.train_angles:
        g = by[(c, float(ang))]
        train += [g[i] for i in rng.permutation(len(g))[:per_angle]]
test = rows('test', [a.test_angle])
with ThreadPoolExecutor(8) as pool:          # threads, not processes: module-level code + macOS spawn hung (2026-11-11)
    Xtr = np.stack(list(pool.map(load, [r['local_path'] for r in train])))
    Xte = np.stack(list(pool.map(load, [r['local_path'] for r in test])))
ytr = np.array([classes.index(r['target_type']) for r in train]); yte = np.array([classes.index(r['target_type']) for r in test])
perm = rng.permutation(len(ytr)); nva = len(ytr) // 10; va, tr = perm[:nva], perm[nva:]
print(f'{TAG}: train {len(tr)} val {len(va)} test {len(yte)} ({len(classes)} classes)', flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


def prep(X, idx, aug):
    out = X[idx].copy()
    lo = np.percentile(out.reshape(len(idx), -1), 1, axis=1)[:, None, None]
    hi = np.percentile(out.reshape(len(idx), -1), 99, axis=1)[:, None, None]
    out = np.clip((out - lo) / (hi - lo + 1e-9), 0, 1)
    if aug:
        flip = rng.random(len(idx)) < 0.5; out[flip] = out[flip][:, :, ::-1]
    t = torch.from_numpy(np.ascontiguousarray(out))[:, None].repeat(1, 3, 1, 1)
    return (t - MEAN) / STD


def predict(model, X, idx):
    model.eval(); P = []
    with torch.no_grad():
        for s in range(0, len(idx), 256): P.append(model(prep(X, idx[s:s + 256], False).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(P)


model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes)); model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
best, state = -1, None
for ep in range(1, a.epochs + 1):
    model.train(); t0 = time.time(); order = rng.permutation(tr)
    for s in range(0, len(order), 128):
        b = order[s:s + 128]
        loss = F.cross_entropy(model(prep(Xtr, b, True).to(dev)), torch.from_numpy(ytr[b]).to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    sch.step()
    acc = float((predict(model, Xtr, va) == ytr[va]).mean())
    if acc > best: best, state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'  epoch {ep:2d} val acc {acc*100:.1f} ({time.time()-t0:.0f}s)', flush=True)
model.load_state_dict(state)
yp = predict(model, Xte, np.arange(len(yte)))
acc = float((yp == yte).mean())
print(f'TEST angle {a.test_angle:.0f}  acc {acc*100:.1f}  (chance {100/len(classes):.1f})', flush=True)
json.dump({'args': vars(a), 'val_acc': best, 'test_acc': acc, 'n_test': int(len(yte)), 'n_train': int(len(tr))},
          open(OUT / f'{TAG}.json', 'w'), indent=2)
