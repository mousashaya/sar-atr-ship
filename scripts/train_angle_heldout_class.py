"""
Partial-library angle test (paper 2, 2026-11-11). Companion to train_angle_heldout.py / train_angle_all.py.

Question: when a target class has NO training chips at test angle B, how much does it help that the OTHER classes
do have chips at B (a realistic, partially filled reference library)? The earlier runs remove angle B for all classes
(the worst case: no library at B at all).

Design: train angles {A, B}. The 40 SOC classes are split into 4 fixed folds of 10. Classes in the held-out fold get
all --per-class chips at A only; the other 30 classes get --per-class chips split equally over A and B (same total per
class as before, so 'library at B' never means 'more data'). Test = official SOC test split at angle B, all 40
classes. Accuracy is reported separately for held-out-fold classes (the quantity of interest) and in-library classes.
Running folds 0-3 covers every class once as held out, so the pooled held-out accuracy is directly comparable to the
all-classes-held-out accuracy from train_angle_all.py at the same (A, B) and seed.

Also reports how often held-out test chips are predicted as an in-library class (a possible shortcut: 'looks like
angle B' -> 'must be one of the classes seen at B').

Usage:
    venv/bin/python scripts/train_angle_heldout_class.py --train-angle 30 --test-angle 45 --fold 0 --seed 0
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
ap.add_argument('--train-angle', type=float, required=True)
ap.add_argument('--test-angle', type=float, required=True)
ap.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3])
ap.add_argument('--per-class', type=int, default=230)
ap.add_argument('--epochs', type=int, default=15)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
assert a.train_angle != a.test_angle
A, B = float(a.train_angle), float(a.test_angle)
TAG = f"classho_train{int(A)}_test{int(B)}_fold{a.fold}_n{a.per_class}_s{a.seed}"
OUT = Path('logs/angle_heldout_class'); OUT.mkdir(parents=True, exist_ok=True)
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


classes = sorted({r['target_type'] for r in rows('train', [15.0, 30.0, 45.0, 60.0])})
fold_perm = np.random.default_rng(12345).permutation(len(classes))     # fixed folds, independent of --seed
held = {classes[i] for i in fold_perm[a.fold::4]}

by = defaultdict(list)
for r in rows('train', [A, B]): by[(r['target_type'], float(r['depression_angle']))].append(r)
train = []
for c in classes:
    plan = [(A, a.per_class)] if c in held else [(A, a.per_class // 2), (B, a.per_class // 2)]
    for ang, n in plan:
        g = by[(c, ang)]
        assert len(g) >= n, f'{c} at {ang}: only {len(g)} chips'
        train += [g[i] for i in rng.permutation(len(g))[:n]]
test = rows('test', [B])
with ThreadPoolExecutor(8) as pool:          # threads, not processes (macOS spawn hang, 2026-11-11)
    Xtr = np.stack(list(pool.map(load, [r['local_path'] for r in train])))
    Xte = np.stack(list(pool.map(load, [r['local_path'] for r in test])))
ytr = np.array([classes.index(r['target_type']) for r in train]); yte = np.array([classes.index(r['target_type']) for r in test])
perm = rng.permutation(len(ytr)); nva = len(ytr) // 10; va, tr = perm[:nva], perm[nva:]
print(f'{TAG}: train {len(tr)} val {len(va)} test {len(yte)} ({len(classes)} classes, {len(held)} held out at {B:.0f})',
      flush=True)

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
    if dev.type == 'mps': torch.mps.empty_cache()
    acc = float((predict(model, Xtr, va) == ytr[va]).mean())
    if acc > best: best, state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'  epoch {ep:2d} val acc {acc*100:.1f} ({time.time()-t0:.0f}s)', flush=True)
model.load_state_dict(state)
yp = predict(model, Xte, np.arange(len(yte)))
held_idx = np.array([classes[y] in held for y in yte])
held_ids = np.array([classes.index(c) for c in sorted(held)])
acc_all = float((yp == yte).mean())
acc_held = float((yp[held_idx] == yte[held_idx]).mean())
acc_lib = float((yp[~held_idx] == yte[~held_idx]).mean())
to_lib = float((~np.isin(yp[held_idx], held_ids)).mean())        # held-out chips predicted as an in-library class
per_class = {classes[k]: float((yp[yte == k] == k).mean()) for k in np.unique(yte)}
print(f'TEST angle {B:.0f}  held-out classes acc {acc_held*100:.1f}  in-library acc {acc_lib*100:.1f}  '
      f'all {acc_all*100:.1f}  held->library {to_lib*100:.1f}%', flush=True)
json.dump({'args': vars(a), 'held_classes': sorted(held), 'val_acc': best, 'test_acc_all': acc_all,
           'test_acc_held': acc_held, 'test_acc_library': acc_lib, 'held_pred_as_library': to_lib,
           'n_test_held': int(held_idx.sum()), 'n_test_library': int((~held_idx).sum()), 'n_train': int(len(tr)),
           'per_class_acc': per_class, 'y_true': yte.tolist(), 'y_pred': yp.tolist()},
          open(OUT / f'{TAG}.json', 'w'), indent=2)
