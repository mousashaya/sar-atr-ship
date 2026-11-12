"""
Background-only / target-only controls with a STRONG model (paper 2).

Conditions (side S square centred on the target-box centre; MSTAR targets are centred, ATRNet-STAR uses its boxes):
  full     : whole chip
  bg       : S x S square around the target zero-filled  -> background only
  tgt      : everything outside the S x S square zero-filled -> target only
  maskpos  : input is ONLY the binary visibility map of 'bg' (no image) -> does the mask position alone leak the class?
The S x S size is the same for every chip, so the mask cannot leak target size.
Contrast stretch (p1-p99) per chip on VISIBLE pixels only (same as eval_mstar_background.py).

Model: ConvNeXt-Small, ImageNet weights, native 128 px input, grey -> 3 channels. AdamW 1e-4, cosine, hflip + +-8 px shift.
Best epoch chosen on a 10% validation split of the training set; test reported overall and per test group.

Usage:
    venv/bin/python scripts/train_mask.py --data atr_soc --cond bg --size 80 --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, choices=['mstar', 'atr_soc', 'atr_scene'])
ap.add_argument('--cond', required=True, choices=['full', 'bg', 'tgt', 'maskpos'])
ap.add_argument('--size', type=int, default=80)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=None)
ap.add_argument('--arch', default='convnext', choices=['convnext', 'saratrx'],
                help='saratrx = SARATR-X HiViT-Base (SAR MAE-pretrained, Li et al. TIP 2025); chips upsampled 128->224')
ap.add_argument('--bs', type=int, default=128)
a = ap.parse_args()
EPOCHS = a.epochs or {'mstar': 30, 'atr_soc': 10, 'atr_scene': 12}[a.data]
TAG = f'{a.data}_{a.cond}{"" if a.cond == "full" else a.size}_s{a.seed}'
OUT = Path('logs/mask_controls' if a.arch == 'convnext' else f'logs/mask_controls_{a.arch}'); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('/Volumes/Machine Learning/atr_cache')
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load(split):
    d = np.load(CACHE / f'{a.data}_{split}.npz', allow_pickle=False)
    X = d['X'].astype(np.float32)
    if a.data != 'mstar':
        X = np.log10(X + 1.0)                     # uint8 tifs -> log scale, like MSTAR
    return X, d['y'], d['cx'], d['cy'], d['group'], list(d['classes'])


def vis_mask(cx, cy, s=128):
    """True = visible. Square S x S centred on (cx, cy), shifted to stay inside the chip."""
    S = a.size
    x0 = int(np.clip(round(cx - S / 2), 0, s - S)); y0 = int(np.clip(round(cy - S / 2), 0, s - S))
    inner = np.zeros((s, s), bool); inner[y0:y0 + S, x0:x0 + S] = True
    return {'full': np.ones((s, s), bool), 'bg': ~inner, 'tgt': inner, 'maskpos': ~inner}[a.cond]


def prep(X, cx, cy, idx, aug):
    out = np.zeros((len(idx), 128, 128), np.float32)
    for k, i in enumerate(idx):
        v = vis_mask(cx[i], cy[i])
        if a.cond == 'maskpos':
            out[k] = v
        else:
            px = X[i][v]; lo, hi = np.percentile(px, [1, 99])
            out[k][v] = np.clip((px - lo) / (hi - lo + 1e-9), 0, 1)
        if aug:
            if rng.random() < 0.5: out[k] = out[k][:, ::-1]
            dy, dx = rng.integers(-8, 9, 2)
            out[k] = np.roll(np.roll(out[k], dy, 0), dx, 1)
            if dy > 0: out[k][:dy] = 0
            elif dy < 0: out[k][dy:] = 0
            if dx > 0: out[k][:, :dx] = 0
            elif dx < 0: out[k][:, dx:] = 0
    t = torch.from_numpy(out)[:, None].repeat(1, 3, 1, 1)
    if a.arch == 'saratrx':                   # HiViT needs 224 px; masking/stretch done at native 128 px first
        t = F.interpolate(t, size=224, mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (t - mean) / std


def run(model, X, y, cx, cy, idx, train=False, opt=None, bs=None):
    bs = bs or a.bs
    model.train(train); preds = []
    order = rng.permutation(idx) if train else idx
    with torch.set_grad_enabled(train):
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            x = prep(X, cx, cy, b, aug=train).to(dev)
            out = model(x)
            if train:
                loss = F.cross_entropy(out, torch.from_numpy(y[b]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                preds.append(out.argmax(1).cpu().numpy())
    return None if train else np.concatenate(preds)


def metrics(yt, yp, k):
    cm = np.zeros((k, k), int)
    for t, p in zip(yt, yp): cm[t, p] += 1
    pres = [i for i in range(k) if cm[i].sum()]
    f1 = [2 * cm[i, i] / (cm[i].sum() + cm[:, i].sum()) if cm[i, i] else 0 for i in pres]
    return dict(n=int(len(yt)), acc=float((yt == yp).mean()), bal_acc=float(np.mean([cm[i, i] / cm[i].sum() for i in pres])),
                macro_f1=float(np.mean(f1)), chance=1 / len(pres))


Xtr, ytr, cxtr, cytr, _, classes = load('train')
Xte, yte, cxte, cyte, gte, _ = load('test')
K = len(classes)
perm = rng.permutation(len(ytr)); nva = len(ytr) // 10
va, tr = perm[:nva], perm[nva:]
print(f'{TAG}: {K} classes  train {len(tr)}  val {len(va)}  test {len(yte)}  epochs {EPOCHS}', flush=True)

if a.arch == 'convnext':
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, K)
else:
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.saratrx import build_model
    model = build_model(K)                     # loads checkpoint-800.pth encoder; fresh fc_norm, RPE tables, head
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
if a.arch == 'convnext':
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
else:                                          # transformers fine-tune poorly without warmup: first epoch at 0.1x lr, then cosine
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, 1),
                                                      torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS - 1)], [1])
best, best_state = -1, None
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    run(model, Xtr, ytr, cxtr, cytr, tr, train=True, opt=opt); sch.step()
    if dev.type == "mps": torch.mps.empty_cache()
    acc = float((run(model, Xtr, ytr, cxtr, cytr, va) == ytr[va]).mean())
    if acc > best:
        best = acc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'  epoch {ep:2d} val acc {acc*100:.1f} ({time.time()-t0:.0f}s){" *" if acc == best else ""}', flush=True)

model.load_state_dict(best_state)
yp = run(model, Xte, yte, cxte, cyte, np.arange(len(yte)))
res = {'args': vars(a), 'classes': classes, 'val_acc': best, 'test': {'all': metrics(yte, yp, K)}}
for g in sorted(set(gte)):
    m = gte == g
    if m.sum() and len(set(gte)) > 1: res['test'][g] = metrics(yte[m], yp[m], K)
for g, m in res['test'].items():
    print(f'TEST {g:14s} n={m["n"]:6d} acc {m["acc"]*100:5.1f} balAcc {m["bal_acc"]*100:5.1f} macroF1 {m["macro_f1"]*100:5.1f} (chance {m["chance"]*100:.1f})', flush=True)
json.dump(res, open(OUT / f'{TAG}.json', 'w'), indent=2)
