"""
scripts/eval_mstar_background.py
================================
How much of MSTAR SOC accuracy comes from the background, not the vehicle?

SOC protocol: train 17 deg -> test 15 deg, 10 classes (same records as train_mstar_soc.py).
Each chip: centre 128x128 crop of log-magnitude. Variants:
  full        - whole 128x128 chip
  bg_mask{m}  - centre m x m square (vehicle + most of its shadow) set to 0, background only
  tgt_only{m} - only the centre m x m square kept, background set to 0
Contrast stretch (p1-p99) is computed from the VISIBLE pixels only, so hidden pixels cannot leak via scaling.

Classifier: small CNN trained from scratch (same for every variant), 3 seeds. Chance = 10%.

Usage:
    venv/bin/python scripts/eval_mstar_background.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ingest_mstar import read_mstar
from eval_clip_mstar import records, CLASSES, centre

OUT = Path('logs/mstar_background')
OUT.mkdir(parents=True, exist_ok=True)
MASKS = [48, 64, 80]
SEEDS = [0, 1, 2]
EPOCHS = 15


def load_log(rows):
    X, y = [], []
    for r in rows:
        _, mag = read_mstar(Path(r['local_path']))
        X.append(centre(np.log10(np.clip(mag, 1e-6, None))))
        y.append(CLASSES.index(r['target_short']))
    return np.stack(X).astype(np.float32), np.array(y)


def variant(X, kind, m=None):
    n, s, _ = X.shape
    vis = np.ones((s, s), bool)
    if kind != 'full':
        a, b = (s - m) // 2, (s - m) // 2 + m
        inner = np.zeros((s, s), bool); inner[a:b, a:b] = True
        vis = ~inner if kind == 'bg' else inner
    out = np.zeros_like(X)
    for i in range(n):
        v = X[i][vis]
        lo, hi = np.percentile(v, [1, 99])
        out[i][vis] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    return out


class Net(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        ch = [1, 32, 64, 128, 128]
        self.f = nn.Sequential(*[nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]),
                                               nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        self.h = nn.Linear(128, k)

    def forward(self, x):
        return self.h(self.f(x).mean((2, 3)))


def train_eval(Xtr, ytr, Xte, yte, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Xt, yt = torch.from_numpy(Xtr)[:, None], torch.from_numpy(ytr)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), 64):
            j = perm[i:i + 64]
            xb = Xt[j]
            if np.random.rand() < 0.5:   # light aug: random flips + small shifts
                xb = xb.flip(-1)
            dx, dy = np.random.randint(-4, 5, 2)
            xb = torch.roll(xb, (int(dx), int(dy)), (-2, -1))
            loss = F.cross_entropy(net(xb.to(dev)), yt[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval(); preds = []
    with torch.no_grad():
        Xe = torch.from_numpy(Xte)[:, None]
        for i in range(0, len(Xe), 256):
            preds.append(net(Xe[i:i + 256].to(dev)).argmax(1).cpu())
    p = torch.cat(preds).numpy()
    return float((p == yte).mean()), np.bincount(p, minlength=10).tolist()


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    Ltr, ytr = load_log(records(17.0))
    Lte, yte = load_log(records(15.0))
    print(f'train {len(ytr)}  test {len(yte)}  chance 10%', flush=True)
    jobs = [('full', None)] + [('bg', m) for m in MASKS] + [('tgt', m) for m in MASKS]
    res = {}
    for kind, m in jobs:
        name = 'full' if kind == 'full' else f'{kind}_{"mask" if kind == "bg" else "only"}{m}'
        Xtr, Xte = variant(Ltr, kind, m), variant(Lte, kind, m)
        np.save(OUT / f'example_{name}.npy', Xte[:3])
        t = time.time()
        accs = [train_eval(Xtr, ytr, Xte, yte, s, dev)[0] for s in SEEDS]
        res[name] = {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)), 'accs': accs}
        print(f'{name:12s} acc {np.mean(accs)*100:5.1f}% ± {np.std(accs)*100:.1f}  ({time.time()-t:.0f}s)', flush=True)
        json.dump(res, open(OUT / 'results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
