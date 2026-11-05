#!/usr/bin/env python3
"""Generate confusion matrix figure from saved CROMA classification model."""

import json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, 'models')
from croma import build_model as build_croma, chip_to_tensor

CHIPS_DIR   = Path('/Volumes/Machine Learning/sar_ship_chips_gulf')
LABELS_PATH = Path('data/resolved_labels.json')
MODEL_PATH  = Path('models/saved/ship_classify_grd_high_3cls_croma_resolved_matchslc/best.pt')
OUT_PATH    = Path('confusion_matrix_croma.png')

CLASS_NAMES = ['Tanker', 'Cargo', 'Fishing']
SLC_COUNTS  = {'Tanker': 944, 'Cargo': 287, 'Fishing': 52}
SEED        = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Load labels
with open(LABELS_PATH) as f:
    resolved = json.load(f)
trusted = {mmsi: v for mmsi, v in resolved.items()
           if v['label'] != 'Unresolved' and v['confidence'] == 'HIGH'}

# Scan chips — same split logic as training
by_class = defaultdict(list)
for npz in sorted(CHIPS_DIR.glob('**/*.npz')):
    try:
        f    = np.load(npz, allow_pickle=True)
        mmsi = str(int(f['mmsi']))
        info = trusted.get(mmsi)
        if info is None: continue
        cls  = info['label']
        if cls not in CLASS_NAMES: continue
        by_class[cls].append(npz)
    except Exception:
        continue

# Reproduce same 70/15/15 split
test_recs = []
for cls, paths in by_class.items():
    random.shuffle(paths)
    cap   = SLC_COUNTS[cls]
    paths = paths[:cap]
    n     = len(paths)
    n_val = max(1, int(n * 0.15))
    n_tst = max(1, int(n * 0.15))
    idx   = CLASS_NAMES.index(cls)
    test_recs += [(p, idx) for p in paths[n_val:n_val + n_tst]]

print(f"Test set: {len(test_recs)} chips")

# Load model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model  = build_croma(num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Run inference
cm = np.zeros((3, 3), dtype=int)
with torch.no_grad():
    for path, true_idx in test_recs:
        try:
            f   = np.load(path, allow_pickle=True)
            chip_vh = f['chip_vh'] if 'chip_vh' in f else None
            t   = chip_to_tensor(f['chip_vv'], chip_vh, augment=False).unsqueeze(0).to(device)
            pred = model(t).argmax(dim=1).item()
            cm[true_idx, pred] += 1
        except Exception as e:
            continue

print("Confusion matrix (rows=true, cols=pred):")
print(cm)

# Compute per-row percentages
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

# Plot
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label', fontsize=11)

for i in range(3):
    for j in range(3):
        val = cm_pct[i, j]
        n   = cm[i, j]
        color = 'white' if val > 55 else 'black'
        ax.text(j, i, f'{val:.0f}%\n({n})', ha='center', va='center',
                fontsize=10, color=color, fontweight='bold' if i == j else 'normal')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Row %', fontsize=9)

ax.set_title('CROMA — 3-class test set', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_PATH}")
