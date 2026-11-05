#!/usr/bin/env python3
"""Dataset statistics figure: class distribution + length histogram + passes histogram."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

LABELS_PATH = Path('data/resolved_labels.json')
with open(LABELS_PATH) as f:
    resolved = json.load(f)

# Separate HIGH-confidence vessels
high   = {m: v for m, v in resolved.items() if v['confidence'] == 'HIGH'}
medium = {m: v for m, v in resolved.items() if v['confidence'] == 'MEDIUM'}

CLASS_NAMES = ['Tanker', 'Cargo', 'Fishing', 'Tug/Special']
COLORS_H = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
COLORS_M = ['#98df8a', '#aec7e8', '#ffbb78', '#c5b0d5']

# ── Panel 1: class distribution (HIGH vs MEDIUM) ──
high_counts   = [sum(1 for v in high.values()   if v['label'] == c) for c in CLASS_NAMES]
medium_counts = [sum(1 for v in medium.values() if v['label'] == c) for c in CLASS_NAMES]

# ── Panel 2: vessel length distribution (HIGH only, 3 classes) ──
lengths_by_class = defaultdict(list)
for v in high.values():
    if v['label'] in ['Tanker', 'Cargo', 'Fishing'] and v.get('length_m'):
        try:
            l = float(v['length_m'])
            if 20 < l < 500:
                lengths_by_class[v['label']].append(l)
        except (ValueError, TypeError):
            pass

# ── Panel 3: passes per vessel (all resolved) ──
pass_counts = []
for v in resolved.values():
    if v['label'] != 'Unresolved':
        n = v.get('n_chips', 0)
        if n > 0:
            pass_counts.append(min(n, 20))  # cap display at 20

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

# Panel 1
x = np.arange(len(CLASS_NAMES))
w = 0.4
axes[0].bar(x - w/2, high_counts,   width=w, color=COLORS_H, label='HIGH',   alpha=0.9)
axes[0].bar(x + w/2, medium_counts, width=w, color=COLORS_M, label='MEDIUM', alpha=0.9)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Tanker', 'Cargo', 'Fish.', 'Tug'], fontsize=9)
axes[0].set_ylabel('Vessels', fontsize=9)
axes[0].set_title('Class Distribution', fontsize=9)
axes[0].legend(fontsize=8)
axes[0].tick_params(labelsize=8)
for i, (h, m) in enumerate(zip(high_counts, medium_counts)):
    if h > 0: axes[0].text(i - w/2, h + 5, str(h), ha='center', va='bottom', fontsize=7)
    if m > 0: axes[0].text(i + w/2, m + 5, str(m), ha='center', va='bottom', fontsize=7)

# Panel 2
bins = np.arange(0, 420, 30)
cls_colors = {'Tanker': '#2ca02c', 'Cargo': '#1f77b4', 'Fishing': '#ff7f0e'}
for cls, color in cls_colors.items():
    axes[1].hist(lengths_by_class[cls], bins=bins, alpha=0.6, color=color,
                 label=f'{cls} (n={len(lengths_by_class[cls])})', density=False)
axes[1].set_xlabel('Ship length (m)', fontsize=9)
axes[1].set_ylabel('Vessels', fontsize=9)
axes[1].set_title('Length Distribution (HIGH)', fontsize=9)
axes[1].legend(fontsize=7)
axes[1].tick_params(labelsize=8)
axes[1].axvline(200, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes[1].text(202, axes[1].get_ylim()[1]*0.85, '200m', fontsize=7, color='gray')


plt.tight_layout(pad=0.8)
plt.savefig('dataset_stats.png', dpi=150, bbox_inches='tight')
print("Saved → dataset_stats.png")
print(f"\nHIGH counts: {dict(zip(CLASS_NAMES, high_counts))}")
print(f"MEDIUM counts: {dict(zip(CLASS_NAMES, medium_counts))}")
print(f"Length samples: { {c: len(v) for c, v in lengths_by_class.items()} }")
