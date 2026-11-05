#!/usr/bin/env python3
"""Plot validation accuracy curves for all 4 backbones."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

LOGS = {
    'ResNet-50':     'logs/ship_classify_grd_high_3cls_resnet50_resolved_matchslc/results.json',
    'DINOv2':        'logs/ship_classify_grd_high_3cls_dinov2_resolved_matchslc/results.json',
    'ConvNeXt-S':    'logs/ship_classify_grd_high_3cls_convnext_resolved_matchslc/results.json',
    'CROMA':         'logs/ship_classify_grd_high_3cls_croma_resolved_matchslc/results.json',
}
COLORS = {'ResNet-50': '#e07b39', 'DINOv2': '#4c8edd', 'ConvNeXt-S': '#a259c4', 'CROMA': '#2ca02c'}
STYLES = {'ResNet-50': '--', 'DINOv2': ':', 'ConvNeXt-S': '-.', 'CROMA': '-'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), sharey=False)

for name, path in LOGS.items():
    with open(path) as f:
        data = json.load(f)
    history = data['history']
    epochs  = [h['epoch'] for h in history]
    val_oa  = [h['val_oa'] * 100 for h in history]
    val_f1  = [h['val_f1'] * 100 for h in history]
    kw = dict(color=COLORS[name], linestyle=STYLES[name], linewidth=1.8, label=name)
    ax1.plot(epochs, val_oa, **kw)
    ax2.plot(epochs, val_f1, **kw)

for ax, ylabel, title in [
    (ax1, 'Val. OA (%)',     'Overall Accuracy'),
    (ax2, 'Val. Macro-F1 (%)', 'Macro-F1'),
]:
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(1, 30)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.1)
    ax.tick_params(labelsize=9)

ax1.legend(fontsize=8, loc='lower right')
plt.tight_layout(pad=1.0)
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("Saved → training_curves.png")

# Print final test values for reference
print("\nFinal val metrics (epoch 30):")
for name, path in LOGS.items():
    with open(path) as f:
        data = json.load(f)
    h = data['history'][-1]
    print(f"  {name:12s}: val_OA={h['val_oa']*100:.1f}%  val_F1={h['val_f1']*100:.1f}%  "
          f"  test_OA={data['test_oa']*100:.1f}%  test_F1={data['test_f1']*100:.1f}%")
