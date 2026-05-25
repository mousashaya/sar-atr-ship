"""
Analyze vessel type vs ship length clustering in SLC chips.
Shows whether length discriminates vessel classes and where confusion zones are.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CHIPS_DIR = '/Volumes/Machine Learning/sar_ship_chips_slc_gulf'

def ais_to_class(code):
    if 80 <= code <= 89: return 'Tanker'
    if 70 <= code <= 79: return 'Cargo'
    if code in range(31, 33) or code in range(50, 60): return 'Tug/Special'
    if code in range(60, 70) or code in range(90, 100): return 'Other'
    return 'Unknown'

# Load metadata
records = defaultdict(lambda: {'lengths': [], 'codes': []})
all_data = []

print("Loading chip metadata...")
for npz in sorted(glob.glob(f'{CHIPS_DIR}/**/*.npz', recursive=True)):
    try:
        f = np.load(npz, allow_pickle=True)
        code   = int(f['vessel_type'])
        length = float(f['length_m'])
        if length <= 0 or code < 0:
            continue
        cls = ais_to_class(code)
        records[cls]['lengths'].append(length)
        records[cls]['codes'].append(code)
        all_data.append((cls, length, code))
    except Exception:
        continue

print(f"\nLoaded {len(all_data)} chips\n")

# ── Per-class length statistics ────────────────────────────────────────────────
print(f"{'Class':<15} {'N':>5} {'Min':>6} {'p25':>6} {'Median':>8} {'p75':>6} {'Max':>6} {'Mean':>7}")
print('-' * 65)
for cls in ['Tanker', 'Cargo', 'Tug/Special', 'Other', 'Unknown']:
    lengths = np.array(records[cls]['lengths'])
    if len(lengths) == 0:
        continue
    print(f"{cls:<15} {len(lengths):>5} {lengths.min():>6.0f} "
          f"{np.percentile(lengths,25):>6.0f} {np.median(lengths):>8.0f} "
          f"{np.percentile(lengths,75):>6.0f} {lengths.max():>6.0f} {lengths.mean():>7.0f}")

# ── Length-based classification rules ─────────────────────────────────────────
print("\n\nLength-based rule accuracy (how well does length alone classify?):")
print("  Rule: <60m → Tug/Special, 60-120m → Cargo, >120m → Tanker\n")

correct = total = 0
confusion = defaultdict(lambda: defaultdict(int))
for cls, length, code in all_data:
    if cls == 'Other' or cls == 'Unknown':
        continue
    if length < 60:
        pred = 'Tug/Special'
    elif length < 120:
        pred = 'Cargo'
    else:
        pred = 'Tanker'
    confusion[cls][pred] += 1
    if pred == cls:
        correct += 1
    total += 1

print(f"  Overall accuracy (3-class, length only): {correct/total*100:.1f}%\n")
print(f"  {'True \\ Pred':<15} {'Tanker':>10} {'Cargo':>10} {'Tug/Special':>12}")
print('  ' + '-' * 50)
for true_cls in ['Tanker', 'Cargo', 'Tug/Special']:
    row = confusion[true_cls]
    total_row = sum(row.values())
    print(f"  {true_cls:<15} {row['Tanker']:>10} {row['Cargo']:>10} {row['Tug/Special']:>12}  "
          f"(recall {row[true_cls]/total_row*100:.0f}%)")

# ── Overlap zones ──────────────────────────────────────────────────────────────
print("\n\nOverlap zones — where length is ambiguous:")
for length_range, label in [((60,120), "60-120m (Cargo/Tug overlap)"),
                              ((120,200), "120-200m (Cargo/Tanker overlap)")]:
    lo, hi = length_range
    counts = defaultdict(int)
    for cls, length, _ in all_data:
        if lo <= length < hi and cls not in ('Other', 'Unknown'):
            counts[cls] += 1
    total = sum(counts.values())
    if total > 0:
        breakdown = ', '.join(f"{k}: {v/total*100:.0f}%" for k,v in sorted(counts.items()))
        print(f"  {label}: {total} chips — {breakdown}")

# ── Plot ───────────────────────────────────────────────────────────────────────
colors = {'Tanker': '#e74c3c', 'Cargo': '#3498db',
          'Tug/Special': '#2ecc71', 'Other': '#95a5a6'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: AIS type code vs length
ax = axes[0]
for cls in ['Tanker', 'Cargo', 'Tug/Special', 'Other']:
    lengths = records[cls]['lengths']
    codes   = records[cls]['codes']
    ax.scatter(codes, lengths, alpha=0.3, s=10, color=colors[cls], label=cls)
ax.set_xlabel('AIS Vessel Type Code')
ax.set_ylabel('Ship Length (m)')
ax.set_title('AIS Type Code vs Ship Length')
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram: length distribution per class
ax = axes[1]
bins = np.linspace(0, 400, 40)
for cls in ['Tanker', 'Cargo', 'Tug/Special']:
    lengths = records[cls]['lengths']
    ax.hist(lengths, bins=bins, alpha=0.5, color=colors[cls], label=cls, density=True)
ax.axvline(60,  color='black', linestyle='--', alpha=0.5, label='60m rule')
ax.axvline(120, color='gray',  linestyle='--', alpha=0.5, label='120m rule')
ax.set_xlabel('Ship Length (m)')
ax.set_ylabel('Density')
ax.set_title('Length Distribution by Class')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vessel_type_length_analysis.png', dpi=150)
print(f"\nPlot saved: vessel_type_length_analysis.png")
