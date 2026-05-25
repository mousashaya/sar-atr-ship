"""
scripts/train_angle_ablation.py
=================================
Angle ablation: which depression angles contribute most to generalization?

Fix N=200 per group, vary which angles are available for training:
  - 15° only          (EOC baseline — already known: 43.7%)
  - 15° + 60°         (most different angles)
  - 15° + 30° + 60°   (skip 45°)
  - All 4 angles      (E3 full — already known: 86.1%)

Test always on held-out 15° images, same as E3.

Answers: which angle adds the most information on top of 15°?
Operational use: given your platform's available angles, what accuracy to expect?

Usage:
    python3 scripts/train_angle_ablation.py --angles 15 60
    python3 scripts/train_angle_ablation.py --angles 15 30 60
    python3 scripts/train_angle_ablation.py --all
"""

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR    = Path(__file__).parent.parent / 'logs' / 'angle_ablation'
SEED        = 42
MAX_PER_GROUP = 200

ANGLE_SETS = [
    [15.0, 60.0],
    [15.0, 30.0, 60.0],
]

KNOWN_RESULTS = {
    (15.0,):              43.7,   # EOC baseline
    (15.0, 30.0, 45.0, 60.0): 86.1,  # E3 full N=200
}


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_data(angles: list, max_per_group: int = MAX_PER_GROUP, seed: int = SEED):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND depression_angle IS NOT NULL
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    class_names = sorted(set(r['target_type'] for r in rows))
    rng = np.random.default_rng(seed)

    # Filter to requested angles
    rows_filtered = [r for r in rows if float(r['depression_angle']) in angles]

    groups = defaultdict(list)
    for r in rows_filtered:
        key = (r['target_type'], float(r['depression_angle']))
        groups[key].append(r)

    capped = []
    for key, recs in groups.items():
        recs = list(recs)
        rng.shuffle(recs)
        capped.extend(recs[:max_per_group])

    by_class = defaultdict(list)
    for r in capped:
        by_class[r['target_type']].append(r)

    train_recs, val_recs = [], []
    for cls, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * 0.15))
        val_recs.extend(recs[:n_val])
        train_recs.extend(recs[n_val:])

    # Test: held-out 15° images not in train/val
    train_ids = {r['id'] for r in train_recs} | {r['id'] for r in val_recs}
    test_recs = [r for r in rows
                 if float(r['depression_angle']) == 15.0
                 and r['id'] not in train_ids]

    angle_str = '+'.join(str(int(a)) for a in sorted(angles))
    print(f"  Angles={angle_str}°  N={max_per_group}/group  "
          f"Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Test (15°): {len(test_recs):,}")
    return train_recs, val_recs, test_recs, class_names


class ChipDataset(Dataset):
    def __init__(self, records, class_to_idx, transform):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        return self.transform(img), self.class_to_idx[rec['target_type']]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total      += len(labels)
    return total_loss / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def run(angles: list, epochs: int = 5, batch_size: int = 64,
        lr: float = 1e-4) -> dict:
    device    = get_device()
    angle_str = '+'.join(str(int(a)) for a in sorted(angles))
    print(f"\n{'='*55}")
    print(f"  Angle Ablation — angles={angle_str}°  N={MAX_PER_GROUP}")
    print(f"{'='*55}")

    train_recs, val_recs, test_recs, class_names = load_data(angles)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    model         = mod.build_model(num_classes=len(class_names)).to(device)

    train_loader = DataLoader(
        ChipDataset(train_recs, class_to_idx, transform),
        batch_size=batch_size, shuffle=True,
        num_workers=4, persistent_workers=True)
    val_loader = DataLoader(
        ChipDataset(val_recs, class_to_idx, val_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=4, persistent_workers=True)
    test_loader = DataLoader(
        ChipDataset(test_recs, class_to_idx, val_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=4, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / f'angle_ablation_{angle_str}_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch = 0.0, 0
    for ep in range(1, epochs + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val  = evaluate(model, val_loader, device)
        print(f"  Epoch {ep}/{epochs}  loss={loss:.4f}  val={val*100:.1f}%"
              f"  ({int(time.time()-t0)}s)")
        if val > best_val:
            best_val, best_epoch = val, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                      map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  Test accuracy (15° held-out): {test_acc*100:.1f}%")
    print(f"  Best val accuracy:            {best_val*100:.1f}%  (epoch {best_epoch})")

    result = {
        'angles':         sorted(angles),
        'angle_str':      angle_str,
        'n_angles':       len(angles),
        'max_per_group':  MAX_PER_GROUP,
        'best_val_acc':   round(best_val * 100, 1),
        'test_acc_15deg': round(test_acc * 100, 1),
        'epochs':         epochs,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    return result


def plot_results(results: list):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Add known results
    all_results = list(results)
    all_results.append({'angle_str': '15',         'n_angles': 1,
                        'test_acc_15deg': 43.7,    'angles': [15.0]})
    all_results.append({'angle_str': '15+30+45+60','n_angles': 4,
                        'test_acc_15deg': 86.1,    'angles': [15.0,30.0,45.0,60.0]})
    all_results.sort(key=lambda r: r['n_angles'])

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [r['angle_str'] + '°' for r in all_results]
    accs   = [r['test_acc_15deg'] for r in all_results]
    colors = ['#d62728' if r['n_angles'] in [1, 4] else 'steelblue'
              for r in all_results]

    bars = ax.bar(labels, accs, color=colors, edgecolor='white', linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)

    ax.axhline(97.2, color='green', linestyle=':', alpha=0.7,
               label='SOC ceiling (97.2%)')
    ax.axhline(43.7, color='gray',  linestyle=':', alpha=0.7,
               label='EOC baseline — 15° only (43.7%)')

    ax.set_xlabel('Training angles available', fontsize=12)
    ax.set_ylabel('Test accuracy on held-out 15° images (%)', fontsize=12)
    ax.set_title(f'Angle Ablation — N={MAX_PER_GROUP} per group\n'
                 f'Which depression angles contribute most?', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out = LOGS_DIR / 'angle_ablation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")

    print(f"\n  {'Angles':<20}  {'Test (15°)':>10}  {'vs 15° only':>12}")
    print(f"  {'-'*46}")
    for r in all_results:
        delta = r['test_acc_15deg'] - 43.7
        print(f"  {r['angle_str']+'°':<20}  {r['test_acc_15deg']:>9.1f}%  {delta:>+11.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles',     type=float, nargs='+',
                        help='Depression angles to train on (e.g. 15 60)')
    parser.add_argument('--epochs',     type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--all',        action='store_true',
                        help='Run all angle combinations')
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        results = []
        for angle_set in ANGLE_SETS:
            r = run(angle_set, epochs=args.epochs, batch_size=args.batch_size)
            results.append(r)
        plot_results(results)
        return

    if args.angles is None:
        parser.error("Provide --angles A1 A2 ... or --all")

    r = run(args.angles, epochs=args.epochs, batch_size=args.batch_size)
    plot_results([r])


if __name__ == '__main__':
    main()
