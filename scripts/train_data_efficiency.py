"""
scripts/train_data_efficiency.py
==================================
E3: How much multi-angle data do you need to learn angle-invariant features?

Trains a SOC-style model with ALL 4 depression angles but caps the number of
images per (vehicle, angle) group to N. Then tests on held-out 15° images.

Run for N = 5, 10, 25, 50, 100, 200 to plot accuracy vs data size.
The breakeven point tells us how few labeled multi-angle images are needed
before a model generalizes — directly motivating the pseudo-labeling approach.

Usage:
    python3 scripts/train_data_efficiency.py --max-per-group 10
    python3 scripts/train_data_efficiency.py --max-per-group 50
    python3 scripts/train_data_efficiency.py --plot-curve   # plots saved results
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

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'data_efficiency'
SEED       = 42
N_VALUES   = [5, 10, 25, 50, 100, 200, 300, 400, 500]


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(max_per_group: int, seed: int = SEED):
    """
    Load SOC_40classes images, capped to max_per_group per (vehicle, angle).
    Returns train_recs, val_recs, test_recs (15° only), class_names.
    """
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

    # Cap to max_per_group per (vehicle, angle)
    groups = defaultdict(list)
    for r in rows:
        key = (r['target_type'], float(r['depression_angle']))
        groups[key].append(r)

    capped = []
    for key, recs in groups.items():
        recs = list(recs)
        rng.shuffle(recs)
        capped.extend(recs[:max_per_group])

    # Stratified train/val split (85/15) by vehicle
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

    # Test set: held-out 15° images (not in train/val)
    train_ids = {r['id'] for r in train_recs} | {r['id'] for r in val_recs}
    test_recs = [r for r in rows
                 if float(r['depression_angle']) == 15.0
                 and r['id'] not in train_ids]

    print(f"  Cap={max_per_group}/group  "
          f"Train: {len(train_recs):,}  "
          f"Val: {len(val_recs):,}  "
          f"Test (15°): {len(test_recs):,}  "
          f"Classes: {len(class_names)}")

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


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (model(imgs).argmax(1) == labels).sum().item()
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


# ── Main train run ─────────────────────────────────────────────────────────────

def run(max_per_group: int, epochs: int = 5, batch_size: int = 64,
        lr: float = 1e-4) -> dict:
    device = get_device()
    print(f"\n{'='*55}")
    print(f"  Data Efficiency — max_per_group={max_per_group}")
    print(f"{'='*55}")

    train_recs, val_recs, test_recs, class_names = load_data(max_per_group)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    mod        = importlib.import_module('models.convnext_small')
    transform  = mod.get_train_transform()
    val_transform = mod.get_transform()
    model      = mod.build_model(num_classes=len(class_names)).to(device)

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

    save_dir = MODELS_DIR / f'data_efficiency_n{max_per_group}_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch = 0.0, 0
    for ep in range(1, epochs + 1):
        t0  = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val  = evaluate(model, val_loader, device)
        print(f"  Epoch {ep}/{epochs}  loss={loss:.4f}  val={val*100:.1f}%"
              f"  ({int(time.time()-t0)}s)")
        if val > best_val:
            best_val, best_epoch = val, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    # Test on held-out 15° images
    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                      map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  Test accuracy (15° held-out): {test_acc*100:.1f}%")
    print(f"  Best val accuracy           : {best_val*100:.1f}%  (epoch {best_epoch})")

    result = {
        'max_per_group': max_per_group,
        'train_size':    len(train_recs),
        'val_size':      len(val_recs),
        'test_size':     len(test_recs),
        'best_val_acc':  round(best_val * 100, 1),
        'test_acc_15deg': round(test_acc * 100, 1),
        'epochs':        epochs,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    return result


# ── Curve plotting ────────────────────────────────────────────────────────────

def plot_curve():
    """Read saved results and plot accuracy vs N."""
    results = []
    for n in N_VALUES:
        cfg_path = MODELS_DIR / f'data_efficiency_n{n}_v1' / 'config.json'
        if cfg_path.exists():
            results.append(json.loads(cfg_path.read_text()))

    if not results:
        print("No results found. Run training first.")
        return

    ns       = [r['max_per_group'] for r in results]
    val_accs = [r['best_val_acc']   for r in results]
    tst_accs = [r['test_acc_15deg'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ns, val_accs, 'o-', color='steelblue', label='Val acc (all angles)')
    ax.plot(ns, tst_accs, 's--', color='tomato',   label='Test acc (15° held-out)')
    ax.axhline(97.2, color='green', linestyle=':', alpha=0.7,
               label='SOC ceiling (full data, 97.2%)')
    ax.axhline(43.7, color='gray', linestyle=':', alpha=0.7,
               label='EOC baseline (15°-only training, 43.7%)')
    ax.set_xlabel('Max images per (vehicle, angle) group')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Data Efficiency: How much multi-angle data is needed?\n'
                 'Each point = train on ALL 4 angles, capped to N per group, '
                 'test on held-out 15° images')
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / 'data_efficiency_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    print("\n  N    Val    Test(15°)")
    print("  " + "-"*25)
    for r in results:
        print(f"  {r['max_per_group']:>3}  {r['best_val_acc']:>5.1f}%  "
              f"{r['test_acc_15deg']:>5.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-group', type=int, default=None,
                        help='Images per (vehicle, angle) group')
    parser.add_argument('--epochs',        type=int, default=5)
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--all',           action='store_true',
                        help='Run all N values sequentially: 5,10,25,50,100,200')
    parser.add_argument('--plot-curve',    action='store_true',
                        help='Plot the accuracy vs N curve from saved results')
    args = parser.parse_args()

    if args.plot_curve:
        plot_curve()
        return

    if args.all:
        all_results = []
        for n in N_VALUES:
            r = run(n, epochs=args.epochs, batch_size=args.batch_size)
            all_results.append(r)
        print("\n\nFull sweep complete. Plotting curve...")
        plot_curve()
        return

    if args.max_per_group is None:
        parser.error("Provide --max-per-group N or --all")

    run(args.max_per_group, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
