"""
scripts/train_weighted_sampling.py
====================================
E3 variant: does weighted sampling (upweighting rare/hard angles) improve
accuracy without collecting more data?

Weights inversely proportional to angle frequency, so 60° images (fewest,
hardest) get the most attention during training.

Usage:
    python3 scripts/train_weighted_sampling.py --max-per-group 25
    python3 scripts/train_weighted_sampling.py --all
    python3 scripts/train_weighted_sampling.py --plot-curve
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'weighted_sampling'
SEED       = 42
N_VALUES   = [5, 10, 25, 50, 100, 200]


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_data(max_per_group: int, seed: int = SEED):
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

    groups = defaultdict(list)
    for r in rows:
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

    train_ids = {r['id'] for r in train_recs} | {r['id'] for r in val_recs}
    test_recs = [r for r in rows
                 if float(r['depression_angle']) == 15.0
                 and r['id'] not in train_ids]

    print(f"  Cap={max_per_group}/group  "
          f"Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Test (15°): {len(test_recs):,}  Classes: {len(class_names)}")
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


def make_weighted_sampler(train_recs):
    """Inverse-frequency weights by depression angle."""
    angle_counts = defaultdict(int)
    for r in train_recs:
        angle_counts[float(r['depression_angle'])] += 1
    total = len(train_recs)
    weights = [total / angle_counts[float(r['depression_angle'])]
               for r in train_recs]
    return WeightedRandomSampler(weights, num_samples=len(weights),
                                 replacement=True)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
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


def run(max_per_group: int, epochs: int = 5, batch_size: int = 64,
        lr: float = 1e-4) -> dict:
    device = get_device()
    print(f"\n{'='*55}")
    print(f"  Weighted Sampling — max_per_group={max_per_group}")
    print(f"{'='*55}")

    train_recs, val_recs, test_recs, class_names = load_data(max_per_group)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    model         = mod.build_model(num_classes=len(class_names)).to(device)

    sampler = make_weighted_sampler(train_recs)

    # Print angle weights for transparency
    angle_counts = defaultdict(int)
    for r in train_recs:
        angle_counts[float(r['depression_angle'])] += 1
    print("  Angle weights (inverse freq):")
    for angle in sorted(angle_counts):
        w = len(train_recs) / angle_counts[angle]
        print(f"    {int(angle)}°: {angle_counts[angle]} images → weight {w:.2f}x")

    train_loader = DataLoader(
        ChipDataset(train_recs, class_to_idx, transform),
        batch_size=batch_size, sampler=sampler,
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

    save_dir = MODELS_DIR / f'weighted_n{max_per_group}_v1'
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

    result = {
        'max_per_group':   max_per_group,
        'best_val_acc':    round(best_val * 100, 1),
        'test_acc_15deg':  round(test_acc * 100, 1),
        'epochs':          epochs,
        'method':          'weighted_sampling',
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    return result


def plot_curve():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load E3 baseline
    baseline = []
    for n in N_VALUES:
        p = MODELS_DIR / f'data_efficiency_n{n}_v1' / 'config.json'
        if p.exists():
            baseline.append(json.loads(p.read_text()))

    weighted = []
    for n in N_VALUES:
        p = MODELS_DIR / f'weighted_n{n}_v1' / 'config.json'
        if p.exists():
            weighted.append(json.loads(p.read_text()))

    if not baseline and not weighted:
        print("No results found.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline:
        ns  = [r['max_per_group'] for r in baseline]
        acc = [r['test_acc_15deg'] for r in baseline]
        ax.plot(ns, acc, 'o-', color='steelblue', label='E3 baseline (uniform sampling)')

    if weighted:
        ns  = [r['max_per_group'] for r in weighted]
        acc = [r['test_acc_15deg'] for r in weighted]
        ax.plot(ns, acc, 's--', color='tomato', label='Weighted sampling (angle-balanced)')

    ax.axhline(97.2, color='green', linestyle=':', alpha=0.7,
               label='SOC ceiling (97.2%)')
    ax.axhline(43.7, color='gray', linestyle=':', alpha=0.7,
               label='EOC baseline (43.7%)')
    ax.set_xlabel('Max images per (vehicle, angle) group')
    ax.set_ylabel('Test accuracy on held-out 15° images (%)')
    ax.set_title('Does weighted sampling help?\nUniform vs angle-balanced sampling')
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out = LOGS_DIR / 'weighted_sampling_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

    print("\n  N    Baseline   Weighted   Delta")
    print("  " + "-"*38)
    b_map = {r['max_per_group']: r['test_acc_15deg'] for r in baseline}
    w_map = {r['max_per_group']: r['test_acc_15deg'] for r in weighted}
    for n in N_VALUES:
        b = b_map.get(n, '--')
        w = w_map.get(n, '--')
        delta = f"{w-b:+.1f}%" if isinstance(b, float) and isinstance(w, float) else '--'
        print(f"  {n:>3}  {str(b):>7}%  {str(w):>7}%  {delta}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-group', type=int, default=None)
    parser.add_argument('--epochs',        type=int, default=5)
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--all',           action='store_true')
    parser.add_argument('--plot-curve',    action='store_true')
    args = parser.parse_args()

    if args.plot_curve:
        plot_curve()
        return

    if args.all:
        for n in N_VALUES:
            run(n, epochs=args.epochs, batch_size=args.batch_size)
        plot_curve()
        return

    if args.max_per_group is None:
        parser.error("Provide --max-per-group N or --all")

    run(args.max_per_group, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
