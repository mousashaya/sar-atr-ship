"""
scripts/train_azimuth_vehicle.py
=================================
Trains a vehicle classifier for ONE azimuth bin on 15° X-band SOC data.

This is the second step of the EOC-Depression routing pipeline:
  chip → azimuth detector → route here → vehicle class

Run once per bin. With 4 bins, run 4 times (bin 0, 1, 2, 3).

The idea: group vehicles by the horizontal angle they were imaged from.
Images in the same azimuth bin look similar regardless of depression angle,
so a classifier trained at 15° can still classify at 30°/45°/60°.

Usage:
    # Train all 4 bins (run sequentially):
    python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --bin 0
    python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --bin 1
    python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --bin 2
    python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --bin 3

    # Or train all bins in one call:
    python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --all-bins
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def azimuth_to_bin(azimuth: float, n_bins: int) -> int:
    bin_size = 360.0 / n_bins
    return int(azimuth / bin_size) % n_bins


def bin_label(bin_idx: int, n_bins: int) -> str:
    bin_size = 360.0 / n_bins
    lo = int(bin_idx * bin_size)
    hi = int((bin_idx + 1) * bin_size)
    return f"{lo}°-{hi}°"


class VehicleDataset(Dataset):
    def __init__(self, records, class_to_idx: dict, transform=None):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), color=(64, 64, 64))
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[rec['target_type']]
        return img, label, rec['image_id']


def load_bin_data(bin_idx: int, n_bins: int, seed: int = 42):
    """
    Load 15° X-band SOC data for a single azimuth bin.
    Returns (train_recs, val_recs, class_names).
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, azimuth_angle, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = 15
              AND band = 'X'
              AND split = 'train'
              AND azimuth_angle IS NOT NULL
              AND target_type IS NOT NULL
              AND local_path IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()

    # Filter to this azimuth bin
    records = [dict(r) for r in rows
               if azimuth_to_bin(r['azimuth_angle'], n_bins) == bin_idx]

    class_names = sorted(set(r['target_type'] for r in records))

    # Stratified val split (15%) by class
    rng = np.random.default_rng(seed)
    by_class = {}
    for rec in records:
        by_class.setdefault(rec['target_type'], []).append(rec)

    train_recs, val_recs = [], []
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.15))
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])

    return train_recs, val_recs, class_names


def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    return acc, [class_names[p] for p in all_preds], [class_names[t] for t in all_true]


def train_one_bin(bin_idx: int, n_bins: int, args, device, model_module,
                  transform, train_transform):
    label = bin_label(bin_idx, n_bins)
    print(f"\n{'='*60}")
    print(f"  Azimuth bin {bin_idx}  ({label})")
    print(f"{'='*60}")

    train_recs, val_recs, class_names = load_bin_data(bin_idx, n_bins)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_cls = len(class_names)

    print(f"Train: {len(train_recs):,}  Val: {len(val_recs):,}  Classes: {n_cls}")
    counts = Counter(r['target_type'] for r in train_recs)
    sparse = [c for c in class_names if counts.get(c, 0) < 50]
    if sparse:
        print(f"  Sparse classes (<50 images): {sparse}")

    train_ds = VehicleDataset(train_recs, class_to_idx, train_transform)
    val_ds   = VehicleDataset(val_recs,   class_to_idx, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=n_cls).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = (Path(__file__).parent.parent / 'models' / 'saved' /
                f'azimuth_vehicle_{args.name}_bin{bin_idx}of{n_bins}')
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = (Path(__file__).parent.parent / 'logs' /
                f'train_{args.name}_bin{bin_idx}.log')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Azimuth vehicle classifier — bin {bin_idx} ({label})")
    log(f"Classes ({n_cls}): {class_names}")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader, device, class_names)
        elapsed = time.time() - start
        log(f"Epoch {epoch}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
            f"val_acc={val_acc*100:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    log(f"\nBest val accuracy: {best_val_acc*100:.1f}%")

    # Quick SOC test at 15° for this bin
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, azimuth_angle, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = 15 AND band = 'X'
              AND split = 'test'
              AND azimuth_angle IS NOT NULL
              AND target_type IS NOT NULL AND local_path IS NOT NULL
        """)
        test_rows = cur.fetchall()
    conn.close()

    test_recs = [dict(r) for r in test_rows
                 if azimuth_to_bin(r['azimuth_angle'], n_bins) == bin_idx]

    if test_recs:
        test_ds     = VehicleDataset(test_recs, class_to_idx, transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, persistent_workers=True)
        model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
        test_acc, _, _ = evaluate(model, test_loader, device, class_names)
        log(f"SOC test accuracy (15°, bin {bin_idx}): {test_acc*100:.1f}%")
    else:
        test_acc = 0.0
        log(f"No SOC test images for bin {bin_idx}")

    config = {
        'name':          args.name,
        'task':          'azimuth_vehicle_classification',
        'bin_idx':       bin_idx,
        'n_bins':        n_bins,
        'bin_label':     label,
        'training':      '15deg_Xband_SOC_only',
        'class_names':   class_names,
        'n_classes':     n_cls,
        'best_val_acc':  round(best_val_acc * 100, 1),
        'soc_test_acc':  round(test_acc * 100, 1),
        'date':          datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"Weights saved to {save_dir}")
    log_file.close()

    return best_val_acc, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',        required=True,
                        help='Base name (e.g. azimuth_v1) — bin index appended automatically')
    parser.add_argument('--bins',        type=int, default=4,
                        help='Total number of azimuth bins (must match azimuth detector)')
    parser.add_argument('--bin',         type=int, default=None,
                        help='Which bin to train (0-indexed). Omit if --all-bins')
    parser.add_argument('--all-bins',    action='store_true',
                        help='Train all bins sequentially in one run')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int,   default=4)
    args = parser.parse_args()

    if args.bin is None and not args.all_bins:
        print("Error: specify --bin N or --all-bins")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    print(f"Bins: {args.bins}  ({360//args.bins}° each)")

    model_module    = importlib.import_module('models.convnext')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    bins_to_train = list(range(args.bins)) if args.all_bins else [args.bin]

    summary = {}
    for b in bins_to_train:
        val_acc, test_acc = train_one_bin(
            b, args.bins, args, device, model_module, transform, train_transform)
        summary[b] = {'val_acc': round(val_acc*100, 1),
                      'test_acc': round(test_acc*100, 1)}

    if len(bins_to_train) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — all {args.bins} azimuth bins")
        print(f"{'='*60}")
        print(f"  {'Bin':>6}  {'Range':>12}  {'Val':>8}  {'SOC Test':>10}")
        for b, m in summary.items():
            lbl = bin_label(b, args.bins)
            print(f"  {b:>6}  {lbl:>12}  {m['val_acc']:>7.1f}%  {m['test_acc']:>9.1f}%")


if __name__ == '__main__':
    main()
