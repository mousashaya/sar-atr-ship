"""
scripts/train_azimuth_detector.py
==================================
Trains ConvNeXt to predict azimuth bin from SAR chip.
Trained ONLY on 15° depression angle, X-band SOC data.

This is the first step of the EOC-Depression routing pipeline:
  chip → azimuth detector → route to azimuth-specific vehicle classifier

The idea: azimuth (horizontal viewing direction) is readable at any depression
angle. Train on 15° to learn azimuth, then use at test time on 30°/45°/60°.

Usage:
    python3 scripts/train_azimuth_detector.py --name azimuth_detector_v1 --bins 4
    python3 scripts/train_azimuth_detector.py --name azimuth_detector_v1 --bins 8
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
from sklearn.metrics import confusion_matrix
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
    """Map azimuth angle (0-360) to bin index."""
    bin_size = 360.0 / n_bins
    return int(azimuth / bin_size) % n_bins


def bin_label(bin_idx: int, n_bins: int) -> str:
    bin_size = 360.0 / n_bins
    lo = int(bin_idx * bin_size)
    hi = int((bin_idx + 1) * bin_size)
    return f"{lo}°-{hi}°"


class AzimuthDataset(Dataset):
    def __init__(self, records, n_bins: int, transform=None):
        self.records   = records
        self.n_bins    = n_bins
        self.transform = transform

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
        label = azimuth_to_bin(rec['azimuth_angle'], self.n_bins)
        return img, label, rec['image_id']


def load_data(n_bins: int, seed: int = 42):
    """Load 15° X-band SOC training data, split into train/val."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, azimuth_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = 15
              AND band = 'X'
              AND split = 'train'
              AND azimuth_angle IS NOT NULL
              AND local_path IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()

    records = [dict(r) for r in rows]

    # Stratified split by azimuth bin
    rng = np.random.default_rng(seed)
    by_bin = {}
    for rec in records:
        b = azimuth_to_bin(rec['azimuth_angle'], n_bins)
        by_bin.setdefault(b, []).append(rec)

    train_recs, val_recs = [], []
    for b, items in by_bin.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.15))
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])

    return train_recs, val_recs


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    return acc, all_preds, all_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',        required=True)
    parser.add_argument('--bins',        type=int,   default=4,
                        help='Number of azimuth bins (default: 4 = 90° each)')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int,   default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Azimuth bins: {args.bins} ({360//args.bins}° each)")

    model_module    = importlib.import_module('models.convnext')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    print("Loading 15° X-band SOC training data...")
    train_recs, val_recs = load_data(args.bins)
    print(f"Train: {len(train_recs):,}  Val: {len(val_recs):,}")

    bin_counts = Counter(azimuth_to_bin(r['azimuth_angle'], args.bins) for r in train_recs)
    for b in range(args.bins):
        label = bin_label(b, args.bins)
        print(f"  Bin {b} ({label}): {bin_counts.get(b, 0):,} images")

    train_ds = AzimuthDataset(train_recs, args.bins, train_transform)
    val_ds   = AzimuthDataset(val_recs,   args.bins, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=args.bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(__file__).parent.parent / 'models' / 'saved' / f'azimuth_{args.name}'
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(__file__).parent.parent / 'logs' / f'train_{args.name}.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Azimuth detector — {args.name}  ({args.bins} bins)")
    log(f"Random baseline: {100/args.bins:.1f}%")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        log(f"Epoch {epoch}/{args.epochs} started at {datetime.now().strftime('%H:%M:%S')} ...")
        model.train()
        total_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        log(f"Epoch {epoch}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  "
            f"val_acc={val_acc*100:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    log(f"\nBest val accuracy: {best_val_acc*100:.1f}%")
    log(f"Random baseline:   {100/args.bins:.1f}%")

    # Test on held-out SOC test data at 15°
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, azimuth_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = 15 AND band = 'X'
              AND split = 'test'
              AND azimuth_angle IS NOT NULL AND local_path IS NOT NULL
        """)
        test_recs = [dict(r) for r in cur.fetchall()]
    conn.close()

    test_ds     = AzimuthDataset(test_recs, args.bins, transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, persistent_workers=True)

    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    test_acc, test_preds, test_true = evaluate(model, test_loader, device)
    cm = confusion_matrix(test_true, test_preds, labels=list(range(args.bins)))
    bin_labels = [bin_label(b, args.bins) for b in range(args.bins)]

    log(f"\nSOC TEST accuracy (15°): {test_acc*100:.1f}%")
    log(f"Confusion matrix:")
    header = ''.join(f"{l:>12}" for l in bin_labels)
    log(f"  {'':>12}{header}")
    for i, lbl in enumerate(bin_labels):
        row = ''.join(f"{cm[i,j]:>12}" for j in range(args.bins))
        log(f"  {lbl:>12}{row}")

    config = {
        'name': args.name, 'task': 'azimuth_detection',
        'n_bins': args.bins,
        'bin_size_deg': 360 // args.bins,
        'bin_labels': bin_labels,
        'training': '15deg_Xband_SOC_only',
        'best_val_acc': round(best_val_acc * 100, 1),
        'soc_test_acc': round(test_acc * 100, 1),
        'random_baseline': round(100 / args.bins, 1),
        'date': datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"\nDone. Weights saved to {save_dir}")
    log_file.close()


if __name__ == '__main__':
    main()
