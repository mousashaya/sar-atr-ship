"""
scripts/train_angle.py
======================
Trains ConvNeXt to predict depression angle from SAR chip.

This is the first step of the hierarchical pipeline:
  angle classifier → route to angle-specific vehicle classifier

4 output classes: 15°, 30°, 45°, 60°

Usage:
    python3 scripts/train_angle.py --name "angle_detector_v1"
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from datetime import datetime

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


ANGLE_CLASSES = [15.0, 30.0, 45.0, 60.0]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class AngleDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform
        self.angle_to_idx = {a: i for i, a in enumerate(ANGLE_CLASSES)}

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
        label = self.angle_to_idx[rec['depression_angle']]
        return img, label, rec['image_id']


def load_data(setting='SOC_40classes'):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, depression_angle, split
            FROM defense.atrnet_images
            WHERE experiment_setting = %s
              AND depression_angle IS NOT NULL
              AND local_path IS NOT NULL
              AND split IS NOT NULL
        """, (setting,))
        rows = cur.fetchall()
    conn.close()

    train_recs = [dict(r) for r in rows if r['split'] == 'train']
    test_recs  = [dict(r) for r in rows if r['split'] == 'test']

    # Carve val from train (15%)
    rng = np.random.default_rng(42)
    by_angle = {}
    for rec in train_recs:
        a = rec['depression_angle']
        by_angle.setdefault(a, []).append(rec)

    val_recs   = []
    train_only = []
    for angle, items in by_angle.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.15))
        val_recs.extend(items[:n_val])
        train_only.extend(items[n_val:])

    return train_only, val_recs, test_recs


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    return acc, all_preds, all_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',       required=True)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=0.0001)
    parser.add_argument('--num-workers',type=int,   default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_module = importlib.import_module('models.convnext')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    print("Loading data...")
    train_recs, val_recs, test_recs = load_data()
    print(f"Train: {len(train_recs):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}")
    for a in ANGLE_CLASSES:
        n = sum(1 for r in train_recs if r['depression_angle'] == a)
        print(f"  {a}°: {n:,} train images")

    train_ds = AngleDataset(train_recs, train_transform)
    val_ds   = AngleDataset(val_recs,   transform)
    test_ds  = AngleDataset(test_recs,  transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(__file__).parent.parent / 'models' / 'saved' / f'angle_{args.name}'
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(__file__).parent.parent / 'logs' / f'train_{args.name}.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Angle detector — {args.name}")
    log(f"Classes: {[f'{a}°' for a in ANGLE_CLASSES]}")

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

    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    test_acc, test_preds, test_true = evaluate(model, test_loader, device)
    angle_names = [f'{a}°' for a in ANGLE_CLASSES]
    cm = confusion_matrix(test_true, test_preds, labels=list(range(4)))

    log(f"\nTEST accuracy: {test_acc*100:.1f}%")
    log(f"Confusion matrix (rows=true, cols=predicted):")
    header = ''.join(f"{c:>10}" for c in angle_names)
    log(f"  {'':>8}{header}")
    for i, label in enumerate(angle_names):
        row = ''.join(f"{cm[i,j]:>10}" for j in range(4))
        log(f"  {label:>8}{row}")

    config = {
        'name': args.name, 'task': 'angle_detection',
        'classes': angle_names, 'best_val_acc': round(best_val_acc * 100, 1),
        'test_acc': round(test_acc * 100, 1), 'date': datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"\nDone. Weights saved to {save_dir}")
    log_file.close()


if __name__ == '__main__':
    main()
