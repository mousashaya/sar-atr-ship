"""
scripts/train_eoc_depression_small.py
======================================
Train ConvNeXt-Small on EOC-Depression (15° only) and evaluate directly
on the EOC-Depression test set (30°/45°/60°).

This is a direct apples-to-apples comparison to the paper's Table 5
ConvNeXt result of 43.1%.

No routing, no hierarchy — one flat model, same restriction as the paper.

Usage:
    python3 scripts/train_eoc_depression_small.py --epochs 5
    python3 scripts/train_eoc_depression_small.py --epochs 10
"""

import argparse
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ChipDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
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
        return img, label


def load_train_data(seed=42):
    """Load EOC-Depression training data: 15° only."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND split = 'train'
              AND depression_angle = 15
              AND target_type IS NOT NULL
              AND local_path IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()

    records     = [dict(r) for r in rows]
    class_names = sorted(set(r['target_type'] for r in records))

    # Stratified val split 15% — monitors convergence only, NOT the EOC test
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


def load_test_data(class_names):
    """Load EOC-Depression test data: 30°/45°/60°."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND split = 'test'
              AND target_type IS NOT NULL
              AND local_path IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()
    # Only keep classes seen during training
    records = [dict(r) for r in rows if r['target_type'] in class_names]
    return records


class TestDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
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
        angle = float(rec['depression_angle'])
        return img, label, angle


def evaluate_val(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / total


def evaluate_test(model, loader, device):
    """Evaluate on test set, report overall + per-angle accuracy."""
    model.eval()
    by_angle = {}  # angle -> [correct, total]
    overall_correct, overall_total = 0, 0

    with torch.no_grad():
        for images, labels, angles in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct_mask = (preds == labels)
            overall_correct += correct_mask.sum().item()
            overall_total   += len(labels)
            for i, angle in enumerate(angles.tolist()):
                if angle not in by_angle:
                    by_angle[angle] = [0, 0]
                by_angle[angle][0] += correct_mask[i].item()
                by_angle[angle][1] += 1

    print(f"\n  EOC-Depression test results:")
    print(f"  {'Angle':>8}  {'Accuracy':>10}  {'N':>8}")
    print(f"  {'-'*30}")
    for angle in sorted(by_angle):
        c, t = by_angle[angle]
        print(f"  {angle:>7.0f}°  {c/t*100:>9.1f}%  {t:>8,}")
    overall = overall_correct / overall_total
    print(f"  {'Overall':>8}  {overall*100:>9.1f}%  {overall_total:>8,}")
    print(f"\n  Paper ConvNeXt SOTA: 43.1%  |  Our result: {overall*100:.1f}%  |  Delta: {overall*100-43.1:+.1f}%")
    return overall, by_angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--num-workers', type=int,   default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"ConvNeXt-Small | EOC-Depression | {args.epochs} epochs | lr={args.lr}")

    from models import convnext_small as model_module
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    print("\nLoading data...")
    train_recs, val_recs, class_names = load_train_data()
    test_recs = load_test_data(class_names)
    print(f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Test: {len(test_recs):,}  Classes: {len(class_names)}")

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_loader = DataLoader(
        ChipDataset(train_recs, class_to_idx, train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=True)

    val_loader = DataLoader(
        ChipDataset(val_recs, class_to_idx, transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True)

    test_loader = DataLoader(
        TestDataset(test_recs, class_to_idx, transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / f'convnext_small_eoc_depression_flat_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate_val(model, val_loader, device)
        elapsed = time.time() - start
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc*100:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    print(f"\n  Best val acc: {best_val_acc*100:.1f}%")

    # Load best weights and evaluate on real EOC test set
    print("\nLoading best weights for EOC test evaluation...")
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    overall_acc, by_angle = evaluate_test(model, test_loader, device)

    config = {
        'model':        'convnext_small',
        'condition':    'EOC-Depression',
        'restriction':  '15° training only',
        'epochs':       args.epochs,
        'lr':           args.lr,
        'best_val_acc': round(best_val_acc * 100, 1),
        'test_acc':     round(overall_acc * 100, 1),
        'by_angle':     {str(a): round(v[0]/v[1]*100, 1) for a, v in by_angle.items()},
        'date':         datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f"\n  Results saved to {save_dir}")


if __name__ == '__main__':
    main()
