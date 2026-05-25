"""
scripts/train_eoc_60only.py
============================
EOC experiment: train on 60° only, test on all other angles separately.

Hypothesis: 60° images capture the most discriminative structural features,
so a model trained solely on 60° should generalise to 15°, 30°, and 45°
better than a model trained on 15° alone.

Compare against:
  Train 15° only  → test others: EOC baseline study
  Train 60° only  → test others: this script

Results are printed per test-angle and saved to logs/eoc_60only/.

Usage:
    python3 scripts/train_eoc_60only.py
    python3 scripts/train_eoc_60only.py --epochs 10 --config 40class
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

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
from dataset import get_images, CONFIG_FIELD

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'eoc_60only'
SEED       = 42
TEST_ANGLES = [15.0, 30.0, 45.0]
TRAIN_ANGLE = 60.0


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


class ChipDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
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
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[rec['true_class']]


def load_records(angle, config_name, setting='SOC_40classes'):
    conn = psycopg2.connect(**DB_CONFIG)
    recs = get_images(conn, config_name, angle=angle, setting=setting)
    conn.close()
    return recs


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--config',     default='40class',
                        choices=['4class', '21class', '40class'])
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"\nDevice: {device}")
    print("=" * 60)
    print(f"  EOC Experiment: Train 60° only → Test 15° / 30° / 45°")
    print(f"  Config: {args.config}   Epochs: {args.epochs}")
    print("=" * 60)

    # ── Load 60° training data ────────────────────────────────────────────────
    print(f"\nLoading 60° training records...")
    train_recs = load_records(TRAIN_ANGLE, args.config)
    print(f"  60° records: {len(train_recs):,}")

    if not train_recs:
        print("ERROR: No 60° records found. Check DB and setting.")
        sys.exit(1)

    class_names  = sorted(set(r['true_class'] for r in train_recs))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_classes    = len(class_names)
    print(f"  Classes: {n_classes}")

    # Val split: 15% of 60° data
    rng = np.random.default_rng(SEED)
    by_class = {}
    for r in train_recs:
        by_class.setdefault(r['true_class'], []).append(r)
    train_final, val_recs = [], []
    for cls, recs in by_class.items():
        recs = list(recs); rng.shuffle(recs)
        n_val = max(1, int(len(recs) * 0.15))
        val_recs.extend(recs[:n_val])
        train_final.extend(recs[n_val:])
    print(f"  Train: {len(train_final):,}  Val: {len(val_recs):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    model         = mod.build_model(num_classes=n_classes).to(device)

    train_loader = DataLoader(
        ChipDataset(train_final, class_to_idx, transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, persistent_workers=True)
    val_loader = DataLoader(
        ChipDataset(val_recs, class_to_idx, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / 'eoc_60only_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\nTraining on 60° only...")
    best_val, best_epoch = 0.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total      += len(labels)
        val_acc = evaluate(model, val_loader, device)
        print(f"  Epoch {ep}/{args.epochs}  loss={total_loss/total:.4f}"
              f"  val(60°)={val_acc*100:.1f}%  ({int(time.time()-t0)}s)")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    print(f"\n  Best val (60°): {best_val*100:.1f}%  (epoch {best_epoch})")

    # ── Test on each held-out angle ───────────────────────────────────────────
    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                     map_location=device, weights_only=True))

    print(f"\n{'='*60}")
    print(f"  Test results (model trained on 60° only):")
    print(f"  {'Angle':>8}  {'Accuracy':>10}  {'N test':>8}")
    print(f"  {'-'*32}")

    results = []
    for angle in TEST_ANGLES:
        test_recs = load_records(angle, args.config)
        # Filter to same classes seen in training
        test_recs = [r for r in test_recs if r['true_class'] in class_to_idx]
        if not test_recs:
            print(f"  {int(angle):>7}°  {'no data':>10}")
            continue
        test_loader = DataLoader(
            ChipDataset(test_recs, class_to_idx, val_transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, persistent_workers=True)
        acc = evaluate(model, test_loader, device)
        print(f"  {int(angle):>7}°  {acc*100:>9.1f}%  {len(test_recs):>8,}")
        results.append({'train_angle': 60, 'test_angle': int(angle),
                        'accuracy': round(acc*100, 1), 'n_test': len(test_recs),
                        'config': args.config})

    # Also report 60° val as the "same-angle" ceiling
    print(f"  {'60° val':>8}  {best_val*100:>9.1f}%  (same-angle ceiling)")
    results.append({'train_angle': 60, 'test_angle': 60,
                    'accuracy': round(best_val*100, 1), 'note': 'val_ceiling',
                    'config': args.config})

    print(f"\n  Compare to EOC baseline (train all except 15°, test 15°): 43.7%")
    print(f"  Compare to SOC ceiling (train+test same angle):            97.2%")

    (LOGS_DIR / 'results.json').write_text(json.dumps(results, indent=2))
    (save_dir / 'config.json').write_text(json.dumps({
        'train_angle': 60, 'test_angles': TEST_ANGLES,
        'class_names': class_names, 'n_classes': n_classes,
        'best_val_acc': round(best_val*100, 1), 'epochs': args.epochs,
        'config': args.config,
    }, indent=2))

    print(f"\nResults saved to {LOGS_DIR}/results.json")
    print(f"Model saved to   {save_dir}/")


if __name__ == '__main__':
    main()
