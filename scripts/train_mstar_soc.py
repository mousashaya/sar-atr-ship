"""
scripts/train_mstar_soc.py
===========================
MSTAR SOC baseline: train ConvNeXt on 17° images, test on 15° images.

Classic MSTAR benchmark — literature reports 97-99% with CNNs.
This establishes that our framework reproduces published results before
we run the prototypical and cross-dataset experiments.

10-class set: BMP2, BTR60, BTR70, 2S1, BRDM2, D7, T62, T72, ZSU234, ZIL131

Usage:
    python3 scripts/train_mstar_soc.py
    python3 scripts/train_mstar_soc.py --epochs 20
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
from ingest_mstar import read_mstar, mag_to_uint8

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
SEED = 42

# Normalise DB target_short names → clean 10-class labels
LABEL_MAP = {
    'BMP2':       'BMP2',
    'BTR60':      'BTR60',
    'BTR70_TRAN': 'BTR70',
    '2S1':        '2S1',
    'BRDM2_TRUC': 'BRDM2',
    'D7':         'D7',
    'T62':        'T62',
    'T72':        'T72',
    'ZSU23-4_GU': 'ZSU234',
    'ZIL131_TRU': 'ZIL131',
}


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_records(depression: float) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        placeholders = ','.join(['%s'] * len(LABEL_MAP))
        cur.execute(f"""
            SELECT local_path, target_short
            FROM defense.mstar_images
            WHERE desired_depression = %s
              AND target_short IN ({placeholders})
              AND local_path IS NOT NULL
        """, [depression] + list(LABEL_MAP.keys()))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


class MSTARDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = LABEL_MAP[rec['target_short']]
        try:
            _, mag = read_mstar(Path(rec['local_path']))
            if mag is None:
                raise ValueError("empty")
            img = Image.fromarray(mag_to_uint8(mag)).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


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
    parser.add_argument('--epochs',     type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")
    print("="*55)
    print("  MSTAR SOC: train 17°  →  test 15°")
    print("="*55)

    train_recs = load_records(17.0)
    test_recs  = load_records(15.0)

    # Val split: 15% of train
    rng = np.random.default_rng(SEED)
    by_class = {}
    for r in train_recs:
        by_class.setdefault(LABEL_MAP[r['target_short']], []).append(r)
    train_final, val_recs = [], []
    for cls, recs in by_class.items():
        recs = list(recs); rng.shuffle(recs)
        n_val = max(1, int(len(recs) * 0.15))
        val_recs.extend(recs[:n_val])
        train_final.extend(recs[n_val:])

    class_names  = sorted(set(LABEL_MAP.values()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"  Train: {len(train_final):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}")
    print(f"  Classes: {class_names}")

    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    model         = mod.build_model(num_classes=len(class_names)).to(device)

    train_loader = DataLoader(MSTARDataset(train_final, class_to_idx, transform),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader   = DataLoader(MSTARDataset(val_recs,    class_to_idx, val_transform),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True)
    test_loader  = DataLoader(MSTARDataset(test_recs,   class_to_idx, val_transform),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / 'mstar_soc_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

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
        val = evaluate(model, val_loader, device)
        print(f"  Epoch {ep}/{args.epochs}  loss={total_loss/total:.4f}  val={val*100:.1f}%"
              f"  ({int(time.time()-t0)}s)")
        if val > best_val:
            best_val, best_epoch = val, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')
            # Save class names for prototypical script
            (save_dir / 'config.json').write_text(json.dumps({
                'class_names': class_names, 'best_val_acc': round(best_val*100,1)}, indent=2))

    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                      map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n  Test accuracy (15°): {test_acc*100:.1f}%")
    print(f"  Best val accuracy  : {best_val*100:.1f}%  (epoch {best_epoch})")
    print(f"  Literature target  : ~97-99%")

    result = {
        'class_names':   class_names,
        'n_classes':     len(class_names),
        'train_size':    len(train_final),
        'val_size':      len(val_recs),
        'test_size':     len(test_recs),
        'best_val_acc':  round(best_val*100, 1),
        'test_acc_15deg': round(test_acc*100, 1),
        'epochs':        args.epochs,
        'train_dep':     17.0,
        'test_dep':      15.0,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    print(f"\nModel saved to {save_dir}")


if __name__ == '__main__':
    main()
