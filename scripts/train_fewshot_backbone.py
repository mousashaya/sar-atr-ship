"""
scripts/train_fewshot_backbone.py
==================================
Backbones for the novel-class transfer experiment: does depression-angle
diversity during training help a frozen backbone generalize to classes it
never saw, not just angles it never saw?

Trains ConvNeXt-Small classifiers on the 30 BASE classes only (10 target
types held out entirely — see NOVEL_CLASSES). One backbone per angle
condition, same protocol as train_angle_ablation.py so results are directly
comparable to the known-class angle-ablation numbers:
  - 15° only            (low diversity)
  - 15° + 60°
  - 15° + 30° + 60°
  - All 4 angles        (high diversity)

scripts/eval_fewshot_novel.py then uses these as frozen feature extractors
to do prototypical few-shot classification on the 10 held-out classes.

Usage:
    python3 scripts/train_fewshot_backbone.py --angles 15
    python3 scripts/train_fewshot_backbone.py --all
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR    = Path(__file__).parent.parent / 'models' / 'saved'
SEED          = 42
MAX_PER_GROUP = 200

# Held out entirely from base training — one per multi-member subclass,
# proportional to Bus/Car/Truck data volume (2/4/4). Every subclass keeps
# at least one sibling in the base set; all 4 Special subclasses (each a
# singleton) stay fully in base. See project memory (atrnet_star_rewrite)
# for the full split rationale.
NOVEL_CLASSES = [
    'MAXUS_V80',                 # Bus / Medium_Bus
    'Dongfeng_Forthing_Lingzhi', # Bus / Small_Bus
    'Mitsubishi_Outlander_2003', # Car / Compact_SUV
    'Hongqi_h5',                 # Car / Large_Car
    'Chery_Arrizo 5',            # Car / Medium_Car
    'Chevrolet_Blazer_1998',     # Car / Medium_SUV
    'SHACMAN_DeLong_M3000',      # Truck / Heavy_FT
    'SHACMAN_DeLong_X3000',      # Truck / Heavy_ST
    'WAW_Aochi_1800',            # Truck / Light_DT
    'WAW_Aochi_Hongrui',         # Truck / Medium_TT
]

ANGLE_SETS = [
    [15.0],
    [15.0, 60.0],
    [15.0, 30.0, 60.0],
    [15.0, 30.0, 45.0, 60.0],
]


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_base_class_data(angles: list, max_per_group: int = MAX_PER_GROUP,
                          seed: int = SEED):
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

    novel_set = set(NOVEL_CLASSES)
    rows = [r for r in rows if r['target_type'] not in novel_set]
    class_names = sorted(set(r['target_type'] for r in rows))
    assert len(class_names) == 30, f"expected 30 base classes, got {len(class_names)}"

    rng = np.random.default_rng(seed)
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

    angle_str = '+'.join(str(int(a)) for a in sorted(angles))
    print(f"  Base classes: {len(class_names)}  (10 novel held out entirely)")
    print(f"  Angles={angle_str}°  N={max_per_group}/group  "
          f"Train: {len(train_recs):,}  Val: {len(val_recs):,}")
    return train_recs, val_recs, class_names


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
    print(f"\n{'='*60}")
    print(f"  Few-shot backbone — base classes only — angles={angle_str}°")
    print(f"{'='*60}")

    train_recs, val_recs, class_names = load_base_class_data(angles)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / f'fewshot_backbone_{angle_str}_v1'
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

    result = {
        'angles':        sorted(angles),
        'angle_str':      angle_str,
        'n_angles':       len(angles),
        'max_per_group':  MAX_PER_GROUP,
        'best_val_acc':   round(best_val * 100, 1),
        'best_epoch':     best_epoch,
        'epochs':         epochs,
        'class_names':    class_names,
        'model_type':     'convnext_small',
        'novel_classes':  NOVEL_CLASSES,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    print(f"\n  Saved: {save_dir}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles',     type=float, nargs='+',
                        help='Depression angles to train on (e.g. 15 60)')
    parser.add_argument('--epochs',     type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--all',        action='store_true',
                        help='Run all 4 angle conditions sequentially')
    args = parser.parse_args()

    if args.all:
        for angle_set in ANGLE_SETS:
            run(angle_set, epochs=args.epochs, batch_size=args.batch_size)
        return

    if args.angles is None:
        parser.error("Provide --angles A1 A2 ... or --all")

    run(args.angles, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
