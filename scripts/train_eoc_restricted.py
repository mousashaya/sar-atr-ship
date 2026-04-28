"""
scripts/train_eoc_restricted.py
================================
Trains angle-specific classifiers with RESTRICTED training data
as defined by the ATRNet-STAR benchmark (Table 4).

This produces LEGITIMATE benchmark comparisons to Table 5.

EOC condition → training restriction:
  EOC-Scene:        simple scenes only (sandstone + bare_soil), all 4 angles
  EOC-Azimuth:      azimuth 0°-60° only, all 4 angles
  EOC-Band:         X-band only, all 4 angles
  EOC-Polarization: HH polarization only, all 4 angles

For each condition we train 4 angle-specific classifiers (15/30/45/60°)
then eval_eoc.py routes chips to the right one.

Usage:
    python3 scripts/train_eoc_restricted.py --condition EOC-Scene   --name scene_restricted_v1
    python3 scripts/train_eoc_restricted.py --condition EOC-Azimuth --name azimuth_restricted_v1
    python3 scripts/train_eoc_restricted.py --condition EOC-Band    --name band_restricted_v1
    python3 scripts/train_eoc_restricted.py --condition EOC-Pol     --name pol_restricted_v1
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

ANGLE_CLASSES   = [15.0, 30.0, 45.0, 60.0]
SIMPLE_SCENES   = ['sandstone', 'bare_soil']


# ── Restriction filters ───────────────────────────────────────────────────────

CONDITION_FILTERS = {
    'EOC-Scene': {
        'description': 'Simple scenes only (sandstone + bare_soil)',
        'where': "AND scene = ANY(ARRAY['sandstone', 'bare_soil'])",
        'params': [],
    },
    'EOC-Azimuth': {
        'description': 'Azimuth 0°-60° only',
        'where': "AND azimuth_angle BETWEEN 0 AND 60",
        'params': [],
    },
    'EOC-Band': {
        'description': 'X-band only',
        'where': "AND band = 'X'",
        'params': [],
    },
    'EOC-Pol': {
        'description': 'HH polarization only',
        'where': "AND polarization = 'HH'",
        'params': [],
    },
}


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
        return img, label, rec['image_id']


def load_restricted_data(condition: str, angle: float, seed: int = 42):
    """Load restricted SOC training data for one condition and one angle."""
    filt = CONDITION_FILTERS[condition]
    query = f"""
        SELECT id AS image_id, local_path, target_type
        FROM defense.atrnet_images
        WHERE experiment_setting = 'SOC_40classes'
          AND split = 'train'
          AND depression_angle = %s
          AND target_type IS NOT NULL
          AND local_path IS NOT NULL
          {filt['where']}
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (angle,))
        rows = cur.fetchall()
    conn.close()

    records     = [dict(r) for r in rows]
    class_names = sorted(set(r['target_type'] for r in records))

    # Stratified val split 15%
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


def train_one_angle(condition, angle, args, device, model_module,
                    transform, train_transform):
    print(f"\n  --- {condition}  angle={angle}° ---")

    train_recs, val_recs, class_names = load_restricted_data(condition, angle)
    if not train_recs:
        print(f"  No training data — skipping")
        return None, None

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Classes: {len(class_names)}")

    train_ds = ChipDataset(train_recs, class_to_idx, train_transform)
    val_ds   = ChipDataset(val_recs,   class_to_idx, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    cond_tag = condition.replace('-', '_').lower()
    # Name matches eval_eoc.py pattern: convnext_{config}_angle{A}_v1
    # config = f'{args.model_type}_40class_{cond_tag}'
    save_dir = (Path(__file__).parent.parent / 'models' / 'saved' /
                f'convnext_{args.model_type}_40class_{cond_tag}_angle{int(angle)}_v1')
    save_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc*100:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    config = {
        'name':             args.name,
        'model_type':       args.model_type,
        'model_module':     f'models.convnext_{args.model_type}',
        'condition':        condition,
        'restriction':      CONDITION_FILTERS[condition]['description'],
        'angle':            angle,
        'class_names':      class_names,
        'best_val_acc':     round(best_val_acc * 100, 1),
        'date':             datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f"  Best val acc: {best_val_acc*100:.1f}%  → {save_dir.name}")
    return best_val_acc, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True,
                        choices=list(CONDITION_FILTERS.keys()),
                        help='EOC condition to train for')
    parser.add_argument('--name',        required=True)
    parser.add_argument('--model-type',  default='small',
                        choices=['small', 'base'],
                        help='convnext_small (paper baseline) or convnext_base (our model)')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int,   default=4)
    args = parser.parse_args()
    # argparse converts --model-type to args.model_type
    if not hasattr(args, 'model_type'):
        args.model_type = 'small'

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Condition: {args.condition}")
    print(f"Restriction: {CONDITION_FILTERS[args.condition]['description']}")
    print(f"Model: convnext_{args.model_type}")

    model_module    = importlib.import_module(f'models.convnext_{args.model_type}')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    cond_tag = args.condition.replace('-', '_').lower()
    summary = {}
    for angle in ANGLE_CLASSES:
        val_acc, _ = train_one_angle(
            args.condition, angle, args, device,
            model_module, transform, train_transform)
        if val_acc is not None:
            summary[angle] = round(val_acc * 100, 1)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.condition} restricted training (convnext_{args.model_type})")
    print(f"{'='*60}")
    for angle, acc in summary.items():
        print(f"  {angle:>4.0f}°:  val_acc = {acc:.1f}%")
    config_tag = f'{args.model_type}_40class_{cond_tag}'
    print(f"\nNext: eval_eoc.py --setting {args.condition.replace('-','_')} --config {config_tag}")


if __name__ == '__main__':
    main()
