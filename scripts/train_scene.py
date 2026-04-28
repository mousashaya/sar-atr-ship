"""
scripts/train_scene.py
======================
Trains ConvNeXt to predict scene type from SAR chip.

Diagnostic question: is scene information visible in a tight 128x128 crop?
If yes, scene routing could be added as a second routing stage after angle.

5 scene classes: bare_soil, factory, sandstone, urban, woodland

Usage:
    python3 scripts/train_scene.py --name "scene_detector_v1"
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


SCENE_CLASSES = ['bare_soil', 'factory', 'sandstone', 'urban', 'woodland']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class SceneDataset(Dataset):
    def __init__(self, records, transform=None, mask_target=False):
        self.records      = records
        self.transform    = transform
        self.mask_target  = mask_target
        self.scene_to_idx = {s: i for i, s in enumerate(SCENE_CLASSES)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), color=(64, 64, 64))

        if self.mask_target and rec.get('bbox'):
            # Zero out the vehicle bbox — leave only background scene pixels
            import numpy as np
            arr = np.array(img)
            xmin, ymin, xmax, ymax = [int(v) for v in rec['bbox']]
            arr[ymin:ymax, xmin:xmax] = 0
            img = Image.fromarray(arr)

        if self.transform:
            img = self.transform(img)
        label = self.scene_to_idx[rec['scene']]
        return img, label, rec['image_id']


def load_data(setting='SOC_40classes'):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, scene, split,
                   bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
            FROM defense.atrnet_images
            WHERE experiment_setting = %s
              AND scene IS NOT NULL
              AND local_path IS NOT NULL
              AND split IS NOT NULL
              AND scene = ANY(%s)
        """, (setting, SCENE_CLASSES))
        rows = cur.fetchall()
    conn.close()

    def to_rec(r):
        d = dict(r)
        d['bbox'] = (r['bbox_xmin'], r['bbox_ymin'], r['bbox_xmax'], r['bbox_ymax']) \
                    if r['bbox_xmin'] is not None else None
        return d

    train_recs = [to_rec(r) for r in rows if r['split'] == 'train']
    test_recs  = [to_rec(r) for r in rows if r['split'] == 'test']

    # Carve val from train (15%), stratified by scene
    rng = np.random.default_rng(42)
    by_scene = {}
    for rec in train_recs:
        s = rec['scene']
        by_scene.setdefault(s, []).append(rec)

    val_recs   = []
    train_only = []
    for scene, items in by_scene.items():
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
    parser.add_argument('--name',         required=True)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=0.0001)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--mask-target',  action='store_true',
                        help='Zero out vehicle bbox before training — scene background only')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_module    = importlib.import_module('models.convnext')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    print("Loading data...")
    train_recs, val_recs, test_recs = load_data()
    print(f"Train: {len(train_recs):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}")
    from collections import Counter
    counts = Counter(r['scene'] for r in train_recs)
    for s in SCENE_CLASSES:
        print(f"  {s:<15} {counts.get(s, 0):,} train images")

    if args.mask_target:
        log("Target masking: ON — vehicle bbox zeroed out, scene background only")

    train_ds = SceneDataset(train_recs, train_transform, mask_target=args.mask_target)
    val_ds   = SceneDataset(val_recs,   transform,       mask_target=args.mask_target)
    test_ds  = SceneDataset(test_recs,  transform,       mask_target=args.mask_target)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    model     = model_module.build_model(num_classes=len(SCENE_CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(__file__).parent.parent / 'models' / 'saved' / f'scene_{args.name}'
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(__file__).parent.parent / 'logs' / f'train_{args.name}.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Scene detector — {args.name}")
    log(f"Classes: {SCENE_CLASSES}")
    log(f"Random baseline: {100/len(SCENE_CLASSES):.1f}%")

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
    log(f"Random baseline:   {100/len(SCENE_CLASSES):.1f}%")

    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    test_acc, test_preds, test_true = evaluate(model, test_loader, device)
    cm = confusion_matrix(test_true, test_preds, labels=list(range(len(SCENE_CLASSES))))

    log(f"\nTEST accuracy: {test_acc*100:.1f}%")
    log(f"Confusion matrix (rows=true, cols=predicted):")
    header = ''.join(f"{s:>12}" for s in SCENE_CLASSES)
    log(f"  {'':>12}{header}")
    for i, label in enumerate(SCENE_CLASSES):
        row = ''.join(f"{cm[i,j]:>12}" for j in range(len(SCENE_CLASSES)))
        log(f"  {label:>12}{row}")

    log(f"\nPer-class accuracy:")
    for i, s in enumerate(SCENE_CLASSES):
        n   = cm[i].sum()
        acc = cm[i, i] / n if n > 0 else 0
        log(f"  {s:<15} {acc*100:.1f}%  ({cm[i,i]:,}/{n:,})")

    config = {
        'name': args.name, 'task': 'scene_detection',
        'classes': SCENE_CLASSES,
        'best_val_acc': round(best_val_acc * 100, 1),
        'test_acc': round(test_acc * 100, 1),
        'random_baseline': round(100/len(SCENE_CLASSES), 1),
        'date': datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"\nDone. Weights saved to {save_dir}")
    log_file.close()


if __name__ == '__main__':
    main()
