"""
scripts/train_hierarchy.py
===========================
Trains TRUE hierarchical classifiers for EOC-Depression.

Structure:
  Level 1 (4-class):   already trained → exp_006_convnext_4class_angle15_v1
  Level 2 (subtype):   4 classifiers, one per main class, trained on images
                       ONLY within that main class
  Level 3 (vehicle):   21 leaf classifiers, one per subtype, trained ONLY on
                       images within that subtype. Single-vehicle subtypes
                       are skipped (no classification needed).

All trained on 15° X-band SOC data only — within EOC-Depression rules.

The key difference from flat classification:
  Flat 40-class: model sees all 40 vehicles competing simultaneously
  True hierarchy: leaf classifier sees only 2-4 vehicles within confirmed subtype
  → simpler problem, less confusion, sparse classes have relatively more data

Usage:
    # Smoke test — train one small classifier first
    python3 scripts/train_hierarchy.py --level leaf --subtype Medium_Car --epochs 5

    # Train full hierarchy (all levels)
    python3 scripts/train_hierarchy.py --all --epochs 10

    # Train just subtype level
    python3 scripts/train_hierarchy.py --level subtype --epochs 10

    # Train just leaf level
    python3 scripts/train_hierarchy.py --level leaf --epochs 10
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'

HIERARCHY = {
    'Bus': {
        'Large_Bus':  ['Yutong_ZK6120HY1'],
        'Medium_Bus': ['Dongfeng_EQ6608LTV', 'MAXUS_V80', 'Yangzi_YZK6590XCA'],
        'Small_Bus':  ['Buick_GL8', "Chang'an_Starlight_4500",
                       'Dongfeng_Forthing_Lingzhi', 'Wuling_Rongguang_V'],
    },
    'Car': {
        'Compact_SUV': ['Jeep_Patriot', 'Lincoln_MKC', 'Mitsubishi_Outlander_2003'],
        'Large_Car':   ['Great_Wall_Voleex_C50', 'Hongqi_CA7180A3E', 'Hongqi_h5'],
        'Medium_Car':  ['Buick_Excelle_GT', 'Chery_Arrizo 5'],
        'Medium_SUV':  ["Chang'an_CS75_Plus", 'Changfeng_Cheetah_CFA6473C',
                        'Chevrolet_Blazer_1998'],
        'Mini_Car':    ['Chery_qq3', 'Hawtai_EV160B'],
    },
    'Speacial': {
        'Ambulance':     ['JINBEI_SY5033XJH'],
        'ECV':           ['Lveco_Proud_2009'],
        'Road_Roller':   ['Changlin_8228-5'],
        'Shovel_Loader': ['SDLG_ZL40F'],
    },
    'Truck': {
        'Forklift':    ['Hyundai_HLF25_II'],
        'Heavy_DT':    ['Dongfeng_Tianjin_DFH2200B'],
        'Heavy_FT':    ['FAW_J6P', 'SHACMAN_DeLong_M3000'],
        'Heavy_ST':    ['Dongfeng_Tianjin_KR230', 'SHACMAN_DeLong_X3000'],
        'Light_DT':    ['FAW_Jiabao_T51', 'Foton_BJ1045V9JB5-54', 'WAW_Aochi_1800'],
        'Light_PV':    ['JAC_Junling'],
        'Medium_TT':   ['Dongfeng_Duolika', 'WAW_Aochi_Hongrui'],
        'Mixer_Truck': ['CNHTC_HOWO'],
        'Pickup':      ['Great_Wall_poer', 'Huanghai_N1'],
    },
}

# Single-vehicle subtypes — no classifier needed
TRIVIAL_SUBTYPES = {sub for subs in HIERARCHY.values()
                    for sub, vehs in subs.items() if len(vehs) == 1}


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
        return img, self.class_to_idx[rec['label']], rec['image_id']


def load_data(filter_field: str, filter_values: list, label_field: str,
              seed: int = 42):
    """
    Load 15° X-band SOC training images filtered to specific classes.
    filter_field: 'main_class', 'subclass', or 'target_type'
    filter_values: list of valid values for that field
    label_field: what to use as the classification label
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT id AS image_id, local_path, {label_field} AS label
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = 15
              AND band = 'X'
              AND split = 'train'
              AND {filter_field} = ANY(%s)
              AND {label_field} IS NOT NULL
              AND local_path IS NOT NULL
        """, (filter_values,))
        rows = cur.fetchall()
    conn.close()

    records     = [dict(r) for r in rows]
    class_names = sorted(set(r['label'] for r in records))

    rng = np.random.default_rng(seed)
    by_class = {}
    for rec in records:
        by_class.setdefault(rec['label'], []).append(rec)

    train_recs, val_recs = [], []
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.15))
        val_recs.extend(items[:n_val])
        train_recs.extend(items[n_val:])

    return train_recs, val_recs, class_names


def train_classifier(name: str, train_recs, val_recs, class_names,
                     args, device, model_module, transform, train_transform):
    """Train one classifier and save weights + config."""
    if len(class_names) == 1:
        print(f"  {name}: trivial (1 class: {class_names[0]}) — skipping")
        return None, class_names

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"  {name}: {len(class_names)} classes, "
          f"{len(train_recs):,} train, {len(val_recs):,} val")

    train_ds = ChipDataset(train_recs, class_to_idx, train_transform)
    val_ds   = ChipDataset(val_recs,   class_to_idx, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              persistent_workers=True)

    model     = model_module.build_model(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = MODELS_DIR / f'hier_{name}'
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

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                preds  = model(images).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(labels.tolist())
        val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
        elapsed = time.time() - start
        print(f"    Epoch {epoch}/{args.epochs}  "
              f"loss={total_loss/len(train_loader):.4f}  "
              f"val_acc={val_acc*100:.1f}%  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    config = {
        'name':         name,
        'class_names':  class_names,
        'n_classes':    len(class_names),
        'best_val_acc': round(best_val_acc * 100, 1),
        'training':     '15deg_Xband_SOC_only',
        'date':         datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f"    Best val: {best_val_acc*100:.1f}%  → hier_{name}")
    return best_val_acc, class_names


def train_subtype_level(args, device, model_module, transform, train_transform):
    """Train 4 subtype classifiers (one per main class)."""
    print(f"\n{'='*60}")
    print(f"  LEVEL 2: Subtype classifiers (4 total)")
    print(f"{'='*60}")
    results = {}
    for main_class, subs in sorted(HIERARCHY.items()):
        subtype_names  = sorted(subs.keys())
        all_subtypes   = [s for subs_dict in HIERARCHY.values()
                          for s in subs_dict.keys()]
        train_recs, val_recs, class_names = load_data(
            'main_class', [main_class], 'subclass')
        acc, _ = train_classifier(
            f'subtype_{main_class}', train_recs, val_recs, class_names,
            args, device, model_module, transform, train_transform)
        results[main_class] = acc
    return results


def train_leaf_level(args, device, model_module, transform, train_transform,
                     only_subtype=None):
    """Train leaf vehicle classifiers (one per subtype, skip trivial)."""
    print(f"\n{'='*60}")
    print(f"  LEVEL 3: Leaf vehicle classifiers")
    print(f"  Trivial (1-vehicle) subtypes skipped: {sorted(TRIVIAL_SUBTYPES)}")
    print(f"{'='*60}")
    results = {}
    for main_class, subs in sorted(HIERARCHY.items()):
        for subtype, vehicles in sorted(subs.items()):
            if only_subtype and subtype != only_subtype:
                continue
            if subtype in TRIVIAL_SUBTYPES:
                print(f"  {subtype}: trivial → {vehicles[0]}")
                continue
            train_recs, val_recs, class_names = load_data(
                'subclass', [subtype], 'target_type')
            acc, _ = train_classifier(
                f'leaf_{subtype}', train_recs, val_recs, class_names,
                args, device, model_module, transform, train_transform)
            results[subtype] = acc
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all',          action='store_true',
                        help='Train all levels (subtype + leaf)')
    parser.add_argument('--level',        choices=['subtype', 'leaf'],
                        help='Train one level only')
    parser.add_argument('--subtype',      default=None,
                        help='Train one specific leaf subtype (smoke test)')
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=0.0001)
    parser.add_argument('--num-workers',  type=int,   default=4)
    args = parser.parse_args()

    if not args.all and not args.level and not args.subtype:
        print("Error: specify --all, --level, or --subtype for smoke test")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    print(f"Training on: 15° X-band SOC only")

    model_module    = importlib.import_module('models.convnext')
    transform       = model_module.get_transform()
    train_transform = model_module.get_train_transform()

    if args.subtype:
        # Smoke test: one leaf classifier
        print(f"\nSmoke test: {args.subtype}")
        train_leaf_level(args, device, model_module, transform, train_transform,
                         only_subtype=args.subtype)
        return

    if args.all or args.level == 'subtype':
        train_subtype_level(args, device, model_module, transform, train_transform)

    if args.all or args.level == 'leaf':
        train_leaf_level(args, device, model_module, transform, train_transform)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Next: run eval_eoc_hierarchy_full.py to evaluate")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
