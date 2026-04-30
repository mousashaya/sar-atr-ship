"""
scripts/eval_eoc_flat.py
========================
Evaluates a flat (no routing, no hierarchy) model on all EOC conditions.
Used to compare against the cascade pipeline (eval_eoc.py) on equal footing:
  - Same architecture (ConvNeXt)
  - Same clean acquisition-key split
  - Only difference: no angle routing, no hierarchy

Usage:
    python3 scripts/eval_eoc_flat.py --exp-id 95
    python3 scripts/eval_eoc_flat.py --exp-id 95 --all-settings
    python3 scripts/eval_eoc_flat.py --exp-id 95 --setting EOC_depression
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

EOC_SETTINGS = ['EOC_azimuth', 'EOC_band', 'EOC_depression', 'EOC_pol', 'EOC_scene']
MODELS_DIR   = Path(__file__).parent.parent / 'models' / 'saved'


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_flat_model(exp_id: int, device):
    """Load a flat classifier by experiment ID."""
    import importlib
    matches = [d for d in MODELS_DIR.iterdir()
               if d.is_dir() and d.name.startswith(f'exp_{exp_id:03d}_')]
    if not matches:
        raise FileNotFoundError(
            f"No saved model found for exp {exp_id} in {MODELS_DIR}. "
            f"Has training finished?")
    save_dir    = sorted(matches)[-1]
    config      = json.loads((save_dir / 'config.json').read_text())
    class_names = config['class_names']
    n_cls       = len(class_names)

    model_module = importlib.import_module('models.convnext')
    model        = model_module.build_model(num_classes=n_cls).to(device)
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    model.eval()
    print(f"  Loaded exp_{exp_id}: {save_dir.name}")
    print(f"  Classes ({n_cls}): {class_names[:5]}{'...' if n_cls > 5 else ''}")
    return model, class_names, model_module.get_transform()


class EOCDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
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
        return img, rec['true_class'], rec['true_angle']


def load_eoc_images(setting: str) -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, depression_angle,
                   target_type AS true_class
            FROM defense.atrnet_images
            WHERE experiment_setting = %s
              AND local_path IS NOT NULL
              AND depression_angle IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
        """, (setting,))
        rows = cur.fetchall()
    conn.close()
    return [{'image_id':   r['image_id'],
             'local_path': r['local_path'],
             'true_angle': float(r['depression_angle']),
             'true_class': r['true_class']} for r in rows]


def run_flat(records, model, class_names, transform, device, batch_size=64):
    """Single-pass flat inference — no routing."""
    dataset = EOCDataset(records, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, persistent_workers=True)

    all_true, all_pred, all_angles = [], [], []
    with torch.no_grad():
        for imgs, true_classes, true_angles in loader:
            imgs     = imgs.to(device)
            logits   = model(imgs)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            all_pred.extend([class_names[i] for i in pred_idx])
            all_true.extend(true_classes)
            all_angles.extend(true_angles.tolist())

    return all_true, all_pred, all_angles


def report(true_classes, pred_classes, true_angles, setting: str):
    overall_acc = sum(p == t for p, t in zip(pred_classes, true_classes)) / len(true_classes)

    class_names = sorted(set(true_classes))
    cm = confusion_matrix(true_classes, pred_classes, labels=class_names)

    print(f"\n{'='*60}")
    print(f"  {setting}  (flat, no routing)")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {overall_acc*100:.1f}%")

    # Per-angle breakdown
    by_angle = defaultdict(lambda: {'correct': 0, 'total': 0})
    for t, p, a in zip(true_classes, pred_classes, true_angles):
        by_angle[a]['total']   += 1
        by_angle[a]['correct'] += int(t == p)
    print(f"\n  By depression angle:")
    for angle in sorted(by_angle):
        d   = by_angle[angle]
        acc = d['correct'] / d['total'] * 100
        print(f"    {angle}°: {acc:.1f}%  ({d['total']:,} images)")

    print(f"\n  Per-class accuracy (bottom 10):")
    cls_accs = []
    for i, cls in enumerate(class_names):
        n = cm[i].sum()
        cls_accs.append((cls, cm[i, i] / n * 100 if n > 0 else 0, n))
    for cls, acc, n in sorted(cls_accs, key=lambda x: x[1])[:10]:
        print(f"    {cls:<35} {acc:.1f}%  ({n:,})")

    return overall_acc


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate flat (no routing) model on EOC conditions")
    parser.add_argument('--exp-id',      type=int, required=True,
                        help='Experiment ID of flat model (e.g. 95)')
    parser.add_argument('--setting',     default='EOC_depression',
                        help='EOC setting to evaluate')
    parser.add_argument('--all-settings', action='store_true',
                        help='Evaluate on all 5 EOC settings')
    parser.add_argument('--batch-size',  type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"\nLoading flat model (exp_{args.exp_id})...")
    model, class_names, transform = load_flat_model(args.exp_id, device)

    settings = EOC_SETTINGS if args.all_settings else [args.setting]
    summary  = {}

    for setting in settings:
        print(f"\nLoading {setting} images...")
        records = load_eoc_images(setting)
        print(f"  {len(records):,} images")
        true_cls, pred_cls, true_ang = run_flat(
            records, model, class_names, transform, device, args.batch_size)
        acc = report(true_cls, pred_cls, true_ang, setting)
        summary[setting] = round(acc * 100, 1)

    if len(settings) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — Flat model exp_{args.exp_id} vs Cascade")
        print(f"{'='*60}")

        cascade = {
            'EOC_depression': 93.5, 'EOC_azimuth': 93.5,
            'EOC_band': 92.5, 'EOC_pol': 67.6, 'EOC_scene': 26.5,
        }
        paper = {
            'EOC_depression': 43.1, 'EOC_azimuth': 46.7,
            'EOC_band': 89.2, 'EOC_pol': 84.6, 'EOC_scene': 29.7,
        }
        print(f"\n  {'Setting':<20} {'Paper':>8} {'Flat':>8} {'Cascade':>10} {'Gain':>8}")
        print(f"  {'-'*56}")
        for s in EOC_SETTINGS:
            flat_acc    = summary.get(s, 0)
            cascade_acc = cascade.get(s, 0)
            paper_acc   = paper.get(s, 0)
            gain        = cascade_acc - flat_acc
            print(f"  {s:<20} {paper_acc:>7.1f}% {flat_acc:>7.1f}% "
                  f"{cascade_acc:>9.1f}% {gain:>+7.1f}%")


if __name__ == '__main__':
    main()
