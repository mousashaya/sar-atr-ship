"""
scripts/eval_eoc_oracle.py
===========================
Oracle evaluation — uses TRUE depression angle from metadata instead of
the learned angle detector. Establishes upper bound of angle routing.

For each EOC condition:
  - Scene/Azimuth/Band/Pol: uses the same restricted models as eval_eoc.py
    but routes with perfect angle knowledge instead of the detector
  - Depression: uses unrestricted angle-specific models (trained on full SOC)
    to show theoretical upper bound of routing on this hardest condition

This is NOT a benchmark claim — it answers the scientific question:
"How much does perfect angle routing help across all 5 EOC conditions?"

Usage:
    python3 scripts/eval_eoc_oracle.py
    python3 scripts/eval_eoc_oracle.py --setting EOC_depression
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import psycopg2
import psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import importlib

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

ANGLE_CLASSES = [15.0, 30.0, 45.0, 60.0]
MODELS_DIR    = Path(__file__).parent.parent / 'models' / 'saved'

# Config pattern for each condition's restricted models
# Depression uses unrestricted SOC models (exp_0XX_convnext_40class_angleXX_v1)
CONDITION_CONFIG = {
    'EOC_scene':       'small_40class_eoc_scene',
    'EOC_azimuth':     'small_40class_eoc_azimuth',
    'EOC_band':        'small_40class_eoc_band',
    'EOC_pol':         'small_40class_eoc_pol',
    'EOC_depression':  '40class',   # unrestricted SOC models
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def find_classifier(angle: float, config: str) -> Path:
    pattern = f'convnext_{config}_angle{int(angle)}_v1'
    matches = [d for d in MODELS_DIR.iterdir()
               if d.is_dir() and pattern in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No classifier found for angle {angle}° config={config}. "
            f"Pattern: '{pattern}'")
    return sorted(matches)[-1]


def load_classifiers(config: str, device) -> dict:
    classifiers = {}
    for angle in ANGLE_CLASSES:
        save_dir   = find_classifier(angle, config)
        model_cfg  = json.loads((save_dir / 'config.json').read_text())
        n_cls      = len(model_cfg['class_names'])
        module_name = model_cfg.get('model_module', 'models.convnext')
        model_module = importlib.import_module(module_name)
        model = model_module.build_model(num_classes=n_cls).to(device)
        model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
        model.eval()
        classifiers[angle] = {'model': model, 'class_names': model_cfg['class_names']}
        print(f"  Loaded {angle}° classifier: {save_dir.name} ({n_cls} classes)")
    return classifiers


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


def load_test_images(setting: str) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, target_type AS true_class,
                   depression_angle AS true_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = %s
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL
              AND depression_angle IS NOT NULL
        """, (setting,))
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_oracle(records, classifiers, transform, device, batch_size=64):
    """Route using TRUE angle — no detector, perfect routing."""

    # Group by true angle
    by_angle = defaultdict(list)
    for i, rec in enumerate(records):
        angle = float(rec['true_angle'])
        # Snap to nearest known angle
        nearest = min(ANGLE_CLASSES, key=lambda a: abs(a - angle))
        by_angle[nearest].append((i, rec))

    pred_classes = [''] * len(records)
    true_classes = [r['true_class'] for r in records]
    true_angles  = [float(r['true_angle']) for r in records]
    by_angle_results = {}

    for angle in sorted(by_angle.keys()):
        items   = by_angle[angle]
        indices = [i for i, _ in items]
        recs    = [r for _, r in items]

        if angle not in classifiers:
            print(f"  {angle}°: no classifier available, skipping")
            continue

        clf       = classifiers[angle]
        cls_names = clf['class_names']
        valid_classes = set(cls_names)

        # Filter to classes the classifier knows
        valid_items   = [(i, r) for i, r in zip(indices, recs)
                         if r['true_class'] in valid_classes]
        if not valid_items:
            continue

        valid_indices = [i for i, _ in valid_items]
        valid_recs    = [r for _, r in valid_items]

        ds     = EOCDataset(valid_recs, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True)

        preds = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs   = imgs.to(device)
                logits = clf['model'](imgs)
                preds.extend([cls_names[i] for i in logits.argmax(dim=1).cpu().tolist()])

        for local_i, global_i in enumerate(valid_indices):
            pred_classes[global_i] = preds[local_i]

        correct = sum(1 for i, p in zip(valid_indices, preds)
                      if p == true_classes[i])
        total   = len(valid_indices)
        acc     = correct / total * 100
        by_angle_results[angle] = (correct, total, acc)
        print(f"    {angle}°  ({total:,} images):  {acc:.1f}%")

    # Overall (only images that got a prediction)
    evaluated = [(t, p) for t, p in zip(true_classes, pred_classes) if p]
    overall   = sum(t == p for t, p in evaluated) / len(evaluated) * 100

    return overall, by_angle_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default=None,
                        help='Single EOC setting to evaluate (default: all)')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")

    settings = ([args.setting] if args.setting
                else list(CONDITION_CONFIG.keys()))

    summary = {}

    for setting in settings:
        config = CONDITION_CONFIG[setting]
        print(f"\n{'='*60}")
        print(f"  ORACLE EVAL: {setting}  (config={config})")
        print(f"{'='*60}")

        print("\nLoading classifiers...")
        classifiers = load_classifiers(config, device)

        module_name  = 'models.convnext_small' if 'small' in config else 'models.convnext'
        model_module = importlib.import_module(module_name)
        transform    = model_module.get_transform()

        print(f"\nLoading {setting} test images...")
        records = load_test_images(setting)
        print(f"  {len(records):,} images")

        print(f"\nOracle routing (true angles)...")
        overall, by_angle = run_oracle(records, classifiers, transform,
                                       device, args.batch_size)

        print(f"\n  Overall oracle accuracy: {overall:.1f}%")
        summary[setting] = {'oracle_acc': round(overall, 1),
                             'by_angle':  {str(a): round(v[2], 1)
                                           for a, v in by_angle.items()}}

    # Print comparison table
    ACTUAL = {
        'EOC_scene':      26.1,
        'EOC_azimuth':    43.7,
        'EOC_band':       79.9,
        'EOC_pol':        66.4,
        'EOC_depression': 43.6,
    }
    PAPER = {
        'EOC_scene':      16.5,
        'EOC_azimuth':    21.1,
        'EOC_band':       88.4,
        'EOC_pol':        84.6,
        'EOC_depression': 43.1,
    }

    print(f"\n{'='*70}")
    print(f"  ORACLE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<20} {'Paper':>8} {'Ours':>8} {'Oracle':>8} {'Oracle gain':>12}")
    print(f"  {'-'*60}")
    for setting, m in summary.items():
        oracle  = m['oracle_acc']
        actual  = ACTUAL.get(setting, 0)
        paper   = PAPER.get(setting, 0)
        gain    = oracle - actual
        print(f"  {setting:<20} {paper:>7.1f}% {actual:>7.1f}% {oracle:>7.1f}% {gain:>+11.1f}%")

    # Save results
    out_path = Path(__file__).parent.parent / 'logs' / 'oracle_results.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
