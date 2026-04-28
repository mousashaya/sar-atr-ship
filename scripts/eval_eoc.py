"""
scripts/eval_eoc.py
===================
End-to-end evaluation of the angle-routing pipeline on EOC data.

Pipeline:
  EOC image → angle detector → route to angle-specific classifier → vehicle class

Angle detector weights:  models/saved/angle_{name}/weights.pt
Vehicle classifier:      models/saved/exp_???_convnext_4class_angle{A}_v1/weights.pt

Usage:
    python3 scripts/eval_eoc.py
    python3 scripts/eval_eoc.py --angle-model angle_detector_v2
    python3 scripts/eval_eoc.py --setting EOC_scene
    python3 scripts/eval_eoc.py --all-settings
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

ANGLE_CLASSES  = [15.0, 30.0, 45.0, 60.0]
EOC_SETTINGS   = ['EOC_azimuth', 'EOC_band', 'EOC_depression', 'EOC_pol', 'EOC_scene']
MODELS_DIR     = Path(__file__).parent.parent / 'models' / 'saved'


# ── Model loading ─────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_angle_detector(name: str, device):
    """Load ConvNeXt angle classifier (4 outputs: 15/30/45/60°)."""
    import importlib
    model_module = importlib.import_module('models.convnext')
    model = model_module.build_model(num_classes=4).to(device)
    weights = MODELS_DIR / f'angle_{name}' / 'weights.pt'
    if not weights.exists():
        raise FileNotFoundError(f"Angle detector weights not found: {weights}")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print(f"  Loaded angle detector: {weights}")
    return model


def find_vehicle_classifier(angle: float, config: str = '4class') -> Path:
    """Auto-discover the vehicle classifier for a given angle by name pattern."""
    pattern = f'convnext_{config}_angle{int(angle)}_v1'
    matches = [d for d in MODELS_DIR.iterdir()
               if d.is_dir() and pattern in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No vehicle classifier found for angle {angle}° config={config}. "
            f"Looking for pattern '{pattern}' in {MODELS_DIR}. "
            f"Make sure all 4 angle-specific classifiers have finished training.")
    # Pick latest if multiple
    return sorted(matches)[-1]


def load_vehicle_classifiers(device, config: str = '4class') -> dict:
    """Load all 4 angle-specific vehicle classifiers. Returns {angle: model}."""
    import importlib

    classifiers = {}
    for angle in ANGLE_CLASSES:
        save_dir   = find_vehicle_classifier(angle, config)
        model_cfg  = json.loads((save_dir / 'config.json').read_text())
        n_cls      = len(model_cfg['class_names'])
        # Use model_module stored in config if present, else default to convnext
        module_name  = model_cfg.get('model_module', 'models.convnext')
        model_module = importlib.import_module(module_name)
        model        = model_module.build_model(num_classes=n_cls).to(device)
        model.load_state_dict(
            torch.load(save_dir / 'weights.pt', map_location=device))
        model.eval()
        classifiers[angle] = {
            'model':       model,
            'class_names': model_cfg['class_names'],
        }
        print(f"  Loaded {angle}° classifier: {save_dir.name}  "
              f"({n_cls} classes, {module_name})")
    return classifiers


# ── Data ──────────────────────────────────────────────────────────────────────

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
        return img, rec['image_id'], rec['true_angle'], rec['true_class']


CONFIG_FIELD = {'4class': 'main_class', '21class': 'subclass', '40class': 'target_type'}

def _class_field_for_config(config: str) -> str:
    """Extract the DB field from a config string like 'small_40class_eoc_scene'."""
    for key in CONFIG_FIELD:
        if key in config:
            return CONFIG_FIELD[key]
    raise ValueError(f"Cannot determine class level from config '{config}'. "
                     f"Expected one of {list(CONFIG_FIELD.keys())} to appear in the name.")

def load_eoc_images(setting: str, config: str = '4class') -> list[dict]:
    class_field = _class_field_for_config(config)
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT id AS image_id, local_path, depression_angle, {class_field} AS true_class
            FROM defense.atrnet_images
            WHERE experiment_setting = %s
              AND local_path IS NOT NULL
              AND depression_angle IS NOT NULL
              AND {class_field} IS NOT NULL AND {class_field} != ''
        """, (setting,))
        rows = cur.fetchall()
    conn.close()
    return [{'image_id':   r['image_id'],
             'local_path': r['local_path'],
             'true_angle': float(r['depression_angle']),
             'true_class': r['true_class']} for r in rows]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_pipeline(records, angle_detector, vehicle_classifiers,
                 transform, device, batch_size=64):
    """
    Two-pass inference:
      Pass 1: angle detector on all images → predicted angles
      Pass 2: group by predicted angle, run each angle-specific classifier
    Returns list of (true_class, predicted_class, true_angle, predicted_angle)
    """
    dataset    = EOCDataset(records, transform)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True)

    angle_idx_to_val = {i: a for i, a in enumerate(ANGLE_CLASSES)}

    # Pass 1: predict angles
    print(f"  Pass 1: angle detection on {len(records):,} images...")
    all_image_ids, all_true_angles, all_pred_angles, all_true_classes = [], [], [], []

    with torch.no_grad():
        for imgs, image_ids, true_angles, true_classes in loader:
            imgs    = imgs.to(device)
            logits  = angle_detector(imgs)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            all_image_ids.extend(image_ids.tolist())
            all_true_angles.extend(true_angles.tolist())
            all_pred_angles.extend([angle_idx_to_val[i] for i in pred_idx])
            all_true_classes.extend(true_classes)

    angle_acc = sum(p == t for p, t in zip(all_pred_angles, all_true_angles)) / len(all_true_angles)
    print(f"  Angle detection accuracy: {angle_acc*100:.1f}%")

    # Pass 2: route to angle-specific classifiers
    print(f"  Pass 2: vehicle classification by routed angle...")
    # Build index: position → predicted_class
    pred_vehicle = [''] * len(records)

    by_angle = defaultdict(list)
    for i, pred_angle in enumerate(all_pred_angles):
        by_angle[pred_angle].append(i)

    for angle, indices in sorted(by_angle.items()):
        clf_info  = vehicle_classifiers[angle]
        clf_model = clf_info['model']
        cls_names = clf_info['class_names']
        cls_to_idx = {c: i for i, c in enumerate(cls_names)}

        angle_records = [records[i] for i in indices]
        angle_dataset = EOCDataset(angle_records, transform)
        angle_loader  = DataLoader(angle_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4,
                                   persistent_workers=True)

        preds_for_angle = []
        with torch.no_grad():
            for imgs, _, _, _ in angle_loader:
                imgs   = imgs.to(device)
                logits = clf_model(imgs)
                pred_idx = logits.argmax(dim=1).cpu().tolist()
                preds_for_angle.extend([cls_names[i] for i in pred_idx])

        for local_i, global_i in enumerate(indices):
            pred_vehicle[global_i] = preds_for_angle[local_i]

        n_correct = sum(1 for i, p in zip(indices, preds_for_angle)
                        if p == all_true_classes[i])
        print(f"    {angle}° ({len(indices):,} images routed):  "
              f"{n_correct/len(indices)*100:.1f}% vehicle accuracy")

    results = list(zip(all_true_classes, pred_vehicle,
                       all_true_angles, all_pred_angles))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(results, setting: str):
    true_classes = [r[0] for r in results]
    pred_classes = [r[1] for r in results]
    true_angles  = [r[2] for r in results]
    pred_angles  = [r[3] for r in results]

    overall_acc  = sum(p == t for p, t in zip(pred_classes, true_classes)) / len(true_classes)
    routing_acc  = sum(p == t for p, t in zip(pred_angles, true_angles))   / len(true_angles)

    class_names = sorted(set(true_classes))
    cm = confusion_matrix(true_classes, pred_classes, labels=class_names)

    print(f"\n{'='*60}")
    print(f"  {setting}")
    print(f"{'='*60}")
    print(f"  Routing accuracy:  {routing_acc*100:.1f}%")
    print(f"  Vehicle accuracy:  {overall_acc*100:.1f}%")
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(class_names):
        n   = cm[i].sum()
        acc = cm[i, i] / n if n > 0 else 0
        print(f"    {cls:<12} {acc*100:.1f}%  ({cm[i,i]:,}/{n:,})")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = ''.join(f"{c:>12}" for c in class_names)
    print(f"  {'':>12}{header}")
    for i, row_label in enumerate(class_names):
        row = ''.join(f"{cm[i,j]:>12}" for j in range(len(class_names)))
        print(f"  {row_label:>12}{row}")

    return overall_acc, routing_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate angle-routing pipeline on EOC data")
    parser.add_argument('--angle-model', default='angle_detector_v1',
                        help='Angle detector name (default: angle_detector_v1)')
    parser.add_argument('--config',      default='4class',
                        help='Classification level or full config name (default: 4class)')
    parser.add_argument('--setting',     default='EOC_scene',
                        help='EOC setting to evaluate (default: EOC_scene)')
    parser.add_argument('--all-settings', action='store_true',
                        help='Evaluate on all 5 EOC settings')
    parser.add_argument('--batch-size',  type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")

    import importlib
    model_module = importlib.import_module('models.convnext')
    transform    = model_module.get_transform()

    print("\nLoading models...")
    angle_detector      = load_angle_detector(args.angle_model, device)
    vehicle_classifiers = load_vehicle_classifiers(device, args.config)

    settings = EOC_SETTINGS if args.all_settings else [args.setting]
    summary  = {}

    for setting in settings:
        print(f"\nLoading {setting} images...")
        records = load_eoc_images(setting, args.config)
        print(f"  {len(records):,} images")

        results = run_pipeline(records, angle_detector, vehicle_classifiers,
                               transform, device, args.batch_size)
        veh_acc, ang_acc = report(results, setting)
        summary[setting] = {'vehicle_acc': round(veh_acc*100, 1),
                            'routing_acc': round(ang_acc*100, 1)}

    if len(settings) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Setting':<20} {'Routing':>10} {'Vehicle':>10}")
        print(f"  {'-'*40}")
        for s, m in summary.items():
            print(f"  {s:<20} {m['routing_acc']:>9.1f}% {m['vehicle_acc']:>9.1f}%")
        avg_veh = np.mean([m['vehicle_acc'] for m in summary.values()])
        print(f"\n  Average vehicle accuracy across EOC: {avg_veh:.1f}%")
        print(f"  ATRNet-STAR reported SOTA EOC:        ~43%")


if __name__ == '__main__':
    main()
