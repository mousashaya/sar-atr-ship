"""
scripts/eval_eoc_cascade.py
============================
Evaluate the EOC-Depression cascade vs flat comparison.

Loads:
  eoc_dep_L1_v1       — 4-way main_class classifier (ConvNeXt-Small, trained on 15°)
  eoc_dep_L2_Bus_v1   — Bus target_type classifier
  eoc_dep_L2_Car_v1   — Car target_type classifier
  eoc_dep_L2_Speacial_v1
  eoc_dep_L2_Truck_v1
  eoc_dep_flat_v1     — flat 40-class classifier (for comparison via its config.json)

Test set: EOC_depression split='test' (30°, 45°, 60°, never seen during training)

Reports:
  Flat test accuracy   — from train.py's saved config.json (already computed)
  Cascade predicted    — L1 routes by predicted main_class → L2 vehicle type
  Cascade oracle       — routes by true main_class (upper bound, no routing errors)

Usage:
    python3 scripts/eval_eoc_cascade.py
    python3 scripts/eval_eoc_cascade.py --verbose
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from collections import defaultdict

import psycopg2
import psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
MAIN_CLASSES = ['Bus', 'Car', 'Speacial', 'Truck']


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_exp_dir(name: str) -> Path:
    matches = sorted(MODELS_DIR.glob(f'exp_*_{name}'))
    if not matches:
        raise FileNotFoundError(
            f"No saved model found for name '{name}'. "
            f"Run scripts/run_eoc_depression.sh first.")
    return matches[-1]


def load_model(exp_name: str, device):
    exp_dir     = find_exp_dir(exp_name)
    config      = json.loads((exp_dir / 'config.json').read_text())
    class_names = config['class_names']
    model_type  = config.get('model_type', 'convnext_small')
    module_name = f"models.{model_type}"
    mod         = importlib.import_module(module_name)
    model       = mod.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded {exp_name}: {len(class_names)} classes  ({exp_dir.name})")
    return model, class_names, config


# ── Dataset ───────────────────────────────────────────────────────────────────

class EOCDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        if self.transform:
            img = self.transform(img)
        return img, idx


def load_test_images() -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, main_class, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND split = 'test'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND main_class IS NOT NULL AND main_class != ''
        """)
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, records, transform, device, batch_size=64) -> list[int]:
    """Return list of predicted class indices (same order as records)."""
    ds     = EOCDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    preds  = [None] * len(records)
    with torch.no_grad():
        for imgs, indices in loader:
            logits = model(imgs.to(device))
            top    = logits.argmax(dim=1).cpu().tolist()
            for i, pred in zip(indices.tolist(), top):
                preds[i] = pred
    return preds


# ── Reporting ─────────────────────────────────────────────────────────────────

def accuracy(pred_classes, true_classes):
    return sum(p == t for p, t in zip(pred_classes, true_classes)) / len(true_classes)


def per_angle_accuracy(pred_classes, true_classes, angles):
    by_angle = defaultdict(lambda: {'correct': 0, 'total': 0})
    for p, t, a in zip(pred_classes, true_classes, angles):
        by_angle[a]['total'] += 1
        if p == t:
            by_angle[a]['correct'] += 1
    return {a: v['correct'] / v['total'] for a, v in sorted(by_angle.items())}


def report_results(label, pred_classes, true_classes, angles):
    acc = accuracy(pred_classes, true_classes)
    print(f"\n  {label}: {acc*100:.1f}%")
    for a, a_acc in per_angle_accuracy(pred_classes, true_classes, angles).items():
        print(f"    {a:>4.0f}°:  {a_acc*100:.1f}%")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--verbose',    action='store_true')
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    l1_model, l1_classes, _    = load_model('eoc_dep_L1_v1', device)
    l2_models = {}
    for mc in MAIN_CLASSES:
        model, classes, _ = load_model(f'eoc_dep_L2_{mc}_v1', device)
        l2_models[mc] = {'model': model, 'classes': classes}

    # Read flat model test accuracy from its saved config
    try:
        flat_dir    = find_exp_dir('eoc_dep_flat_v1')
        flat_config = json.loads((flat_dir / 'config.json').read_text())
        flat_val_acc = flat_config.get('best_val_acc', '?')
        print(f"\n  Flat model best val acc (from config): {flat_val_acc}%")
        print(f"  (Flat test accuracy is in the training log — see logs/eoc_depression.log)")
    except FileNotFoundError:
        print("\n  Flat model not found — run training first or check log for test accuracy")
        flat_val_acc = None

    # ── Load test images ──────────────────────────────────────────────────────
    print("\nLoading EOC_depression test images (30°/45°/60°)...")
    records = load_test_images()
    print(f"  Total: {len(records):,}")
    by_angle = defaultdict(int)
    for r in records:
        by_angle[r['depression_angle']] += 1
    for a in sorted(by_angle):
        print(f"    {a:>4.0f}°:  {by_angle[a]:,}")

    true_classes = [r['target_type'] for r in records]
    true_main    = [r['main_class']   for r in records]
    angles       = [r['depression_angle'] for r in records]

    # Get transform (same for all small models)
    import models.convnext_small as csm
    transform = csm.get_transform()

    # ── L1 inference ──────────────────────────────────────────────────────────
    print(f"\nPass 1: L1 main_class prediction on {len(records):,} images...")
    l1_pred_idx = run_inference(l1_model, records, transform, device, args.batch_size)
    pred_main   = [l1_classes[i] for i in l1_pred_idx]

    l1_acc = accuracy(pred_main, true_main)
    print(f"  L1 main_class accuracy: {l1_acc*100:.1f}%")

    # ── L2 inference: predicted routing ──────────────────────────────────────
    print("\nPass 2a: L2 vehicle type via PREDICTED routing...")
    pred_vehicle_predicted = [None] * len(records)
    by_main = defaultdict(list)
    for i, mc in enumerate(pred_main):
        by_main[mc].append(i)

    for mc, indices in sorted(by_main.items()):
        if mc not in l2_models:
            print(f"  WARNING: no L2 model for '{mc}' — {len(indices)} images unrouted")
            for i in indices:
                pred_vehicle_predicted[i] = '?'
            continue
        l2_info  = l2_models[mc]
        sub_recs = [records[i] for i in indices]
        sub_pred = run_inference(l2_info['model'], sub_recs, transform, device, args.batch_size)
        for local_i, global_i in enumerate(indices):
            pred_vehicle_predicted[global_i] = l2_info['classes'][sub_pred[local_i]]
        sub_acc = accuracy(
            [pred_vehicle_predicted[i] for i in indices],
            [true_classes[i] for i in indices])
        print(f"  {mc:>10}: {len(indices):>6,} routed  →  {sub_acc*100:.1f}% vehicle acc")

    # ── L2 inference: oracle routing ─────────────────────────────────────────
    print("\nPass 2b: L2 vehicle type via ORACLE routing (true main class)...")
    pred_vehicle_oracle = [None] * len(records)
    by_true_main = defaultdict(list)
    for i, mc in enumerate(true_main):
        by_true_main[mc].append(i)

    for mc, indices in sorted(by_true_main.items()):
        l2_info  = l2_models[mc]
        sub_recs = [records[i] for i in indices]
        sub_pred = run_inference(l2_info['model'], sub_recs, transform, device, args.batch_size)
        for local_i, global_i in enumerate(indices):
            pred_vehicle_oracle[global_i] = l2_info['classes'][sub_pred[local_i]]
        sub_acc = accuracy(
            [pred_vehicle_oracle[i] for i in indices],
            [true_classes[i] for i in indices])
        print(f"  {mc:>10}: {len(indices):>6,} images  →  {sub_acc*100:.1f}% vehicle acc")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EOC-Depression Results (trained on 15° only, tested on 30/45/60°)")
    print(f"{'='*60}")
    if flat_val_acc is not None:
        print(f"\n  Flat (val acc {flat_val_acc}%): see logs/eoc_depression.log for TEST acc")

    cascade_predicted_acc = report_results(
        'Cascade (predicted routing)', pred_vehicle_predicted, true_classes, angles)
    cascade_oracle_acc = report_results(
        'Cascade (oracle routing)',    pred_vehicle_oracle,    true_classes, angles)

    print(f"\n  ATRNet-STAR paper baseline (ConvNeXt): ~37%")
    print(f"  Cascade predicted routing:              {cascade_predicted_acc*100:.1f}%")
    print(f"  Cascade oracle routing (ceiling):       {cascade_oracle_acc*100:.1f}%")
    print(f"  (Compare flat test acc from log)")

    # Save results
    out = {
        'l1_main_class_acc':         round(l1_acc * 100, 1),
        'cascade_predicted_acc':     round(cascade_predicted_acc * 100, 1),
        'cascade_oracle_acc':        round(cascade_oracle_acc * 100, 1),
        'paper_baseline':            37.2,
        'n_test_images':             len(records),
    }
    out_path = Path(__file__).parent.parent / 'logs' / 'eoc_depression_cascade.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
