"""
scripts/eval_eoc_depression.py
================================
End-to-end evaluation of the EOC-Depression challenge using azimuth routing.

The EOC-Depression problem:
  Training data is restricted to 15° depression angle only.
  Standard angle routing fails — no 30°/45°/60° classifiers can be trained.

Our novel approach:
  Use AZIMUTH (horizontal viewing direction) as the routing variable instead.
  Azimuth is visible at any depression angle, so a classifier trained at 15°
  can still route and classify chips captured at 30°/45°/60°.

Pipeline:
  EOC chip (at 30°/45°/60°) → azimuth detector → route to azimuth-specific
  vehicle classifier (trained at 15°) → predicted vehicle class

Requirements (must train first):
  1. python3 scripts/train_azimuth_detector.py --name azimuth_detector_v1 --bins 4
  2. python3 scripts/train_azimuth_vehicle.py --name azimuth_v1 --bins 4 --all-bins

Usage:
    python3 scripts/eval_eoc_depression.py
    python3 scripts/eval_eoc_depression.py --azimuth-model azimuth_detector_v1 --bins 4
    python3 scripts/eval_eoc_depression.py --verbose
"""

import argparse
import importlib
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
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR    = Path(__file__).parent.parent / 'models' / 'saved'
EOC_ANGLES    = [30.0, 45.0, 60.0]   # test angles never seen during training


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Azimuth helpers ───────────────────────────────────────────────────────────

def azimuth_to_bin(azimuth: float, n_bins: int) -> int:
    bin_size = 360.0 / n_bins
    return int(azimuth / bin_size) % n_bins


def bin_label(bin_idx: int, n_bins: int) -> str:
    bin_size = 360.0 / n_bins
    lo = int(bin_idx * bin_size)
    hi = int((bin_idx + 1) * bin_size)
    return f"{lo}°-{hi}°"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_azimuth_detector(name: str, n_bins: int, device):
    """Load the azimuth bin classifier."""
    model_module = importlib.import_module('models.convnext')
    model = model_module.build_model(num_classes=n_bins).to(device)
    weights = MODELS_DIR / f'azimuth_{name}' / 'weights.pt'
    if not weights.exists():
        raise FileNotFoundError(
            f"Azimuth detector not found: {weights}\n"
            f"Run: python3 scripts/train_azimuth_detector.py --name {name} --bins {n_bins}")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print(f"  Loaded azimuth detector: {weights}")
    return model


def load_azimuth_vehicle_classifiers(name: str, n_bins: int, device) -> dict:
    """
    Load all azimuth-bin-specific vehicle classifiers.
    Returns {bin_idx: {'model': model, 'class_names': [...]}}
    """
    model_module = importlib.import_module('models.convnext')
    classifiers  = {}

    for b in range(n_bins):
        save_dir = MODELS_DIR / f'azimuth_vehicle_{name}_bin{b}of{n_bins}'
        if not save_dir.exists():
            raise FileNotFoundError(
                f"Vehicle classifier for bin {b} not found: {save_dir}\n"
                f"Run: python3 scripts/train_azimuth_vehicle.py "
                f"--name {name} --bins {n_bins} --bin {b}")
        model_cfg   = json.loads((save_dir / 'config.json').read_text())
        class_names = model_cfg['class_names']
        n_cls       = len(class_names)
        model       = model_module.build_model(num_classes=n_cls).to(device)
        model.load_state_dict(
            torch.load(save_dir / 'weights.pt', map_location=device))
        model.eval()
        lbl = bin_label(b, n_bins)
        classifiers[b] = {'model': model, 'class_names': class_names}
        print(f"  Loaded bin {b} ({lbl}) vehicle classifier: {n_cls} classes")

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


def load_eoc_depression_images() -> list[dict]:
    """
    Load EOC_depression test images at 30°/45°/60°.
    These are the chips the model has NEVER seen (trained only on 15°).
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, depression_angle,
                   target_type, azimuth_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND local_path IS NOT NULL
              AND depression_angle IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
        """)
        rows = cur.fetchall()
    conn.close()
    return [{'image_id':    r['image_id'],
             'local_path':  r['local_path'],
             'true_angle':  float(r['depression_angle']),
             'true_class':  r['target_type'],
             'azimuth_angle': r['azimuth_angle']} for r in rows]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(records, azimuth_detector, vehicle_classifiers,
                 n_bins, transform, device, batch_size=64, verbose=False):
    """
    Two-pass inference:
      Pass 1: azimuth detector → predicted azimuth bin
      Pass 2: group by bin → azimuth-specific classifier → vehicle class

    Returns list of (true_class, pred_class, true_angle, pred_bin, true_bin)
    """
    dataset = EOCDataset(records, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, persistent_workers=True)

    # Pass 1: predict azimuth bins
    print(f"  Pass 1: azimuth detection on {len(records):,} images...")
    all_image_ids, all_true_angles, all_true_classes = [], [], []
    all_pred_bins, all_true_bins = [], []

    with torch.no_grad():
        for imgs, image_ids, true_angles, true_classes in loader:
            imgs      = imgs.to(device)
            logits    = azimuth_detector(imgs)
            pred_bins = logits.argmax(dim=1).cpu().tolist()
            all_image_ids.extend(image_ids.tolist())
            all_true_angles.extend(true_angles.tolist())
            all_true_classes.extend(true_classes)
            all_pred_bins.extend(pred_bins)

    # Compute azimuth bin accuracy against known true azimuth (where available)
    true_bin_known = 0
    true_bin_total = 0
    for i, rec in enumerate(records):
        if rec['azimuth_angle'] is not None:
            true_b = azimuth_to_bin(rec['azimuth_angle'], n_bins)
            all_true_bins.append(true_b)
            if all_pred_bins[i] == true_b:
                true_bin_known += 1
            true_bin_total += 1
        else:
            all_true_bins.append(-1)

    if true_bin_total > 0:
        az_acc = true_bin_known / true_bin_total
        print(f"  Azimuth bin accuracy (where known): {az_acc*100:.1f}%  "
              f"({true_bin_known}/{true_bin_total})")

    bin_counts = defaultdict(int)
    for b in all_pred_bins:
        bin_counts[b] += 1
    print(f"  Routing distribution: " +
          ", ".join(f"bin{b}({bin_label(b,n_bins)})={bin_counts[b]:,}"
                    for b in range(n_bins)))

    # Pass 2: vehicle classification per bin
    print(f"  Pass 2: vehicle classification by azimuth bin...")
    pred_vehicle = [''] * len(records)

    by_bin = defaultdict(list)
    for i, pred_bin in enumerate(all_pred_bins):
        by_bin[pred_bin].append(i)

    for b, indices in sorted(by_bin.items()):
        clf_info   = vehicle_classifiers[b]
        clf_model  = clf_info['model']
        class_names = clf_info['class_names']

        bin_records = [records[i] for i in indices]
        bin_dataset = EOCDataset(bin_records, transform)
        bin_loader  = DataLoader(bin_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4,
                                 persistent_workers=True)

        preds_for_bin = []
        with torch.no_grad():
            for imgs, _, _, _ in bin_loader:
                imgs    = imgs.to(device)
                logits  = clf_model(imgs)
                pred_idx = logits.argmax(dim=1).cpu().tolist()
                preds_for_bin.extend([class_names[i] for i in pred_idx])

        for local_i, global_i in enumerate(indices):
            pred_vehicle[global_i] = preds_for_bin[local_i]

        n_correct = sum(1 for i, p in zip(indices, preds_for_bin)
                        if p == all_true_classes[i])
        lbl = bin_label(b, n_bins)
        print(f"    Bin {b} ({lbl}): {len(indices):,} routed  →  "
              f"{n_correct/len(indices)*100:.1f}% vehicle accuracy")

    results = list(zip(all_true_classes, pred_vehicle,
                       all_true_angles, all_pred_bins, all_true_bins))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(results, n_bins):
    true_classes = [r[0] for r in results]
    pred_classes = [r[1] for r in results]
    true_angles  = [r[2] for r in results]

    overall_acc = sum(p == t for p, t in zip(pred_classes, true_classes)) / len(true_classes)

    class_names = sorted(set(true_classes))
    cm = confusion_matrix(true_classes, pred_classes, labels=class_names)

    print(f"\n{'='*60}")
    print(f"  EOC-DEPRESSION RESULTS  (trained on 15° only)")
    print(f"{'='*60}")
    print(f"  Overall vehicle accuracy:  {overall_acc*100:.1f}%")
    print(f"  ATRNet-STAR SOTA (Table 5): 43.1%")
    if overall_acc * 100 > 43.1:
        print(f"  *** BEATS SOTA by {overall_acc*100 - 43.1:.1f} percentage points ***")

    # Per-depression-angle breakdown
    print(f"\n  Per depression angle:")
    for angle in EOC_ANGLES:
        idx = [i for i, t in enumerate(true_angles) if t == angle]
        if not idx:
            continue
        tc = [true_classes[i] for i in idx]
        pc = [pred_classes[i] for i in idx]
        acc = sum(p == t for p, t in zip(pc, tc)) / len(tc)
        print(f"    {angle:>4.0f}°:  {acc*100:.1f}%  ({len(idx):,} images)")

    # Per-class accuracy
    print(f"\n  Per-class accuracy (40 classes):")
    print(f"  {'Class':<35} {'Acc':>8}  {'Correct':>8}  {'Total':>8}")
    print(f"  {'-'*62}")
    for i, cls in enumerate(class_names):
        n   = cm[i].sum()
        acc = cm[i, i] / n if n > 0 else 0
        print(f"  {cls:<35} {acc*100:>7.1f}%  {cm[i,i]:>8,}  {n:>8,}")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    col_w = max(len(c) for c in class_names) + 2
    header = ''.join(f"{c:>{col_w}}" for c in class_names)
    print(f"  {'':>{col_w}}{header}")
    for i, row_label in enumerate(class_names):
        row = ''.join(f"{cm[i,j]:>{col_w}}" for j in range(len(class_names)))
        print(f"  {row_label:>{col_w}}{row}")

    return overall_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EOC-Depression challenge via azimuth routing")
    parser.add_argument('--azimuth-model', default='azimuth_detector_v1',
                        help='Azimuth detector name (default: azimuth_detector_v1)')
    parser.add_argument('--vehicle-name',  default=None,
                        help='Azimuth vehicle classifier base name '
                             '(default: same as --azimuth-model without detector suffix)')
    parser.add_argument('--bins',          type=int, default=4,
                        help='Number of azimuth bins (default: 4)')
    parser.add_argument('--batch-size',    type=int, default=64)
    parser.add_argument('--verbose',       action='store_true')
    args = parser.parse_args()

    # vehicle name defaults to stripping "_detector" from azimuth model name
    if args.vehicle_name is None:
        args.vehicle_name = args.azimuth_model.replace('_detector', '')

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Azimuth bins: {args.bins}  ({360//args.bins}° each)")

    model_module = importlib.import_module('models.convnext')
    transform    = model_module.get_transform()

    print("\nLoading models...")
    azimuth_detector     = load_azimuth_detector(
        args.azimuth_model, args.bins, device)
    vehicle_classifiers  = load_azimuth_vehicle_classifiers(
        args.vehicle_name, args.bins, device)

    print("\nLoading EOC_depression images (30°/45°/60°)...")
    records = load_eoc_depression_images()
    print(f"  Total: {len(records):,} images")
    from collections import Counter
    angle_counts = Counter(r['true_angle'] for r in records)
    for a in sorted(angle_counts):
        print(f"    {a:>4.0f}°:  {angle_counts[a]:,} images")

    print()
    results = run_pipeline(records, azimuth_detector, vehicle_classifiers,
                           args.bins, transform, device, args.batch_size,
                           args.verbose)

    overall_acc = report(results, args.bins)

    # Save results
    out_path = (Path(__file__).parent.parent / 'logs' /
                f'eoc_depression_{args.azimuth_model}.json')
    output = {
        'azimuth_model':   args.azimuth_model,
        'vehicle_name':    args.vehicle_name,
        'n_bins':          args.bins,
        'overall_acc':     round(overall_acc * 100, 1),
        'sota_comparison': round(overall_acc * 100 - 43.1, 1),
        'n_images':        len(records),
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
