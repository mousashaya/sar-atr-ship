"""
scripts/eval_rotation.py
========================
Demonstrates ConvNeXt sensitivity to image rotation.

Loads a trained flat classifier and evaluates it on test images rotated
by fixed angles on-the-fly — no rotated images saved to disk.

Usage:
    python3 scripts/eval_rotation.py
    python3 scripts/eval_rotation.py --model rot_baseline_v1
    python3 scripts/eval_rotation.py --angles 0 15 30 45 90 135 180
    python3 scripts/eval_rotation.py --batch-size 128
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
SEED       = 42
DEFAULT_ANGLES = [0, 15, 30, 45, 60, 90, 120, 135, 150, 165, 180]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_records(band: str) -> list[dict]:
    """Load SOC test images using the same acquisition-key split as training."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, band, depression_angle, azimuth_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND band = %s
        """, (band,))
        rows = cur.fetchall()
    conn.close()

    # Reproduce acquisition-key split (seed=42, 85% pool → 15% test)
    rng = np.random.default_rng(SEED)
    by_class_acq = {}
    for r in rows:
        key = (r['target_type'], r['band'], r['depression_angle'], r['azimuth_angle'])
        by_class_acq.setdefault(r['target_type'], {}).setdefault(key, []).append(dict(r))

    test_recs = []
    for cls, acqs in by_class_acq.items():
        acq_list = list(acqs.items())
        rng.shuffle(acq_list)
        n_pool = max(1, int(len(acq_list) * 0.85))
        for _, imgs in acq_list[n_pool:]:
            test_recs.extend(imgs)

    return test_recs


class RotatedDataset(Dataset):
    """Loads images and applies a fixed rotation before the standard transform."""

    def __init__(self, records, rotation_deg: float, base_transform):
        self.records       = records
        self.rotation_deg  = rotation_deg
        self.base_transform = base_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)

        if self.rotation_deg != 0:
            img = img.rotate(self.rotation_deg, resample=Image.BILINEAR, expand=False)

        if self.base_transform:
            img = self.base_transform(img)

        return img, idx


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device):
    matches = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))
    if not matches:
        raise FileNotFoundError(
            f"No model found for '{model_name}'. "
            f"Check models/saved/ or run training first.")
    exp_dir     = matches[-1]
    config      = json.loads((exp_dir / 'config.json').read_text())
    class_names = config['class_names']
    model_type  = config.get('model_type', 'convnext_small')
    mod         = importlib.import_module(f'models.{model_type}')
    model       = mod.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded: {exp_dir.name}  ({len(class_names)} classes, val={config.get('best_val_acc','?')}%)")
    return model, class_names, mod.get_transform()


# ── Inference ─────────────────────────────────────────────────────────────────

def eval_at_rotation(model, class_names, records, rotation_deg,
                     base_transform, device, batch_size) -> float:
    ds     = RotatedDataset(records, rotation_deg, base_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    true_classes = [r['target_type'] for r in records]
    preds        = [None] * len(records)

    with torch.no_grad():
        for imgs, indices in loader:
            logits = model(imgs.to(device))
            tops   = logits.argmax(dim=1).cpu().tolist()
            for i, p in zip(indices.tolist(), tops):
                preds[i] = class_names[p]

    correct = sum(p == t for p, t in zip(preds, true_classes))
    return correct / len(true_classes)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      default='rot_baseline_v1',
                        help='Model name as saved in models/saved/ (default: rot_baseline_v1)')
    parser.add_argument('--band',       default='X',
                        help='Band filter used during training (default: X)')
    parser.add_argument('--angles',     nargs='+', type=float,
                        default=DEFAULT_ANGLES,
                        help='Rotation angles to test in degrees')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    print(f"\nLoading model...")
    model, class_names, transform = load_model(args.model, device)

    print(f"\nLoading test images (band={args.band})...")
    records = load_test_records(args.band)
    print(f"  {len(records):,} test images")

    # ── Evaluate at each rotation ─────────────────────────────────────────────
    print(f"\nEvaluating at {len(args.angles)} rotation angles...")
    results = {}
    baseline_acc = None

    for deg in sorted(args.angles):
        acc = eval_at_rotation(model, class_names, records, deg,
                               transform, device, args.batch_size)
        results[deg] = round(acc * 100, 1)
        if deg == 0:
            baseline_acc = acc * 100
        drop = f"  ({acc*100 - baseline_acc:+.1f}%)" if baseline_acc is not None and deg != 0 else ""
        print(f"  {deg:>6}°  →  {acc*100:.1f}%{drop}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"  Rotation Sensitivity — {args.model}")
    print(f"{'='*45}")
    print(f"  {'Rotation':>10}  {'Accuracy':>10}  {'Drop':>10}")
    print(f"  {'-'*35}")
    for deg, acc in sorted(results.items()):
        drop = f"{acc - results[0]:+.1f}%" if deg != 0 else "—"
        print(f"  {deg:>9}°  {acc:>9.1f}%  {drop:>10}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'model':   args.model,
        'band':    args.band,
        'n_test':  len(records),
        'results': results,
    }
    out_path = Path(__file__).parent.parent / 'logs' / 'rotation_sensitivity.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
