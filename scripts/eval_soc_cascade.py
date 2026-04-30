"""
scripts/eval_soc_cascade.py
============================
Compare flat SOC model against four cascade variants:
  - Band routing    (X vs KU     — metadata, perfect routing)
  - Pol routing     (HH/HV/VH/VV — metadata, perfect routing)
  - Depression routing (15/30/45/60° — metadata, perfect routing)
  - Azimuth routing (4 × 90° bins  — metadata, perfect routing)

All models trained on SOC_40classes with acquisition-key split (seed=42).
Test set is independently drawn per model from the same distribution.
Routing uses known metadata labels — no routing errors.

Usage:
    python3 scripts/eval_soc_cascade.py
    python3 scripts/eval_soc_cascade.py --batch-size 128
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

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
SEED       = 42

AZIMUTH_BINS = [
    (0,   90,  'soc_az_0_90_v1'),
    (90,  180, 'soc_az_90_180_v1'),
    (180, 270, 'soc_az_180_270_v1'),
    (270, 360, 'soc_az_270_360_v1'),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_exp_dir(name: str) -> Path:
    matches = sorted(MODELS_DIR.glob(f'exp_*_{name}'))
    if not matches:
        raise FileNotFoundError(f"No model found for '{name}'. Run training first.")
    return matches[-1]


def load_model(exp_name: str, device):
    exp_dir     = find_exp_dir(exp_name)
    config      = json.loads((exp_dir / 'config.json').read_text())
    class_names = config['class_names']
    model_type  = config.get('model_type', 'convnext_small')
    mod         = importlib.import_module(f"models.{model_type}")
    model       = mod.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    val_acc = config.get('best_val_acc', '?')
    print(f"  {exp_name}: {len(class_names)} classes  val={val_acc}%")
    return model, class_names, config


# ── Dataset ───────────────────────────────────────────────────────────────────

class SOCDataset(Dataset):
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


def load_test_images(band=None, polarization=None,
                     angle=None, azimuth_min=None, azimuth_max=None) -> list[dict]:
    """Load SOC test images with optional filtering by routing variable."""
    sql = """
        SELECT id, local_path, target_type, main_class,
               band, polarization, depression_angle, azimuth_angle
        FROM defense.atrnet_images
        WHERE experiment_setting = 'SOC_40classes'
          AND local_path IS NOT NULL
          AND target_type IS NOT NULL AND target_type != ''
    """
    params = []
    if band:
        sql += " AND band = %s"; params.append(band)
    if polarization:
        sql += " AND polarization = %s"; params.append(polarization)
    if angle is not None:
        sql += " AND depression_angle = %s"; params.append(angle)
    if azimuth_min is not None:
        sql += " AND azimuth_angle >= %s"; params.append(azimuth_min)
    if azimuth_max is not None:
        sql += " AND azimuth_angle < %s"; params.append(azimuth_max)

    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()

    # Reproduce acquisition-key split to get test portion (seed=42, test_frac=0.15)
    import numpy as np
    from collections import defaultdict as dd
    rng = np.random.default_rng(SEED)

    by_class_acq = dd(lambda: dd(list))
    for r in rows:
        key = (r['target_type'], r['band'], r['depression_angle'], r['azimuth_angle'])
        by_class_acq[r['target_type']][key].append(dict(r))

    test_recs = []
    pool_frac = 0.85
    for cls, acqs in by_class_acq.items():
        acq_list = list(acqs.items())
        rng.shuffle(acq_list)
        n_pool = max(1, int(len(acq_list) * pool_frac))
        for _, imgs in acq_list[n_pool:]:
            test_recs.extend(imgs)

    return test_recs


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, records, transform, device, batch_size=64) -> list[int]:
    ds     = SOCDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    preds  = [None] * len(records)
    with torch.no_grad():
        for imgs, indices in loader:
            logits = model(imgs.to(device))
            top    = logits.argmax(dim=1).cpu().tolist()
            for i, p in zip(indices.tolist(), top):
                preds[i] = p
    return preds


def eval_cascade(models_dict, all_test_records, routing_key_fn,
                 transform, device, batch_size, label):
    """
    Route each test record to a model using routing_key_fn(record) -> key.
    models_dict: {key: {'model': ..., 'classes': [...]}}
    Returns overall accuracy and per-key accuracy.
    """
    by_key = defaultdict(list)
    for i, rec in enumerate(all_test_records):
        key = routing_key_fn(rec)
        by_key[key].append(i)

    pred_classes = [None] * len(all_test_records)
    true_classes = [r['target_type'] for r in all_test_records]
    per_key_acc  = {}

    for key, indices in sorted(by_key.items()):
        if key not in models_dict:
            print(f"    WARNING: no model for routing key '{key}' — {len(indices)} images skipped")
            for i in indices:
                pred_classes[i] = '?'
            continue
        info     = models_dict[key]
        sub_recs = [all_test_records[i] for i in indices]
        sub_pred = run_inference(info['model'], sub_recs, transform, device, batch_size)
        for local_i, global_i in enumerate(indices):
            pred_classes[global_i] = info['classes'][sub_pred[local_i]]
        sub_acc = sum(pred_classes[i] == true_classes[i] for i in indices) / len(indices)
        per_key_acc[key] = round(sub_acc * 100, 1)
        print(f"    {key}: {len(indices):>6,} images → {sub_acc*100:.1f}%")

    overall = sum(p == t for p, t in zip(pred_classes, true_classes) if p is not None)
    total   = sum(1 for p in pred_classes if p is not None)
    acc     = overall / total if total > 0 else 0
    print(f"  {label}: {acc*100:.1f}%  ({total:,} images)")
    return acc, per_key_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    import models.convnext_small as csm
    transform = csm.get_transform()

    # ── Load all models ───────────────────────────────────────────────────────
    print("\nLoading models...")
    flat_model, flat_classes, flat_cfg = load_model('soc_flat_v1', device)

    band_models = {}
    for b in ['X', 'KU']:
        m, c, _ = load_model(f'soc_band_{b}_v1', device)
        band_models[b] = {'model': m, 'classes': c}

    pol_models = {}
    for p in ['HH', 'HV', 'VH', 'VV']:
        m, c, _ = load_model(f'soc_pol_{p}_v1', device)
        pol_models[p] = {'model': m, 'classes': c}

    dep_models = {}
    for a in [15.0, 30.0, 45.0, 60.0]:
        m, c, _ = load_model(f'soc_dep_{int(a)}_v1', device)
        dep_models[a] = {'model': m, 'classes': c}

    az_models = {}
    for az_min, az_max, name in AZIMUTH_BINS:
        m, c, _ = load_model(name, device)
        az_models[(az_min, az_max)] = {'model': m, 'classes': c}

    # ── Load test images ──────────────────────────────────────────────────────
    print("\nLoading SOC test images...")
    all_recs = load_test_images()
    print(f"  Total test images: {len(all_recs):,}")

    # ── Flat evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating flat model...")
    flat_preds = run_inference(flat_model, all_recs, transform, device, args.batch_size)
    true_classes = [r['target_type'] for r in all_recs]
    flat_acc = sum(flat_classes[p] == t
                   for p, t in zip(flat_preds, true_classes)) / len(true_classes)
    print(f"  Flat SOC: {flat_acc*100:.1f}%  ({len(all_recs):,} images)")

    # ── Cascade evaluations ───────────────────────────────────────────────────
    results = {'flat': round(flat_acc * 100, 1)}

    print("\n-- Band cascade (X vs KU) --")
    band_acc, _ = eval_cascade(
        band_models, all_recs,
        routing_key_fn=lambda r: r['band'],
        transform=transform, device=device,
        batch_size=args.batch_size, label='Band cascade')
    results['band'] = round(band_acc * 100, 1)

    print("\n-- Pol cascade (HH/HV/VH/VV) --")
    pol_acc, _ = eval_cascade(
        pol_models, all_recs,
        routing_key_fn=lambda r: r['polarization'],
        transform=transform, device=device,
        batch_size=args.batch_size, label='Pol cascade')
    results['pol'] = round(pol_acc * 100, 1)

    print("\n-- Depression cascade (15/30/45/60°) --")
    dep_acc, _ = eval_cascade(
        dep_models, all_recs,
        routing_key_fn=lambda r: float(r['depression_angle']),
        transform=transform, device=device,
        batch_size=args.batch_size, label='Depression cascade')
    results['depression'] = round(dep_acc * 100, 1)

    print("\n-- Azimuth cascade (4 × 90° bins) --")
    def az_key(r):
        az = float(r['azimuth_angle'])
        for lo, hi, _ in AZIMUTH_BINS:
            if lo <= az < hi:
                return (lo, hi)
        return (270, 360)  # catch 360°

    az_acc, _ = eval_cascade(
        az_models, all_recs,
        routing_key_fn=az_key,
        transform=transform, device=device,
        batch_size=args.batch_size, label='Azimuth cascade')
    results['azimuth'] = round(az_acc * 100, 1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  SOC Cascade vs Flat — Summary")
    print(f"{'='*55}")
    print(f"  {'Method':<25} {'Accuracy':>10}  {'vs Flat':>10}")
    print(f"  {'-'*47}")
    flat_val = results['flat']
    for method, acc in results.items():
        diff = f"{acc - flat_val:+.1f}%" if method != 'flat' else ''
        marker = ' ✓' if acc > flat_val else (' ✗' if acc < flat_val and method != 'flat' else '')
        print(f"  {method:<25} {acc:>9.1f}%  {diff:>10}{marker}")

    out_path = Path(__file__).parent.parent / 'logs' / 'soc_cascade.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
