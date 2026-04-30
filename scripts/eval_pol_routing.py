"""
scripts/eval_pol_routing.py
===========================
Evaluates pol-routing (L1 only) vs angle-routing on EOC-Pol.

For each EOC-Pol image:
  - Pol-routing:   route to the pol-specific L1 model (HH/HV/VH/VV known from metadata)
  - Angle-routing: route to the angle-specific L1 model (15/30/45/60 from metadata)

Compares L1 4-class accuracy only. If pol-routing wins, building full pol cascade
for L2+L3 is warranted.

Usage:
    python3 scripts/eval_pol_routing.py
"""

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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def find_model(name_contains: str):
    matches = [d for d in MODELS_DIR.iterdir()
               if d.is_dir() and name_contains in d.name]
    if not matches:
        return None
    return sorted(matches)[-1]


def load_model(save_dir: Path, device):
    import importlib
    config      = json.loads((save_dir / 'config.json').read_text())
    class_names = config['class_names']
    model_type  = config.get('model_type', 'convnext')
    model_module = importlib.import_module(f'models.{model_type}')
    model = model_module.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    model.eval()
    return model, class_names, model_module.get_transform()


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
            img = Image.new('RGB', (128, 128), color=(64, 64, 64))
        return self.transform(img), rec['true_class'], rec['polarization'], rec['angle']


def load_eoc_pol_images() -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, main_class AS true_class,
                   polarization, depression_angle AS angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_pol'
              AND local_path IS NOT NULL
              AND main_class IS NOT NULL AND main_class != ''
              AND polarization IS NOT NULL
              AND depression_angle IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()
    return [{'local_path':   r['local_path'],
             'true_class':   r['true_class'],
             'polarization': r['polarization'],
             'angle':        float(r['angle'])} for r in rows]


def run_routing(records, model_map, key_fn, transform, device, batch_size=64):
    """Route each image using key_fn(rec) → model key, run inference."""
    by_key = defaultdict(list)
    for i, rec in enumerate(records):
        by_key[key_fn(rec)].append((i, rec))

    results = [None] * len(records)

    for key, items in by_key.items():
        if key not in model_map:
            print(f"  WARNING: no model for key={key!r} ({len(items)} images skipped)")
            for i, _ in items:
                results[i] = 'NO_MODEL'
            continue

        model, class_names, _ = model_map[key]
        recs_batch = [rec for _, rec in items]

        class EOCSubset(Dataset):
            def __init__(self, recs, tf):
                self.recs = recs; self.tf = tf
            def __len__(self): return len(self.recs)
            def __getitem__(self, idx):
                try:
                    img = Image.open(self.recs[idx]['local_path']).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (128, 128), color=(64, 64, 64))
                return self.tf(img)

        loader = DataLoader(EOCSubset(recs_batch, transform),
                            batch_size=batch_size, shuffle=False, num_workers=2)
        preds = []
        with torch.no_grad():
            for imgs in loader:
                logits = model(imgs.to(device))
                preds.extend([class_names[i] for i in logits.argmax(1).cpu().tolist()])

        for (orig_idx, _), pred in zip(items, preds):
            results[orig_idx] = pred

    return results


def accuracy_breakdown(records, preds, key_fn, key_label):
    by_key = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    for rec, pred in zip(records, preds):
        if pred == 'NO_MODEL':
            continue
        k = key_fn(rec)
        by_key[k]['total']   += 1
        by_key[k]['correct'] += int(pred == rec['true_class'])
        overall_correct += int(pred == rec['true_class'])

    valid_total = sum(1 for p in preds if p != 'NO_MODEL')
    overall_acc = overall_correct / valid_total * 100 if valid_total else 0

    print(f"\n  By {key_label}:")
    for k in sorted(by_key):
        d = by_key[k]
        print(f"    {k:<6}  {d['correct']/d['total']*100:.1f}%  ({d['total']:,} images)")
    return overall_acc


def main():
    device = get_device()
    print(f"\nDevice: {device}")

    # ── Load pol-specific L1 models ───────────────────────────────────────────
    print("\nLoading pol-routing models (pol_L1_*_v1)...")
    pol_models = {}
    transform  = None
    for pol in ('HH', 'HV', 'VH', 'VV'):
        d = find_model(f'pol_L1_{pol}_v1')
        if d:
            model, classes, tf = load_model(d, device)
            pol_models[pol] = (model, classes, tf)
            transform = tf
            print(f"  Loaded {pol}: {d.name}  ({len(classes)} classes: {classes})")
        else:
            print(f"  MISSING: pol_L1_{pol}_v1 — run train_pol_l1_smoke.sh first")

    # ── Load angle-specific L1 models ─────────────────────────────────────────
    print("\nLoading angle-routing models (cascade_L1_angle*_v1)...")
    angle_models = {}
    for angle in (15, 30, 45, 60):
        d = find_model(f'cascade_L1_angle{angle}_v1')
        if d:
            model, classes, tf = load_model(d, device)
            angle_models[angle] = (model, classes, tf)
            if transform is None:
                transform = tf
            print(f"  Loaded {angle}°: {d.name}  ({len(classes)} classes: {classes})")
        else:
            print(f"  MISSING: cascade_L1_angle{angle}_v1")

    if not pol_models and not angle_models:
        print("\nNo models found. Train models first.")
        return

    if transform is None:
        import importlib
        transform = importlib.import_module('models.convnext').get_transform()

    # ── Load EOC-Pol images ───────────────────────────────────────────────────
    print("\nLoading EOC_pol images...")
    records = load_eoc_pol_images()
    print(f"  {len(records):,} images")

    # ── Pol-routing eval ──────────────────────────────────────────────────────
    if pol_models:
        print(f"\n{'='*60}")
        print(f"  POL-ROUTING (route by polarization metadata)")
        print(f"{'='*60}")
        pol_preds = run_routing(records, pol_models,
                                key_fn=lambda r: r['polarization'],
                                transform=transform, device=device)
        pol_acc = accuracy_breakdown(records, pol_preds,
                                     key_fn=lambda r: r['polarization'],
                                     key_label='polarization')
        print(f"\n  Overall L1 accuracy (pol-routing): {pol_acc:.1f}%")

    # ── Angle-routing eval ────────────────────────────────────────────────────
    if angle_models:
        print(f"\n{'='*60}")
        print(f"  ANGLE-ROUTING (route by depression angle metadata)")
        print(f"{'='*60}")
        angle_preds = run_routing(records, angle_models,
                                  key_fn=lambda r: r['angle'],
                                  transform=transform, device=device)
        angle_acc = accuracy_breakdown(records, angle_preds,
                                       key_fn=lambda r: r['angle'],
                                       key_label='angle')
        print(f"\n  Overall L1 accuracy (angle-routing): {angle_acc:.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    if pol_models and angle_models:
        delta = pol_acc - angle_acc
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Pol-routing:   {pol_acc:.1f}%")
        print(f"  Angle-routing: {angle_acc:.1f}%")
        print(f"  Delta:         {delta:+.1f}%")
        if delta > 2:
            print(f"\n  >> Pol-routing wins — worth building full pol cascade (L2+L3)")
        elif delta > 0:
            print(f"\n  >> Marginal pol advantage — may not justify full rebuild")
        else:
            print(f"\n  >> Angle-routing >= pol-routing — issue is feature space, not routing")


if __name__ == '__main__':
    main()
