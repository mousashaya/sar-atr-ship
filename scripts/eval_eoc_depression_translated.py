"""
scripts/eval_eoc_depression_translated.py
==========================================
Evaluates EOC-Depression using image translation + 15° classifier.

Pipeline:
  EOC chip at 30°/45°/60°
    → translator_{angle}to15 (U-Net)
    → synthetic 15° chip
    → 15° vehicle classifier (ConvNeXt-Base)
    → vehicle class prediction

Since we know the depression angle of every EOC_depression chip,
we pick the correct translator directly — no routing needed.

Requirements (must train first):
  1. Train all translators:
     python3 scripts/train_translator.py --name translator_v1 --all-angles
  2. Train 15° vehicle classifier (already done via train.py --angle 15)

Usage:
    python3 scripts/eval_eoc_depression_translated.py --name translator_v1
    python3 scripts/eval_eoc_depression_translated.py --name translator_v1 --verbose
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

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

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
EOC_ANGLES = [30.0, 45.0, 60.0]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Model loading ─────────────────────────────────────────────────────────────

def load_translators(name: str, device) -> dict:
    """Load all 3 U-Net translators. Returns {angle: model}."""
    unet_module  = importlib.import_module('models.unet')
    translators  = {}
    for angle in [30, 45, 60]:
        save_dir = MODELS_DIR / f'translator_{name}_{angle}to15'
        weights  = save_dir / 'weights.pt'
        if not weights.exists():
            raise FileNotFoundError(
                f"Translator not found: {weights}\n"
                f"Run: python3 scripts/train_translator.py "
                f"--name {name} --source-angle {angle}")
        model = unet_module.build_translator().to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        translators[float(angle)] = model
        print(f"  Loaded translator {angle}°→15°: {weights}")
    return translators


def find_15deg_classifier(device):
    """Auto-discover the 15° angle-specific vehicle classifier."""
    pattern = 'convnext_40class_angle15_v1'
    matches = [d for d in MODELS_DIR.iterdir()
               if d.is_dir() and pattern in d.name]
    if not matches:
        raise FileNotFoundError(
            f"No 15° classifier found (pattern: {pattern}). "
            f"Run train.py with --angle 15 --config 40class first.")
    save_dir    = sorted(matches)[-1]
    model_cfg   = json.loads((save_dir / 'config.json').read_text())
    class_names = model_cfg['class_names']

    convnext_module = importlib.import_module('models.convnext')
    model = convnext_module.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))
    model.eval()
    print(f"  Loaded 15° classifier: {save_dir.name}  ({len(class_names)} classes)")
    return model, class_names


# ── Data ──────────────────────────────────────────────────────────────────────

class EOCDataset(Dataset):
    def __init__(self, records, unet_transform, convnext_transform,
                 translator, device_cpu):
        self.records             = records
        self.unet_transform      = unet_transform
        self.convnext_transform  = convnext_transform
        self.translator          = translator
        self.device_cpu          = device_cpu

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), color=(64, 64, 64))
        return self.unet_transform(img), rec['image_id'], rec['true_class']


def load_eoc_depression_images() -> dict:
    """Returns {angle: [records]} for 30°, 45°, 60°."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, depression_angle, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND local_path IS NOT NULL
              AND depression_angle IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
        """)
        rows = cur.fetchall()
    conn.close()

    by_angle = defaultdict(list)
    for r in rows:
        by_angle[float(r['depression_angle'])].append({
            'image_id':   r['image_id'],
            'local_path': r['local_path'],
            'true_class': r['target_type'],
            'true_angle': float(r['depression_angle']),
        })
    return by_angle


class SimpleDS(Dataset):
    def __init__(self, recs, tfm):
        self.recs = recs
        self.tfm  = tfm
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        rec = self.recs[i]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128))
        return self.tfm(img), rec['true_class']


# ── Inference ─────────────────────────────────────────────────────────────────

def run_angle(records, translator, classifier, class_names,
              unet_transform, convnext_transform,
              device, batch_size, angle):
    """
    For chips at one depression angle:
      1. Translate batch to synthetic 15°
      2. Resize/normalize for ConvNeXt
      3. Classify
    """
    loader = DataLoader(SimpleDS(records, unet_transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)

    from torchvision import transforms as T
    # ConvNeXt normalization (applied to translated chip)
    post_process = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    # Denormalize U-Net output from [-1,1] to [0,1]
    def denorm(t):
        return (t * 0.5 + 0.5).clamp(0, 1)

    all_preds, all_true = [], []

    with torch.no_grad():
        for imgs, true_classes in loader:
            imgs = imgs.to(device)

            # Step 1: translate to synthetic 15°
            translated = translator(imgs)          # [-1, 1]
            translated = denorm(translated)        # [0, 1]

            # Step 2: apply ConvNeXt normalization and resize
            cls_input = post_process(translated)   # [B, 3, 224, 224]

            # Step 3: classify
            logits   = classifier(cls_input)
            pred_idx = logits.argmax(dim=1).cpu().tolist()

            all_preds.extend([class_names[i] for i in pred_idx])
            all_true.extend(true_classes)

    acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    print(f"    {angle:>4.0f}°:  {acc*100:.1f}%  ({len(records):,} images)")
    return all_preds, all_true


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',       required=True,
                        help='Translator base name (e.g. translator_v1)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--verbose',    action='store_true')
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")

    print("\nLoading models...")
    translators             = load_translators(args.name, device)
    classifier, class_names = find_15deg_classifier(device)

    unet_module      = importlib.import_module('models.unet')
    unet_transform   = unet_module.get_transform()

    print("\nLoading EOC_depression images...")
    by_angle = load_eoc_depression_images()
    total = sum(len(v) for v in by_angle.values())
    print(f"  Total: {total:,} images")
    for a in sorted(by_angle):
        print(f"    {a:>4.0f}°:  {len(by_angle[a]):,} images")

    print("\nRunning translation + classification...")
    all_preds, all_true = [], []

    for angle in sorted(by_angle.keys()):
        records    = by_angle[angle]
        # 15° images don't need translation — already in target domain
        translator = translators.get(angle, None)
        if translator is None:
            print(f"  {angle:>4.0f}°: no translator needed (already 15°), classifying directly...")
            translator = torch.nn.Identity().to(device)
        preds, true = run_angle(
            records, translator, classifier, class_names,
            unet_transform, None, device, args.batch_size, angle)
        all_preds.extend(preds)
        all_true.extend(true)

    overall_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)

    class_set = sorted(set(all_true))
    cm = confusion_matrix(all_true, all_preds, labels=class_set)

    print(f"\n{'='*60}")
    print(f"  EOC-DEPRESSION (translation pipeline)")
    print(f"{'='*60}")
    print(f"  Overall accuracy:          {overall_acc*100:.1f}%")
    print(f"  Azimuth routing baseline:  37.6%")
    print(f"  ATRNet-STAR SOTA:          43.1%")
    if overall_acc * 100 > 43.1:
        print(f"  *** BEATS SOTA by {overall_acc*100 - 43.1:.1f} pp ***")

    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':<35} {'Acc':>8}  {'N':>8}")
    for i, cls in enumerate(class_set):
        n   = cm[i].sum()
        acc = cm[i, i] / n if n > 0 else 0
        print(f"  {cls:<35} {acc*100:>7.1f}%  {n:>8,}")

    out_path = (Path(__file__).parent.parent / 'logs' /
                f'eoc_depression_translated_{args.name}.json')
    result = {
        'translator_name':  args.name,
        'overall_acc':      round(overall_acc * 100, 1),
        'azimuth_baseline': 37.6,
        'sota':             43.1,
        'delta_vs_sota':    round(overall_acc * 100 - 43.1, 1),
        'n_images':         len(all_true),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
