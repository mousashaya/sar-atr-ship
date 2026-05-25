"""
scripts/eval_tta.py
====================
Test-Time Augmentation (TTA) on existing E3 models — no retraining.

For each test image, run K augmented versions through the model and
average the softmax predictions. Measures free accuracy gain at inference.

Augmentations: horizontal flip, vertical flip, 90/180/270° rotations.
All are valid for SAR imagery (no up/down orientation constraint).

Usage:
    python3 scripts/eval_tta.py                    # all E3 models
    python3 scripts/eval_tta.py --max-per-group 25 # one model
    python3 scripts/eval_tta.py --k 5              # number of augmentations
"""

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'tta'
N_VALUES   = [5, 10, 25, 50, 100, 200]
SEED       = 42

class TTATransform:
    def __init__(self, flip_h=False, flip_v=False, rotate=0):
        self.flip_h  = flip_h
        self.flip_v  = flip_v
        self.rotate  = rotate

    def __call__(self, img):
        if self.rotate:
            img = TF.rotate(img, self.rotate)
        if self.flip_h:
            img = TF.hflip(img)
        if self.flip_v:
            img = TF.vflip(img)
        return img


TTA_TRANSFORMS = [
    TTATransform(),                              # original
    TTATransform(flip_h=True),                   # hflip
    TTATransform(flip_v=True),                   # vflip
    TTATransform(rotate=90),                     # 90°
    TTATransform(rotate=180),                    # 180°
    TTATransform(rotate=270),                    # 270°
    TTATransform(rotate=90, flip_h=True),        # 90° + hflip
]


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_test_records(seed: int = SEED) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND depression_angle IS NOT NULL
              AND depression_angle = 15.0
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


class ChipDataset(Dataset):
    def __init__(self, records, class_to_idx, base_transform, tta_fns):
        self.records         = records
        self.class_to_idx    = class_to_idx
        self.base_transform  = base_transform
        self.tta_fns         = tta_fns

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        # Return all augmented versions
        tensors = torch.stack([self.base_transform(fn(img)) for fn in self.tta_fns])
        label   = self.class_to_idx[rec['target_type']]
        return tensors, label


def load_e3_model(n: int, device):
    save_dir    = MODELS_DIR / f'data_efficiency_n{n}_v1'
    config_path = save_dir / 'config.json'
    weights_path = save_dir / 'weights.pt'
    if not weights_path.exists():
        return None, None, None, None

    config      = json.loads(config_path.read_text())
    class_names = config.get('class_names')

    # Reconstruct class_names from DB if not in config
    if class_names is None:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""SELECT DISTINCT target_type FROM defense.atrnet_images
                           WHERE experiment_setting='SOC_40classes'
                             AND target_type IS NOT NULL AND target_type != ''
                           ORDER BY target_type""")
            class_names = [r[0] for r in cur.fetchall()]
        conn.close()

    mod   = importlib.import_module('models.convnext_small')
    model = mod.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model, mod.get_transform(), class_names, config


def evaluate_tta(model, records, class_to_idx, transform, tta_fns, device,
                 batch_size=32):
    ds     = ChipDataset(records, class_to_idx, transform, tta_fns)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, persistent_workers=True)

    correct_base, correct_tta, total = 0, 0, 0
    with torch.no_grad():
        for tensors, labels in loader:
            # tensors: (B, K, C, H, W)
            B, K, C, H, W = tensors.shape
            labels = labels.to(device)

            # Run all K augmentations in one batch
            flat   = tensors.view(B * K, C, H, W).to(device)
            logits = model(flat)                        # (B*K, num_classes)
            probs  = F.softmax(logits, dim=1)
            probs  = probs.view(B, K, -1)              # (B, K, num_classes)

            # Base (original only, index 0)
            base_pred = probs[:, 0, :].argmax(1)
            correct_base += (base_pred == labels).sum().item()

            # TTA (average all K)
            tta_pred  = probs.mean(dim=1).argmax(1)
            correct_tta  += (tta_pred == labels).sum().item()

            total += B

    return correct_base / total, correct_tta / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-per-group', type=int, default=None)
    parser.add_argument('--k',             type=int, default=7,
                        help='Number of TTA augmentations (max 7)')
    parser.add_argument('--batch-size',    type=int, default=32)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device  = get_device()
    tta_fns = TTA_TRANSFORMS[:args.k]

    print(f"\nDevice: {device}")
    print(f"TTA augmentations: {len(tta_fns)}")
    print(f"  (original, hflip, vflip, rot90, rot180, rot270, rot90+hflip)")

    test_records = load_test_records()
    print(f"Test records (15°): {len(test_records):,}")

    n_values = ([args.max_per_group] if args.max_per_group
                else N_VALUES)

    results = []
    print(f"\n{'='*60}")
    print(f"  TTA Evaluation — E3 models")
    print(f"  EOC baseline: 43.7%")
    print(f"{'='*60}")
    print(f"  {'N':>4}   {'Base':>7}   {'TTA':>7}   {'Delta':>7}")
    print(f"  {'-'*40}")

    for n in n_values:
        model, transform, class_names, config = load_e3_model(n, device)
        if model is None:
            print(f"  {n:>4}   -- model not found --")
            continue

        # Filter test records to classes in this model
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        valid = [r for r in test_records if r['target_type'] in class_to_idx]

        base_acc, tta_acc = evaluate_tta(
            model, valid, class_to_idx, transform, tta_fns,
            device, args.batch_size)

        delta = (tta_acc - base_acc) * 100
        print(f"  {n:>4}   {base_acc*100:>6.1f}%   {tta_acc*100:>6.1f}%   {delta:>+6.1f}%")

        results.append({
            'max_per_group': n,
            'base_acc':      round(base_acc * 100, 1),
            'tta_acc':       round(tta_acc * 100, 1),
            'delta':         round(delta, 1),
            'k':             len(tta_fns),
        })

    out = LOGS_DIR / 'tta_results.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")

    # Plot
    if len(results) > 1:
        ns    = [r['max_per_group'] for r in results]
        base  = [r['base_acc'] for r in results]
        tta   = [r['tta_acc']  for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ns, base, 'o-',  color='steelblue', label='Baseline (no TTA)')
        ax.plot(ns, tta,  's--', color='tomato',    label=f'TTA (K={len(tta_fns)})')
        ax.axhline(97.2, color='green', linestyle=':', alpha=0.7,
                   label='SOC ceiling (97.2%)')
        ax.axhline(43.7, color='gray',  linestyle=':', alpha=0.7,
                   label='EOC baseline (43.7%)')
        ax.set_xlabel('Max images per (vehicle, angle) group')
        ax.set_ylabel('Accuracy on held-out 15° test images (%)')
        ax.set_title(f'Test-Time Augmentation — free gain at inference\n'
                     f'K={len(tta_fns)} augmentations, no retraining')
        ax.legend(fontsize=11)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plot_out = LOGS_DIR / 'tta_curve.png'
        plt.savefig(plot_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_out}")


if __name__ == '__main__':
    main()
