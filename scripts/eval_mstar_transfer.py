"""
scripts/eval_mstar_transfer.py
================================
Cross-dataset transfer: ATRNet-STAR SOC backbone → MSTAR prototypical.

Zero retraining on MSTAR. Uses the ConvNeXt backbone trained on ATRNet-STAR
SOC (40 classes, 4 depression angles) to classify MSTAR vehicles by nearest
centroid in feature space.

This is the most interesting experiment for the paper:
  - If it works: "ATRNet-STAR features generalize across datasets"
  - If it doesn't: "domain gap between datasets is real"
  Either result is a publishable finding.

Reference images: MSTAR 17° (N-shot per class)
Test images:      MSTAR 15°
Backbone:         ATRNet-STAR SOC model (exp_108_rot_baseline_full_v1)

Usage:
    python3 scripts/eval_mstar_transfer.py
    python3 scripts/eval_mstar_transfer.py --n-shot 5
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG
from ingest_mstar import read_mstar, mag_to_uint8
from train_mstar_soc import LABEL_MAP, get_device

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'mstar_transfer'
SEED       = 42
N_SHOTS    = [1, 5, 10, 25, 50, 'all']


class MSTARDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            _, mag = read_mstar(Path(rec['local_path']))
            if mag is None: raise ValueError
            img = Image.fromarray(mag_to_uint8(mag)).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        return self.transform(img), idx


def load_records(depression: float) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        ph = ','.join(['%s'] * len(LABEL_MAP))
        cur.execute(f"""
            SELECT local_path, target_short
            FROM defense.mstar_images
            WHERE desired_depression = %s
              AND target_short IN ({ph})
              AND local_path IS NOT NULL
        """, [depression] + list(LABEL_MAP.keys()))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r['label'] = LABEL_MAP[r['target_short']]
    return rows


def load_atrnet_backbone(device):
    """Load the ATRNet-STAR SOC backbone — same model used in eval_prototypical.py."""
    model_name = 'rot_baseline_full_v1'
    exp_dirs   = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))
    if not exp_dirs:
        raise FileNotFoundError(f"No model matching exp_*_{model_name} in {MODELS_DIR}")
    exp_dir = exp_dirs[-1]
    config  = json.loads((exp_dir / 'config.json').read_text())
    mod     = importlib.import_module(f'models.{config.get("model_type","convnext_small")}')
    model   = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(torch.load(exp_dir / 'weights.pt',
                                      map_location=device, weights_only=True))
    model.eval()
    hook_out = {}
    def hook(module, inp, out):
        hook_out['vec'] = out.detach().cpu()
    model.classifier[1].register_forward_hook(hook)
    print(f"  ATRNet-STAR backbone: {exp_dir.name}  (val={config.get('best_val_acc','?')}%)")
    print(f"  Trained on: ATRNet-STAR SOC 40 classes, 4 angles")
    print(f"  Now being applied to: MSTAR 10 classes, 0 retraining")
    return model, hook_out, mod.get_transform()


def extract_features(model, hook_out, records, transform, device) -> np.ndarray:
    ds     = MSTARDataset(records, transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)
    feats = [None] * len(records)
    with torch.no_grad():
        for imgs, indices in loader:
            _ = model(imgs.to(device))
            vecs = hook_out['vec'].numpy()
            for i, idx in enumerate(indices.tolist()):
                feats[idx] = vecs[i]
    return np.array(feats)


def run_one(model, hook_out, transform, device,
            ref_recs, test_recs, test_feats, n_shot) -> dict:
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for r in ref_recs:
        by_class[r['label']].append(r)
    sampled = []
    for cls, recs in by_class.items():
        recs = list(recs); rng.shuffle(recs)
        sampled.extend(recs if n_shot == 'all' else recs[:n_shot])

    ref_feats = extract_features(model, hook_out, sampled, transform, device)
    groups = defaultdict(list)
    for i, r in enumerate(sampled):
        groups[r['label']].append(ref_feats[i])
    prototypes = {cls: np.mean(vecs, 0) for cls, vecs in groups.items()}

    vehicles  = list(prototypes.keys())
    proto_mat = np.array([prototypes[v] for v in vehicles])
    correct   = sum(
        1 for feat, rec in zip(test_feats, test_recs)
        if vehicles[np.argmin(np.linalg.norm(proto_mat - feat, axis=1))] == rec['label']
    )
    acc   = round(correct / len(test_recs) * 100, 1)
    label = 'all' if n_shot == 'all' else str(n_shot)
    print(f"  n={label:>4}: {acc:>5.1f}%")
    return {'n_shot': label, 'accuracy': acc, 'backbone': 'atrnet_soc'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-shot', default=None)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"\nDevice: {device}")
    print("="*60)
    print("  Cross-Dataset Transfer: ATRNet-STAR backbone → MSTAR")
    print("  Zero MSTAR training. Reference: 17°  Test: 15°")
    print("="*60)

    model, hook_out, transform = load_atrnet_backbone(device)

    ref_recs  = load_records(17.0)
    test_recs = load_records(15.0)
    print(f"\n  Reference pool (17°): {len(ref_recs):,}  Test (15°): {len(test_recs):,}")

    print("\nExtracting test features with ATRNet-STAR backbone...")
    test_feats = extract_features(model, hook_out, test_recs, transform, device)

    n_values = (N_SHOTS if args.n_shot is None
                else [args.n_shot if args.n_shot == 'all' else int(args.n_shot)])

    print(f"\n  Transfer accuracy (ATRNet-STAR backbone, MSTAR test set):")
    results = []
    for n in n_values:
        r = run_one(model, hook_out, transform, device,
                    ref_recs, test_recs, test_feats, n)
        results.append(r)

    (LOGS_DIR / 'results.json').write_text(json.dumps(results, indent=2))

    print(f"\n  Summary:")
    print(f"  {'N':>6}  {'Accuracy':>10}")
    print(f"  {'-'*20}")
    for r in results:
        print(f"  {r['n_shot']:>6}  {r['accuracy']:>9.1f}%")
    print(f"\nResults saved to {LOGS_DIR}/results.json")


if __name__ == '__main__':
    main()
