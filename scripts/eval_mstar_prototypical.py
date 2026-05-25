"""
scripts/eval_mstar_prototypical.py
====================================
Prototypical classification on MSTAR using the MSTAR-trained SOC backbone.

Mirrors eval_prototypical.py exactly but on MSTAR:
  - Reference images: 17° (N-shot per class)
  - Test images: 15°
  - Backbone: MSTAR SOC model (train_mstar_soc.py must run first)

Comparison targets:
  SOC trained (17→15):        ~97-99%  (from train_mstar_soc.py)
  Prototypical n=5 (17° refs): ?        (this script)

Usage:
    python3 scripts/eval_mstar_prototypical.py
    python3 scripts/eval_mstar_prototypical.py --n-shot 5
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
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'mstar_prototypical'
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


def load_mstar_model(device):
    save_dir = MODELS_DIR / 'mstar_soc_v1'
    config   = json.loads((save_dir / 'config.json').read_text())
    mod      = importlib.import_module('models.convnext_small')
    model    = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                      map_location=device, weights_only=True))
    model.eval()
    hook_out = {}
    def hook(module, inp, out):
        hook_out['vec'] = out.detach().cpu()
    model.classifier[1].register_forward_hook(hook)
    print(f"  Loaded MSTAR SOC model (val={config.get('best_val_acc','?')}%)")
    return model, hook_out, mod.get_transform()


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

    ref_feats  = extract_features(model, hook_out, sampled, transform, device)
    prototypes = {}
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
    acc = round(correct / len(test_recs) * 100, 1)
    label = 'all' if n_shot == 'all' else str(n_shot)
    print(f"  n={label:>4}: {acc:>5.1f}%  ({len(sampled):,} ref images)")
    return {'n_shot': label, 'accuracy': acc, 'n_ref': len(sampled)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-shot', default=None)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"\nDevice: {device}")
    print("="*55)
    print("  MSTAR Prototypical — MSTAR SOC backbone")
    print("  Reference: 17°  →  Test: 15°")
    print("="*55)

    model, hook_out, transform = load_mstar_model(device)

    ref_recs  = load_records(17.0)
    test_recs = load_records(15.0)
    print(f"  Reference pool (17°): {len(ref_recs):,}  Test (15°): {len(test_recs):,}")

    print("\nExtracting test features...")
    test_feats = extract_features(model, hook_out, test_recs, transform, device)

    n_values = (N_SHOTS if args.n_shot is None
                else [args.n_shot if args.n_shot == 'all' else int(args.n_shot)])

    print(f"\n  Prototypical results:")
    results = []
    for n in n_values:
        r = run_one(model, hook_out, transform, device,
                    ref_recs, test_recs, test_feats, n)
        results.append(r)

    (LOGS_DIR / 'results.json').write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {LOGS_DIR}/results.json")


if __name__ == '__main__':
    main()
