"""
scripts/eval_prototypical.py
==============================
Prototypical (nearest-centroid) classification using the SOC backbone.

No retraining. Build prototypes from a small labeled reference set,
classify test images by nearest centroid in the 768-dim feature space.

Two prototype strategies:
  --strategy 15only   : prototypes from N images per vehicle at 15° only
  --strategy all      : prototypes from N images per vehicle at ALL 4 angles

Test set: EOC_depression 30°/45°/60° images (same as EOC baseline = 43.7%)

This answers: "How few reference images do you need — and at which angles —
before a frozen SOC backbone can identify vehicles at unseen angles?"

Usage:
    python3 scripts/eval_prototypical.py                   # full sweep
    python3 scripts/eval_prototypical.py --n-shot 10       # one value
    python3 scripts/eval_prototypical.py --strategy 15only # 15° refs only
    python3 scripts/eval_prototypical.py --plot-curve      # plot saved results
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'prototypical'
SEED       = 42
N_SHOTS    = [5, 10, 25, 50, 100, 200, 'all']


# ── Data ──────────────────────────────────────────────────────────────────────

def load_reference_pool() -> list:
    """SOC_40classes images — used to build prototypes."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND depression_angle IS NOT NULL
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_test_records() -> list:
    """EOC_depression test images: 30°/45°/60° — same split as EOC baseline."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND split = 'test'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def sample_reference(pool: list, n_shot, strategy: str,
                     seed: int = SEED) -> list:
    """
    Sample reference images from pool.
    strategy='15only' : only 15° images, N per vehicle
    strategy='all'    : all 4 angles, N per (vehicle, angle)
    n_shot='all'      : use everything available
    """
    rng = np.random.default_rng(seed)

    if strategy == '15only':
        by_vehicle = defaultdict(list)
        for r in pool:
            if float(r['depression_angle']) == 15.0:
                by_vehicle[r['target_type']].append(r)
        ref = []
        for v, recs in by_vehicle.items():
            recs = list(recs)
            rng.shuffle(recs)
            ref.extend(recs if n_shot == 'all' else recs[:n_shot])

    else:  # 'all' angles
        by_group = defaultdict(list)
        for r in pool:
            key = (r['target_type'], float(r['depression_angle']))
            by_group[key].append(r)
        ref = []
        for key, recs in by_group.items():
            recs = list(recs)
            rng.shuffle(recs)
            ref.extend(recs if n_shot == 'all' else recs[:n_shot])

    return ref


# ── Feature extraction ────────────────────────────────────────────────────────

class ChipDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        return self.transform(img), idx


def extract_features(model, hook_out, records, transform, device,
                     batch_size=64) -> np.ndarray:
    ds     = ChipDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    feats = [None] * len(records)
    with torch.no_grad():
        for imgs, indices in loader:
            _ = model(imgs.to(device))
            vecs = hook_out['vec'].numpy()
            for i, idx in enumerate(indices.tolist()):
                feats[idx] = vecs[i]
    return np.array(feats)


def load_soc_model(device):
    model_name = 'rot_baseline_full_v1'
    exp_dir    = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))[-1]
    config     = json.loads((exp_dir / 'config.json').read_text())
    mod        = importlib.import_module(f'models.{config.get("model_type","convnext_small")}')
    model      = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    hook_out = {}
    def hook(module, input, output):
        hook_out['vec'] = output.detach().cpu()
    model.classifier[1].register_forward_hook(hook)
    print(f"  SOC model: {exp_dir.name}  (val={config.get('best_val_acc','?')}%)")
    return model, hook_out, mod.get_transform()


# ── Prototypical classification ───────────────────────────────────────────────

def build_prototypes(ref_records, ref_features) -> dict:
    """Compute mean feature vector per vehicle."""
    groups = defaultdict(list)
    for i, r in enumerate(ref_records):
        groups[r['target_type']].append(ref_features[i])
    return {v: np.mean(vecs, axis=0) for v, vecs in groups.items()}


def classify_by_prototype(test_features, prototypes) -> list:
    """Nearest L2 centroid for each test feature."""
    vehicles = list(prototypes.keys())
    proto_mat = np.array([prototypes[v] for v in vehicles])  # (C, 768)
    preds = []
    for feat in test_features:
        dists = np.linalg.norm(proto_mat - feat, axis=1)
        preds.append(vehicles[np.argmin(dists)])
    return preds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(preds, test_records) -> dict:
    by_angle = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_correct = 0
    for pred, rec in zip(preds, test_records):
        angle = float(rec['depression_angle'])
        correct = int(pred == rec['target_type'])
        by_angle[angle]['correct'] += correct
        by_angle[angle]['total']   += 1
        overall_correct            += correct

    result = {'overall': round(overall_correct / len(test_records) * 100, 1)}
    for angle, counts in sorted(by_angle.items()):
        result[f'{int(angle)}deg'] = round(
            counts['correct'] / counts['total'] * 100, 1)
    return result


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(model, hook_out, transform, device,
            ref_pool, test_records, test_features,
            n_shot, strategy) -> dict:
    label = 'all' if n_shot == 'all' else str(n_shot)

    ref_records  = sample_reference(ref_pool, n_shot, strategy)
    ref_features = extract_features(model, hook_out, ref_records,
                                    transform, device)
    prototypes   = build_prototypes(ref_records, ref_features)

    # Only score test images whose class appears in prototypes
    valid_idx = [i for i, r in enumerate(test_records)
                 if r['target_type'] in prototypes]
    valid_test    = [test_records[i] for i in valid_idx]
    valid_feats   = test_features[valid_idx]

    preds  = classify_by_prototype(valid_feats, prototypes)
    result = evaluate(preds, valid_test)
    result.update({'n_shot': label, 'strategy': strategy,
                   'n_ref': len(ref_records),
                   'n_prototypes': len(prototypes)})

    print(f"  n={label:>4}  strategy={strategy:<7}  "
          f"overall={result['overall']:>5.1f}%  "
          f"30°={result.get('30deg','?'):>5}  "
          f"45°={result.get('45deg','?'):>5}  "
          f"60°={result.get('60deg','?'):>5}")
    return result


# ── Curve plotting ────────────────────────────────────────────────────────────

def plot_curve(all_results: list):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    for strategy, label, color, marker in [
        ('15only', '15° refs only',   'steelblue', 'o'),
        ('all',    'All-angle refs',  'tomato',    's'),
    ]:
        res = [r for r in all_results
               if r['strategy'] == strategy and r['n_shot'] != 'all']
        if not res:
            continue
        ns  = [int(r['n_shot']) for r in res]
        acc = [r['overall'] for r in res]
        ax.plot(ns, acc, f'{marker}-', color=color, label=label, linewidth=2)

        # 'all' points as separate markers
        for r in all_results:
            if r['strategy'] == strategy and r['n_shot'] == 'all':
                ax.axhline(r['overall'], color=color, linestyle=':',
                           alpha=0.6, label=f'{label} (all data): {r["overall"]}%')

    ax.axhline(97.2, color='green',  linestyle=':', alpha=0.7,
               label='SOC ceiling (full training, 97.2%)')
    ax.axhline(43.7, color='gray',   linestyle=':', alpha=0.7,
               label='EOC baseline (15°-only trained, 43.7%)')

    ax.set_xlabel('Reference images per vehicle (N-shot)', fontsize=12)
    ax.set_ylabel('Accuracy on 30°/45°/60° test images (%)', fontsize=12)
    ax.set_title('Prototypical Classification — SOC backbone, no retraining\n'
                 'How few reference images are needed to beat the trained EOC model?',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out = LOGS_DIR / 'prototypical_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-shot',     default=None,
                        help='Single N value (int or "all")')
    parser.add_argument('--strategy',   default='both',
                        choices=['15only', 'all', 'both'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--plot-curve', action='store_true',
                        help='Plot from saved results JSON')
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = LOGS_DIR / 'results.json'

    if args.plot_curve:
        results = json.loads(results_path.read_text())
        plot_curve(results)
        return

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    print("\nLoading SOC model...")
    model, hook_out, transform = load_soc_model(device)

    print("\nLoading reference pool (SOC_40classes)...")
    ref_pool = load_reference_pool()
    print(f"  {len(ref_pool):,} images available")

    print("\nLoading test records (EOC_depression 30°/45°/60°)...")
    test_records = load_test_records()
    test_records = [r for r in test_records
                    if float(r['depression_angle']) != 15.0]
    print(f"  {len(test_records):,} test images")

    print("\nExtracting test features (once)...")
    test_features = extract_features(model, hook_out, test_records,
                                     transform, device, args.batch_size)
    print(f"  {test_features.shape}")

    strategies = (['15only', 'all'] if args.strategy == 'both'
                  else [args.strategy])
    n_values   = (N_SHOTS if args.n_shot is None
                  else [args.n_shot if args.n_shot == 'all'
                        else int(args.n_shot)])

    print(f"\n{'='*65}")
    print(f"  Prototypical Classification — SOC backbone, no retraining")
    print(f"  EOC baseline (trained model): 43.7%")
    print(f"{'='*65}")

    all_results = []
    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        for n in n_values:
            r = run_one(model, hook_out, transform, device,
                        ref_pool, test_records, test_features,
                        n, strategy)
            all_results.append(r)

    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    if args.n_shot is None:
        plot_curve(all_results)


if __name__ == '__main__':
    main()
