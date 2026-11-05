"""
scripts/eval_fewshot_novel.py
==============================
Does depression-angle diversity during training generalize to classes the
backbone never saw at all, not just angles it never saw?

For each backbone in scripts/train_fewshot_backbone.py (trained on the 30
BASE classes under a given angle-diversity condition), freeze it as a
feature extractor and run prototypical (nearest-centroid) classification
on the 10 NOVEL classes — never present in any backbone's training data.

Protocol:
  Support (reference) set : novel-class images at 15° only, k per class
  Query (test) set        : novel-class images at 30°/45°/60°, held out by
                             angle from the support set

Sweeping k-shot x training-angle-diversity gives the key comparison: if
angle diversity only helps within-class generalization, novel-class
accuracy should be flat across backbones. If it reflects a more general
inductive bias, the all-4-angle backbone should beat the 15°-only backbone
here too.

Usage:
    python3 scripts/eval_fewshot_novel.py                # full sweep
    python3 scripts/eval_fewshot_novel.py --k-shot 5      # one k value
    python3 scripts/eval_fewshot_novel.py --plot-curve    # plot saved results
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
from train_fewshot_backbone import NOVEL_CLASSES, ANGLE_SETS

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'fewshot_novel'
SEED       = 42
K_SHOTS    = [1, 5, 10, 25, 'all']


# ── Data ──────────────────────────────────────────────────────────────────────

def load_novel_class_images() -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, local_path, target_type, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type = ANY(%s)
              AND depression_angle IS NOT NULL
        """, (NOVEL_CLASSES,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def sample_support(pool: list, k_shot, seed: int = SEED) -> list:
    """k images per novel class, 15° only."""
    rng = np.random.default_rng(seed)
    support_pool = [r for r in pool if float(r['depression_angle']) == 15.0]
    by_class = defaultdict(list)
    for r in support_pool:
        by_class[r['target_type']].append(r)
    support = []
    for cls, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)
        support.extend(recs if k_shot == 'all' else recs[:k_shot])
    return support


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


def load_fewshot_backbone(angle_str: str, device):
    exp_dir = MODELS_DIR / f'fewshot_backbone_{angle_str}_v1'
    config  = json.loads((exp_dir / 'config.json').read_text())
    mod     = importlib.import_module(f'models.{config["model_type"]}')
    model   = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    hook_out = {}
    def hook(module, input, output):
        hook_out['vec'] = output.detach().cpu()
    model.classifier[1].register_forward_hook(hook)
    print(f"  Backbone: {exp_dir.name}  (base-class val={config['best_val_acc']}%)")
    return model, hook_out, mod.get_transform()


# ── Prototypical classification ───────────────────────────────────────────────

def build_prototypes(ref_records, ref_features) -> dict:
    groups = defaultdict(list)
    for i, r in enumerate(ref_records):
        groups[r['target_type']].append(ref_features[i])
    return {v: np.mean(vecs, axis=0) for v, vecs in groups.items()}


def classify_by_prototype(test_features, prototypes) -> list:
    classes  = list(prototypes.keys())
    proto_mat = np.array([prototypes[c] for c in classes])
    preds = []
    for feat in test_features:
        dists = np.linalg.norm(proto_mat - feat, axis=1)
        preds.append(classes[np.argmin(dists)])
    return preds


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

def run_one(model, hook_out, transform, device, angle_str,
            support_pool, query_records, query_features, k_shot) -> dict:
    label = 'all' if k_shot == 'all' else str(k_shot)

    support_records  = sample_support(support_pool, k_shot)
    support_features = extract_features(model, hook_out, support_records,
                                        transform, device)
    prototypes = build_prototypes(support_records, support_features)

    preds  = classify_by_prototype(query_features, prototypes)
    result = evaluate(preds, query_records)
    result.update({'k_shot': label, 'angle_str': angle_str,
                   'n_support': len(support_records)})

    print(f"  angles={angle_str:<10}  k={label:>4}  "
          f"overall={result['overall']:>5.1f}%  "
          f"30°={result.get('30deg','?'):>5}  "
          f"45°={result.get('45deg','?'):>5}  "
          f"60°={result.get('60deg','?'):>5}")
    return result


# ── Curve plotting ────────────────────────────────────────────────────────────

def plot_curve(all_results: list):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))

    angle_strs = sorted(set(r['angle_str'] for r in all_results),
                        key=lambda s: s.count('+'))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(angle_strs)))

    for angle_str, color in zip(angle_strs, colors):
        res = [r for r in all_results
               if r['angle_str'] == angle_str and r['k_shot'] != 'all']
        if not res:
            continue
        res.sort(key=lambda r: int(r['k_shot']))
        ks  = [int(r['k_shot']) for r in res]
        acc = [r['overall'] for r in res]
        ax.plot(ks, acc, 'o-', color=color, linewidth=2,
                label=f'trained on {angle_str}°')

    ax.set_xlabel('Support images per novel class (k-shot)', fontsize=12)
    ax.set_ylabel('Accuracy on held-out novel classes, 30°/45°/60° query (%)',
                 fontsize=12)
    ax.set_title('Novel-Class Transfer vs. Training-Angle Diversity\n'
                 'Does angle-diverse training generalize beyond the training classes?',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()

    out = LOGS_DIR / 'fewshot_novel_curve.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-shot',     default=None,
                        help='Single k value (int or "all")')
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

    print("\nLoading novel-class images (10 held-out target types)...")
    pool = load_novel_class_images()
    print(f"  {len(pool):,} images across {len(NOVEL_CLASSES)} novel classes")

    support_pool  = pool
    query_records = [r for r in pool if float(r['depression_angle']) != 15.0]
    print(f"  Support pool (15°): "
          f"{sum(1 for r in pool if float(r['depression_angle'])==15.0):,}")
    print(f"  Query pool (30/45/60°): {len(query_records):,}")

    angle_strs = ['+'.join(str(int(a)) for a in sorted(s)) for s in ANGLE_SETS]
    k_values   = (K_SHOTS if args.k_shot is None
                  else [args.k_shot if args.k_shot == 'all' else int(args.k_shot)])

    print(f"\n{'='*70}")
    print(f"  Novel-Class Transfer — prototypical eval, frozen backbones")
    print(f"{'='*70}")

    all_results = []
    for angle_str in angle_strs:
        model, hook_out, transform = load_fewshot_backbone(angle_str, device)
        print("  Extracting query features (once per backbone)...")
        query_features = extract_features(model, hook_out, query_records,
                                          transform, device, args.batch_size)
        for k in k_values:
            r = run_one(model, hook_out, transform, device, angle_str,
                       support_pool, query_records, query_features, k)
            all_results.append(r)

    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    if args.k_shot is None:
        plot_curve(all_results)


if __name__ == '__main__':
    main()
