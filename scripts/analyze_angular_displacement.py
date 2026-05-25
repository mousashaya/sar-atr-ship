"""
scripts/analyze_angular_displacement.py
========================================
Q1: How consistent is the angular displacement across vehicles in feature space?

For each vehicle, extract the SOC model's 768-dim feature centroid at each
depression angle (15/30/45/60). Then compute the displacement vectors:
    delta_30[v] = centroid[v,30] - centroid[v,15]
    delta_45[v] = centroid[v,45] - centroid[v,15]
    delta_60[v] = centroid[v,60] - centroid[v,15]

If these displacement vectors are consistent across vehicles (high cosine
similarity), the analogy transfer holds:
    Car_30° ≈ Car_15° + mean_delta_30

This is the mathematical foundation for feature-space angle transfer (E2).

Usage:
    python3 scripts/analyze_angular_displacement.py
    python3 scripts/analyze_angular_displacement.py --model rot_baseline_full_v1
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
SEED       = 42
ANGLES     = [15.0, 30.0, 45.0, 60.0]
TRANSITIONS = [(15.0, 30.0), (15.0, 45.0), (15.0, 60.0)]
TRANS_LABELS = ['15°→30°', '15°→45°', '15°→60°']


# ── Data ──────────────────────────────────────────────────────────────────────

def load_all_records() -> list:
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


class ChipDataset(Dataset):
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
        return self.transform(img), idx


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name, device):
    matches = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))
    if not matches:
        raise FileNotFoundError(f"No model found for '{model_name}'")
    exp_dir    = matches[-1]
    config     = json.loads((exp_dir / 'config.json').read_text())
    model_type = config.get('model_type', 'convnext_small')
    mod        = importlib.import_module(f'models.{model_type}')
    model      = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    features_out = {}
    def hook(module, input, output):
        features_out['vec'] = output.detach().cpu()
    model.classifier[1].register_forward_hook(hook)
    print(f"  Loaded: {exp_dir.name}  (val={config.get('best_val_acc','?')}%)")
    return model, features_out, mod.get_transform()


def extract_features(model, features_out, records, transform, device,
                     batch_size=64) -> np.ndarray:
    ds     = ChipDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    all_feats = [None] * len(records)
    with torch.no_grad():
        for imgs, indices in loader:
            _ = model(imgs.to(device))
            vecs = features_out['vec'].numpy()
            for i, idx in enumerate(indices.tolist()):
                all_feats[idx] = vecs[i]
    return np.array(all_feats)


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_centroids(records, features) -> dict:
    """Returns {(vehicle, angle): centroid_vector}."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        key = (r['target_type'], float(r['depression_angle']))
        groups[key].append(features[i])
    return {k: np.mean(v, axis=0) for k, v in groups.items()}


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def analyze_displacements(centroids: dict):
    vehicles = sorted(set(v for (v, _) in centroids.keys()))
    results  = {}

    for (a_from, a_to), label in zip(TRANSITIONS, TRANS_LABELS):
        deltas = []
        for v in vehicles:
            if (v, a_from) in centroids and (v, a_to) in centroids:
                delta = centroids[(v, a_to)] - centroids[(v, a_from)]
                deltas.append((v, delta))

        if len(deltas) < 2:
            continue

        vecs = np.array([d for _, d in deltas])
        mags = np.linalg.norm(vecs, axis=1)

        # Pairwise cosine similarities
        n = len(vecs)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        sims = np.array(sims)

        results[label] = {
            'vehicles':    [v for v, _ in deltas],
            'deltas':      vecs,
            'magnitudes':  mags,
            'cosine_sims': sims,
            'mean_cos':    float(np.mean(sims)),
            'std_cos':     float(np.std(sims)),
            'mean_mag':    float(np.mean(mags)),
            'std_mag':     float(np.std(mags)),
        }

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir: Path):
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    for col, (label, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[col])
        ax.hist(res['cosine_sims'], bins=30, color='steelblue', edgecolor='white')
        ax.axvline(res['mean_cos'], color='red', linestyle='--',
                   label=f"mean={res['mean_cos']:.3f}")
        ax.set_title(f'Displacement consistency\n{label}', fontsize=12)
        ax.set_xlabel('Cosine similarity between\nvehicle displacement vectors')
        ax.set_ylabel('Count' if col == 0 else '')
        ax.legend(fontsize=10)
        ax.set_xlim(-1, 1)

    plt.suptitle(
        'How consistent is the angular displacement across vehicles?\n'
        'Mean cosine similarity near 1.0 → analogy transfer will work  |  '
        'Near 0 → vehicle-specific shifts', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'angular_displacement_consistency.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_magnitude_by_vehicle(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, res) in zip(axes, results.items()):
        vehicles = res['vehicles']
        mags     = res['magnitudes']
        order    = np.argsort(mags)[::-1]
        ax.barh([vehicles[i] for i in order], [mags[i] for i in order],
                color='steelblue')
        ax.set_title(f'Displacement magnitude\n{label}', fontsize=11)
        ax.set_xlabel('L2 distance (15° centroid → target angle centroid)')
        ax.tick_params(axis='y', labelsize=7)
    plt.suptitle('Which vehicles shift most between angles?', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'angular_displacement_magnitude.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      default='rot_baseline_full_v1')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    out_dir = Path(__file__).parent.parent / 'logs' / 'angular_displacement'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model, features_out, transform = load_model(args.model, device)

    print("\nLoading all SOC records (all angles)...")
    records = load_all_records()
    print(f"  {len(records):,} records")

    print("\nExtracting features...")
    features = extract_features(model, features_out, records,
                                transform, device, args.batch_size)
    print(f"  Feature matrix: {features.shape}")

    print("\nComputing (vehicle, angle) centroids...")
    centroids = compute_centroids(records, features)
    print(f"  {len(centroids)} (vehicle, angle) centroids")

    print("\nAnalyzing displacement consistency...")
    results = analyze_displacements(centroids)

    print("\n" + "="*55)
    print("  Angular Displacement Consistency — SOC Feature Space")
    print("="*55)
    for label, res in results.items():
        print(f"\n  {label}")
        print(f"    Vehicles with both angles : {len(res['vehicles'])}")
        print(f"    Mean cosine similarity    : {res['mean_cos']:.4f}  "
              f"(std={res['std_cos']:.4f})")
        print(f"    Mean displacement mag     : {res['mean_mag']:.4f}  "
              f"(std={res['std_mag']:.4f})")
        verdict = ("CONSISTENT — analogy transfer likely works"
                   if res['mean_cos'] > 0.7 else
                   "MIXED — partial transfer possible"
                   if res['mean_cos'] > 0.3 else
                   "INCONSISTENT — vehicle-specific shifts dominate")
        print(f"    Verdict                   : {verdict}")

    # Save numeric results
    summary = {label: {k: v for k, v in res.items()
                       if k not in ('vehicles', 'deltas', 'cosine_sims',
                                    'magnitudes')}
               for label, res in results.items()}
    out_json = out_dir / 'displacement_summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved to {out_json}")

    print("\nPlotting...")
    plot_results(results, out_dir)
    plot_magnitude_by_vehicle(results, out_dir)

    print(f"\nDone. Results in logs/angular_displacement/")


if __name__ == '__main__':
    main()
