"""
scripts/feature_analysis.py
============================
Extracts feature vectors from a trained angle-specific classifier and
visualizes them with UMAP to diagnose sparse/confused classes.

Answers: are the sparse classes scattered, overlapping, or just small?
This tells us whether feature augmentation is worth pursuing.

Usage:
    python3 scripts/feature_analysis.py --angle 15
    python3 scripts/feature_analysis.py --angle 15 --highlight "Jeep_Patriot,Chery_Arrizo 5"
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
ANGLE_CLASSES = [15.0, 30.0, 45.0, 60.0]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ChipDataset(Dataset):
    def __init__(self, records, transform=None):
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
        if self.transform:
            img = self.transform(img)
        return img, rec['target_type'], rec['image_id']


def load_soc_images(angle: float, split: str = 'test'):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND split = %s
              AND depression_angle = %s
              AND target_type IS NOT NULL
              AND local_path IS NOT NULL
        """, (split, angle))
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def extract_features(model, loader, device):
    """Extract penultimate layer features (before final classifier head)."""
    features_list, labels_list = [], []

    # Hook to capture features before the final linear layer
    captured = {}
    def hook(module, input, output):
        captured['features'] = input[0].detach().cpu()

    # Register hook on the final linear layer
    final_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            final_layer = module
    if final_layer is None:
        raise ValueError("Could not find final linear layer")
    handle = final_layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            model(imgs)
            features_list.append(captured['features'].numpy())
            labels_list.extend(labels)

    handle.remove()
    return np.vstack(features_list), labels_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle',     type=float, default=15.0,
                        help='Depression angle to analyze (default: 15)')
    parser.add_argument('--highlight', default='',
                        help='Comma-separated class names to highlight')
    parser.add_argument('--split',     default='test',
                        choices=['train', 'test'])
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    highlight_classes = [c.strip() for c in args.highlight.split(',') if c.strip()]

    device = get_device()
    print(f"Device: {device}")

    # Load model
    model_module = importlib.import_module('models.convnext')
    transform    = model_module.get_transform()

    pattern  = f'convnext_40class_angle{int(args.angle)}_v1'
    matches  = [d for d in MODELS_DIR.iterdir() if d.is_dir() and pattern in d.name]
    if not matches:
        print(f"No classifier found for angle {args.angle}°. Run train.py first.")
        return
    save_dir   = sorted(matches)[-1]
    model_cfg  = json.loads((save_dir / 'config.json').read_text())
    class_names = model_cfg['class_names']
    n_cls       = len(class_names)

    print(f"Loading classifier: {save_dir.name}  ({n_cls} classes)")
    model = model_module.build_model(num_classes=n_cls).to(device)
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))

    # Load data
    print(f"Loading SOC {args.split} images at {args.angle}°...")
    records = load_soc_images(args.angle, args.split)
    print(f"  {len(records):,} images")

    dataset = ChipDataset(records, transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, persistent_workers=True)

    # Extract features
    print("Extracting features...")
    features, labels = extract_features(model, loader, device)
    print(f"  Feature matrix: {features.shape}")

    # Per-class stats
    print(f"\nPer-class feature space analysis:")
    print(f"{'Class':<35} {'N':>6} {'Intra-dist':>12} {'Sparse?':>8}")
    print('-' * 65)

    by_class = defaultdict(list)
    for feat, label in zip(features, labels):
        by_class[label].append(feat)

    class_stats = {}
    for cls in sorted(by_class.keys()):
        vecs = np.array(by_class[cls])
        n    = len(vecs)
        # Mean intra-class distance (how spread out is this class?)
        if n > 1:
            center = vecs.mean(axis=0)
            dists  = np.linalg.norm(vecs - center, axis=1)
            intra  = dists.mean()
        else:
            intra = 0.0
        sparse = '<<< SPARSE' if n < 150 else ''
        class_stats[cls] = {'n': n, 'intra': intra, 'center': vecs.mean(axis=0)}
        print(f"{cls:<35} {n:>6} {intra:>12.3f} {sparse}")

    # Inter-class distances for sparse classes
    sparse_classes = [cls for cls, s in class_stats.items() if s['n'] < 150]
    if sparse_classes or highlight_classes:
        focus = list(set(sparse_classes + highlight_classes))
        print(f"\nNearest neighbors for sparse/highlighted classes:")
        print("(closest other classes in feature space = likely confusion partners)\n")
        all_classes = list(class_stats.keys())
        for cls in focus:
            if cls not in class_stats:
                print(f"  {cls}: not found in data")
                continue
            center = class_stats[cls]['center']
            dists  = {c: np.linalg.norm(center - class_stats[c]['center'])
                      for c in all_classes if c != cls}
            nearest = sorted(dists.items(), key=lambda x: x[1])[:5]
            print(f"  {cls} (n={class_stats[cls]['n']}):")
            for neighbor, dist in nearest:
                print(f"    → {neighbor:<35} dist={dist:.3f}  (n={class_stats[neighbor]['n']})")

    # UMAP visualization
    try:
        import umap
        print(f"\nGenerating UMAP visualization...")
        reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        embedded = reducer.fit_transform(features)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: all classes
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}

        ax = axes[0]
        for cls in unique_labels:
            mask = [l == cls for l in labels]
            pts  = embedded[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5,
                      color=cmap(label_to_idx[cls]), label=cls)
        ax.set_title(f'All 40 classes — {args.angle}° ({args.split})')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        # Plot 2: highlight sparse/focus classes
        ax2 = axes[1]
        focus_set = set(sparse_classes + highlight_classes)
        # Background: all non-focus classes in gray
        mask_bg = [l not in focus_set for l in labels]
        ax2.scatter(embedded[mask_bg, 0], embedded[mask_bg, 1],
                   s=3, alpha=0.2, color='lightgray', label='other')
        # Foreground: focus classes in color
        colors = plt.cm.tab10(np.linspace(0, 1, len(focus_set)))
        for i, cls in enumerate(sorted(focus_set)):
            mask = [l == cls for l in labels]
            pts  = embedded[mask]
            if pts.shape[0] == 0:
                continue
            ax2.scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.8,
                       color=colors[i], label=f'{cls} (n={sum(mask)})')
        ax2.set_title(f'Sparse/highlighted classes — {args.angle}°')
        ax2.set_xlabel('UMAP 1')
        ax2.legend(fontsize=7, markerscale=2)

        out_path = Path(__file__).parent.parent / 'logs' / f'feature_umap_angle{int(args.angle)}.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
        plt.close()

    except ImportError:
        print("\nUMAP not installed — skipping visualization.")
        print("Install with: pip install umap-learn matplotlib")


if __name__ == '__main__':
    main()
