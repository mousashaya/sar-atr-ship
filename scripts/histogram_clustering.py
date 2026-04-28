"""
scripts/histogram_clustering.py
================================
Analyzes whether pixel intensity histograms alone can separate vehicle classes.
Runs independently per depression angle — no mixing of angles.

For each angle:
  1. Compute 64-bin intensity histogram for every image
  2. Run UMAP to reduce to 2D
  3. Save scatter plot colored by main_class and by target_type
  4. Compute silhouette score and kNN accuracy as quantitative metrics

Results saved to: data/histogram_analysis/

Usage:
    python3 scripts/histogram_clustering.py
    python3 scripts/histogram_clustering.py --angle 45    # one angle only
    python3 scripts/histogram_clustering.py --config 4class --angle 45
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'histogram_analysis'
N_BINS     = 64    # histogram bins — enough detail without too much noise
MAX_IMAGES = 500   # max images per class per angle — keeps runtime reasonable


def compute_histogram(image_path: str) -> np.ndarray:
    """Load image and compute normalized 64-bin intensity histogram."""
    try:
        img  = Image.open(image_path).convert('L')  # greyscale
        arr  = np.array(img, dtype=np.float32)
        hist, _ = np.histogram(arr.flatten(), bins=N_BINS, range=(0, 255))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total  # normalize to sum to 1
        return hist
    except Exception:
        return np.zeros(N_BINS, dtype=np.float32)


def load_images_for_angle(conn, angle: float, config: str) -> tuple:
    """
    Load image paths and class labels for one depression angle.
    config: '4class' uses main_class, '21class' uses subclass, '40class' uses target_type
    """
    class_field = {'4class': 'main_class', '21class': 'subclass',
                   '40class': 'target_type'}[config]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT local_path, {class_field} AS label, main_class, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = %s
              AND {class_field} IS NOT NULL
              AND local_path IS NOT NULL
            ORDER BY {class_field}, id
        """, (angle,))
        rows = cur.fetchall()

    # Cap per class to MAX_IMAGES
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['label']].append(r)

    sampled = []
    rng = np.random.default_rng(42)
    for cls, items in by_class.items():
        if len(items) > MAX_IMAGES:
            idxs = rng.choice(len(items), MAX_IMAGES, replace=False)
            items = [items[i] for i in idxs]
        sampled.extend(items)

    return sampled


def analyze_angle(conn, angle: float, config: str, results: dict):
    """Run full histogram clustering analysis for one angle."""
    print(f"\n  Depression angle: {angle}°")

    rows = load_images_for_angle(conn, angle, config)
    if not rows:
        print(f"    No images found.")
        return

    print(f"    Loading {len(rows):,} images and computing histograms...")
    histograms = []
    labels     = []
    skipped    = 0

    for r in rows:
        hist = compute_histogram(r['local_path'])
        if hist.sum() == 0:
            skipped += 1
            continue
        histograms.append(hist)
        labels.append(r['label'])

    if skipped:
        print(f"    Skipped {skipped} unreadable images.")

    X = np.array(histograms)
    y = np.array(labels)
    classes = sorted(set(y))
    print(f"    {len(X):,} histograms  {len(classes)} classes")

    # ── kNN accuracy ──────────────────────────────────────────────────────────
    print(f"    Computing kNN accuracy...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    le  = LabelEncoder()
    y_enc = le.fit_transform(y)
    knn = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y_enc, cv=5, scoring='accuracy')
    knn_acc = scores.mean()
    print(f"    kNN 5-fold accuracy: {knn_acc*100:.1f}% ± {scores.std()*100:.1f}%")

    # ── Silhouette score ──────────────────────────────────────────────────────
    print(f"    Computing silhouette score...")
    from sklearn.metrics import silhouette_score
    # Sample for speed if large
    if len(X) > 2000:
        idx = np.random.default_rng(42).choice(len(X), 2000, replace=False)
        sil = silhouette_score(X[idx], y_enc[idx])
    else:
        sil = silhouette_score(X, y_enc)
    print(f"    Silhouette score: {sil:.3f}  (-1=random, 0=overlap, 1=perfect)")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print(f"    Running UMAP...")
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X)
    except ImportError:
        print(f"    UMAP not installed — skipping plot. Run: venv/bin/pip install umap-learn")
        embedding = None

    # ── Save plot ─────────────────────────────────────────────────────────────
    if embedding is not None:
        fig, ax = plt.subplots(figsize=(12, 9))
        colors = cm.tab20(np.linspace(0, 1, len(classes)))
        color_map = {c: colors[i] for i, c in enumerate(classes)}

        for cls in classes:
            mask = y == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color_map[cls]], label=cls, alpha=0.5, s=8)

        ax.set_title(f'Histogram UMAP — {config} — Depression {angle}°\n'
                     f'kNN acc: {knn_acc*100:.1f}%  Silhouette: {sil:.3f}',
                     fontsize=13)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()

        out_path = OUTPUT_DIR / f'umap_{config}_angle{int(angle)}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Plot saved: {out_path.name}")

    # Store results
    results[f'angle_{int(angle)}'] = {
        'angle':        angle,
        'n_images':     len(X),
        'n_classes':    len(classes),
        'knn_accuracy': round(float(knn_acc), 4),
        'knn_std':      round(float(scores.std()), 4),
        'silhouette':   round(float(sil), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--angle',  type=float, default=None,
                        help='Run only this depression angle (default: all)')
    parser.add_argument('--config', default='4class',
                        choices=['4class', '21class', '40class'],
                        help='Classification level (default: 4class)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import sklearn
    except ImportError:
        print("ERROR: pip install scikit-learn umap-learn matplotlib")
        sys.exit(1)

    conn = psycopg2.connect(**DB_CONFIG)

    # Get available angles
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle IS NOT NULL
            ORDER BY 1
        """)
        all_angles = [r[0] for r in cur.fetchall()]

    angles = [args.angle] if args.angle else all_angles
    print(f"Config: {args.config}")
    print(f"Angles to analyze: {angles}")
    print(f"Max images per class per angle: {MAX_IMAGES}")
    print(f"Histogram bins: {N_BINS}")

    results = {}
    for angle in angles:
        analyze_angle(conn, angle, args.config, results)

    conn.close()

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY — {args.config}")
    print(f"{'─'*60}")
    print(f"{'Angle':>8}  {'kNN Acc':>10}  {'Silhouette':>12}  {'N Images':>10}")
    for key, r in sorted(results.items()):
        print(f"  {r['angle']:>5}°  "
              f"{r['knn_accuracy']*100:>8.1f}%  "
              f"{r['silhouette']:>12.3f}  "
              f"{r['n_images']:>10,}")

    # Save JSON summary
    out_json = OUTPUT_DIR / f'summary_{args.config}.json'
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved: {out_json}")
    print(f"Plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
