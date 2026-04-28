"""
scripts/angle_detection.py
===========================
Tests whether depression angle can be predicted from pixel histograms alone.

If angle is predictable from the image, it opens the door to a hierarchical
pipeline: predict angle → route to angle-specific classifier → predict vehicle.

Features: 64-bin intensity histogram (same as histogram_clustering.py)
Label: depression_angle (15, 30, 45, 60)

Usage:
    python3 scripts/angle_detection.py
"""

import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'angle_detection'
N_BINS     = 64
MAX_IMAGES = 500   # per angle — keeps runtime fast


def compute_histogram(image_path: str) -> np.ndarray:
    try:
        img  = Image.open(image_path).convert('L')
        arr  = np.array(img, dtype=np.float32)
        hist, _ = np.histogram(arr.flatten(), bins=N_BINS, range=(0, 255))
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist
    except Exception:
        return np.zeros(N_BINS, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT local_path, depression_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle IS NOT NULL
              AND local_path IS NOT NULL
            ORDER BY depression_angle, RANDOM()
        """)
        rows = cur.fetchall()
    conn.close()

    # Cap per angle
    from collections import defaultdict
    by_angle = defaultdict(list)
    for r in rows:
        by_angle[r['depression_angle']].append(r['local_path'])

    print(f"Images per angle (before cap):")
    for angle in sorted(by_angle.keys()):
        print(f"  {angle}°: {len(by_angle[angle]):,}")

    sampled_paths  = []
    sampled_angles = []
    for angle in sorted(by_angle.keys()):
        paths = by_angle[angle][:MAX_IMAGES]
        sampled_paths.extend(paths)
        sampled_angles.extend([angle] * len(paths))

    print(f"\nLoading {len(sampled_paths):,} images and computing histograms...")
    histograms = []
    skipped    = 0
    for path in sampled_paths:
        h = compute_histogram(path)
        if h.sum() == 0:
            skipped += 1
            histograms.append(np.zeros(N_BINS))
        else:
            histograms.append(h)

    X = np.array(histograms)
    y = np.array(sampled_angles)
    angles = sorted(set(y))
    print(f"Skipped: {skipped}  |  Final: {len(X):,} histograms  {len(angles)} angles")

    # ── kNN accuracy ──────────────────────────────────────────────────────────
    print(f"\nComputing kNN accuracy (5-fold)...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import silhouette_score

    knn    = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    knn_acc = scores.mean()
    print(f"kNN 5-fold accuracy: {knn_acc*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"Random chance:       {100/len(angles):.1f}%")

    sil = silhouette_score(X, y)
    print(f"Silhouette score:    {sil:.3f}")

    # ── Per-angle histogram statistics ────────────────────────────────────────
    print(f"\nPer-angle mean brightness and std:")
    print(f"  {'Angle':>8}  {'Mean brightness':>16}  {'Std':>8}  {'N':>6}")
    for angle in angles:
        mask = y == angle
        brightness = X[mask].mean(axis=1) * 255  # approximate mean pixel value
        print(f"  {angle:>7}°  {brightness.mean():>16.1f}  {brightness.std():>8.2f}  {mask.sum():>6,}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print(f"\nRunning UMAP...")
    try:
        import umap
        reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors    = cm.tab10(np.linspace(0, 1, len(angles)))
        color_map = {a: colors[i] for i, a in enumerate(angles)}
        for angle in angles:
            mask = y == angle
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color_map[angle]], label=f'{angle}°', alpha=0.6, s=10)
        ax.set_title(f'Histogram UMAP — colored by Depression Angle\n'
                     f'kNN: {knn_acc*100:.1f}%  Silhouette: {sil:.3f}', fontsize=13)
        ax.legend(fontsize=12)
        plt.tight_layout()
        out_path = OUTPUT_DIR / 'angle_detection_umap.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved: {out_path.name}")
    except ImportError:
        print("umap-learn not installed — skipping plot")

    print(f"\nConclusion:")
    if knn_acc > 0.85:
        print(f"  {knn_acc*100:.1f}% — angle is HIGHLY predictable from histogram alone")
        print(f"  Hierarchical pipeline (angle → vehicle) is very promising")
    elif knn_acc > 0.60:
        print(f"  {knn_acc*100:.1f}% — angle is MODERATELY predictable")
        print(f"  A ConvNeXt angle classifier would likely do much better")
    else:
        print(f"  {knn_acc*100:.1f}% — histogram alone insufficient")
        print(f"  Need image-based model for angle detection")


if __name__ == '__main__':
    main()
