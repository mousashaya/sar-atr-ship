"""
scripts/bbox_clustering.py
==========================
Analyzes whether bounding box geometry alone can separate vehicle classes.
No image loading needed — all features come directly from the DB.

Features computed per image:
  - aspect_ratio  : width / height of bounding box
  - bbox_area     : width * height (proxy for vehicle size)
  - center_x      : horizontal center of bbox normalized to chip width (0-1)
  - center_y      : vertical center of bbox normalized to chip height (0-1)
  - relative_size : bbox_area / chip_area (how much of chip the vehicle fills)

Run per depression angle independently (no mixing).

Usage:
    python3 scripts/bbox_clustering.py                  # all angles, 4class
    python3 scripts/bbox_clustering.py --angle 45       # one angle only
    python3 scripts/bbox_clustering.py --config 40class # fine-grained
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'bbox_analysis'
CHIP_SIZE  = 128  # ATRNet chips are 128x128


def get_features(conn, angle: float, config: str) -> tuple:
    """
    Pull bbox geometry from DB for one angle.
    Returns (feature_matrix, labels, class_list).
    """
    class_field = {'4class': 'main_class', '21class': 'subclass',
                   '40class': 'target_type'}[config]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT {class_field} AS label,
                   bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND depression_angle = %s
              AND {class_field} IS NOT NULL
              AND bbox_xmin IS NOT NULL
              AND bbox_xmax IS NOT NULL
              AND bbox_ymin IS NOT NULL
              AND bbox_ymax IS NOT NULL
        """, (angle,))
        rows = cur.fetchall()

    if not rows:
        return None, None, None

    features = []
    labels   = []

    for r in rows:
        w = r['bbox_xmax'] - r['bbox_xmin']
        h = r['bbox_ymax'] - r['bbox_ymin']

        if w <= 0 or h <= 0:
            continue

        aspect_ratio  = w / h
        bbox_area     = w * h
        center_x      = (r['bbox_xmin'] + r['bbox_xmax']) / 2 / CHIP_SIZE
        center_y      = (r['bbox_ymin'] + r['bbox_ymax']) / 2 / CHIP_SIZE
        relative_size = bbox_area / (CHIP_SIZE * CHIP_SIZE)

        features.append([aspect_ratio, bbox_area, center_x, center_y, relative_size])
        labels.append(r['label'])

    return np.array(features, dtype=np.float32), np.array(labels), sorted(set(labels))


def analyze_angle(conn, angle: float, config: str, results: dict):
    """Full bbox geometry analysis for one depression angle."""
    print(f"\n  Depression angle: {angle}°")

    X, y, classes = get_features(conn, angle, config)
    if X is None or len(X) == 0:
        print(f"    No data found.")
        return

    print(f"    {len(X):,} images  {len(classes)} classes")
    print(f"    Features: aspect_ratio, bbox_area, center_x, center_y, relative_size")

    # ── Per-class statistics ──────────────────────────────────────────────────
    print(f"\n    Per-class bbox statistics:")
    print(f"    {'Class':<30} {'Aspect Ratio':>14} {'Relative Size':>14} {'N':>6}")
    print(f"    {'─'*65}")
    for cls in classes:
        mask = y == cls
        ar   = X[mask, 0]
        rs   = X[mask, 4]
        print(f"    {cls:<30} {ar.mean():>8.2f}±{ar.std():>4.2f}   "
              f"{rs.mean()*100:>8.1f}%±{rs.std()*100:>4.1f}%   {mask.sum():>6,}")

    # ── Normalize features for kNN and UMAP ──────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import silhouette_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── kNN accuracy ──────────────────────────────────────────────────────────
    print(f"\n    Computing kNN accuracy...")
    knn    = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn, X_scaled, y_enc, cv=5, scoring='accuracy')
    knn_acc = scores.mean()
    print(f"    kNN 5-fold accuracy: {knn_acc*100:.1f}% ± {scores.std()*100:.1f}%")

    # ── Silhouette score ──────────────────────────────────────────────────────
    if len(X_scaled) > 2000:
        idx = np.random.default_rng(42).choice(len(X_scaled), 2000, replace=False)
        sil = silhouette_score(X_scaled[idx], y_enc[idx])
    else:
        sil = silhouette_score(X_scaled, y_enc)
    print(f"    Silhouette score:    {sil:.3f}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print(f"    Running UMAP...")
    try:
        import umap
        reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        # Sample for speed if large
        if len(X_scaled) > 5000:
            idx = np.random.default_rng(42).choice(len(X_scaled), 5000, replace=False)
            X_plot = X_scaled[idx]
            y_plot = y[idx]
        else:
            X_plot = X_scaled
            y_plot = y
        embedding = reducer.fit_transform(X_plot)
    except ImportError:
        print(f"    umap-learn not installed — skipping UMAP plot")
        embedding = None
        y_plot    = y

    # ── Plot ──────────────────────────────────────────────────────────────────
    if embedding is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        colors    = cm.tab20(np.linspace(0, 1, len(classes)))
        color_map = {c: colors[i] for i, c in enumerate(classes)}

        # UMAP scatter
        ax = axes[0]
        for cls in classes:
            mask = y_plot == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color_map[cls]], label=cls, alpha=0.5, s=8)
        ax.set_title(f'UMAP — Bbox Geometry — {config} — {angle}°\n'
                     f'kNN: {knn_acc*100:.1f}%  Silhouette: {sil:.3f}', fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        # Aspect ratio vs relative size scatter
        ax2 = axes[1]
        sample_idx = np.random.default_rng(42).choice(len(X), min(3000, len(X)), replace=False)
        for cls in classes:
            mask = y[sample_idx] == cls
            ax2.scatter(X[sample_idx][mask, 0],      # aspect ratio
                        X[sample_idx][mask, 4] * 100, # relative size %
                        c=[color_map[cls]], label=cls, alpha=0.4, s=8)
        ax2.set_xlabel('Aspect Ratio (width/height)', fontsize=11)
        ax2.set_ylabel('Relative Size (% of chip)', fontsize=11)
        ax2.set_title(f'Aspect Ratio vs Vehicle Size — {angle}°', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        out_path = OUTPUT_DIR / f'bbox_{config}_angle{int(angle)}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Plot saved: {out_path.name}")

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
    parser.add_argument('--angle',  type=float, default=None)
    parser.add_argument('--config', default='4class',
                        choices=['4class', '21class', '40class'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(**DB_CONFIG)

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

    print(f"Bbox geometry clustering analysis")
    print(f"Config: {args.config}  |  Angles: {angles}")
    print(f"Features: aspect_ratio, bbox_area, center_x, center_y, relative_size")

    results = {}
    for angle in angles:
        analyze_angle(conn, angle, args.config, results)

    conn.close()

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"SUMMARY — bbox geometry — {args.config}")
    print(f"{'─'*55}")
    print(f"{'Angle':>8}  {'kNN Acc':>10}  {'Silhouette':>12}  {'N':>8}")
    for key, r in sorted(results.items()):
        print(f"  {r['angle']:>5}°  "
              f"{r['knn_accuracy']*100:>8.1f}%  "
              f"{r['silhouette']:>12.3f}  "
              f"{r['n_images']:>8,}")

    # Compare to histogram results if available
    hist_path = OUTPUT_DIR.parent / 'histogram_analysis' / f'summary_{args.config}.json'
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        print(f"\n{'─'*55}")
        print(f"COMPARISON vs Histogram")
        print(f"{'─'*55}")
        print(f"{'Angle':>8}  {'Hist kNN':>10}  {'Bbox kNN':>10}  {'Delta':>8}")
        for key, r in sorted(results.items()):
            hist_r = hist.get(key, {})
            hist_acc = hist_r.get('knn_accuracy', 0)
            delta = r['knn_accuracy'] - hist_acc
            print(f"  {r['angle']:>5}°  "
                  f"{hist_acc*100:>8.1f}%  "
                  f"{r['knn_accuracy']*100:>8.1f}%  "
                  f"{delta*100:>+7.1f}%")

    out_json = OUTPUT_DIR / f'summary_{args.config}.json'
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_json}")


if __name__ == '__main__':
    main()
