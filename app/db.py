"""
app/db.py
=========
All database queries for the Defense ML Viewer.

Schema: defense (separate from fungix/lichen)

Tables:
  defense.atrnet_images        -- image metadata (path, class hierarchy, scene, angles)
  defense.experiments          -- one row per training run
  defense.experiment_metrics   -- per-class precision/recall/F1 per split
  defense.experiment_predictions -- per-image predictions (for confusion matrix drill-down)
"""

import os
import psycopg2
import psycopg2.extras
from pathlib import Path


def _load_env():
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env()

DB_CONFIG = {
    'host':     os.environ.get('DB_HOST', 'localhost'),
    'port':     int(os.environ.get('DB_PORT', '5432')),
    'dbname':   os.environ.get('DB_NAME', 'fungidb'),   # reuse same Postgres server
    'user':     os.environ.get('DB_USER', ''),
    'password': os.environ.get('DB_PASSWORD', ''),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ── Image queries ─────────────────────────────────────────────────────────────

def get_atrnet_image(image_id):
    """Return metadata for one ATRNet image."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM defense.atrnet_images WHERE id = %s", (image_id,))
        row = cur.fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


# ── Experiment queries ────────────────────────────────────────────────────────

def get_experiments():
    """Return all experiments with val/test accuracy and overall metrics."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT e.id, e.name, e.model_type, e.config_name, e.quality_min,
                   e.epochs, e.notes, e.created_at,
                   MAX(m.accuracy)  FILTER (WHERE m.split='val'  AND m.class_name IS NULL) AS val_accuracy,
                   MAX(m.accuracy)  FILTER (WHERE m.split='test' AND m.class_name IS NULL) AS test_accuracy,
                   MAX(m.precision) FILTER (WHERE m.split='test' AND m.class_name IS NULL) AS test_precision,
                   MAX(m.recall)    FILTER (WHERE m.split='test' AND m.class_name IS NULL) AS test_recall,
                   MAX(m.f1)        FILTER (WHERE m.split='test' AND m.class_name IS NULL) AS test_f1,
                   MAX(m.support)   FILTER (WHERE m.split='test' AND m.class_name IS NULL) AS test_images,
                   ROUND((
                       COALESCE(SUM(m.support) FILTER (WHERE m.split='val'  AND m.class_name IS NOT NULL), 0) +
                       COALESCE(SUM(m.support) FILTER (WHERE m.split='test' AND m.class_name IS NOT NULL), 0)
                   ) / 0.30) AS total_images
            FROM defense.experiments e
            LEFT JOIN defense.experiment_metrics m ON m.experiment_id = e.id
            GROUP BY e.id
            ORDER BY e.id DESC
        """)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_experiment_detail(exp_id):
    """Return one experiment with confusion matrix and per-class metrics."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT * FROM defense.experiments WHERE id = %s", (exp_id,))
        exp = dict(cur.fetchone() or {})
        if not exp:
            return None

        cur.execute("""
            SELECT accuracy FROM defense.experiment_metrics
            WHERE experiment_id = %s AND split = 'test' AND class_name IS NULL
        """, (exp_id,))
        row = cur.fetchone()
        exp['test_accuracy'] = float(row['accuracy']) if row else None

        cur.execute("""
            SELECT class_name, precision, recall, f1, support
            FROM defense.experiment_metrics
            WHERE experiment_id = %s AND split = 'test'
            ORDER BY class_name NULLS LAST
        """, (exp_id,))
        exp['metrics'] = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT true_class, predicted_class, COUNT(*) AS n
            FROM defense.experiment_predictions
            WHERE experiment_id = %s AND split = 'test'
            GROUP BY true_class, predicted_class
            ORDER BY true_class, predicted_class
        """, (exp_id,))
        pred_rows = cur.fetchall()

        classes = sorted(set(r['true_class'] for r in pred_rows))
        idx = {c: i for i, c in enumerate(classes)}
        matrix = [[0] * len(classes) for _ in classes]
        for r in pred_rows:
            matrix[idx[r['true_class']]][idx[r['predicted_class']]] = r['n']

        exp['class_names']      = classes
        exp['confusion_matrix'] = matrix

    finally:
        conn.close()
    return exp


def get_experiment_images(exp_id, true_class, predicted_class, limit=50):
    """Return images for one confusion matrix cell."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT p.image_id, p.true_class, p.predicted_class, p.confidence,
                   i.target_type, i.subclass, i.main_class, i.scene,
                   i.depression_angle, i.azimuth_angle
            FROM defense.experiment_predictions p
            JOIN defense.atrnet_images i ON i.id = p.image_id
            WHERE p.experiment_id = %s
              AND p.true_class = %s
              AND p.predicted_class = %s
              AND p.split = 'test'
            ORDER BY p.confidence DESC
            LIMIT %s
        """, (exp_id, true_class, predicted_class, limit))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
