"""
scripts/umap_paper_figure.py
============================
Generates Figure 1 for the paper: SOC vs EOC backbone feature space.

Left panel  — SOC backbone (trained on all 4 angles):
  15° reference images (gray) + 30°/45°/60° overlay points land on clusters.

Right panel — EOC backbone (trained on 15° only):
  Same layout; 30°/45°/60° points scatter randomly.

Saves embeddings to .npy so reruns are instant (skip --recompute to reuse).

Output: logs/umap_paper_figure/soc_eoc_overlay.png  (300 dpi, 7.16" wide)

Usage:
    venv/bin/python3 scripts/umap_paper_figure.py
    venv/bin/python3 scripts/umap_paper_figure.py --recompute
    venv/bin/python3 scripts/umap_paper_figure.py --samples 50
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import umap as umap_lib
except ImportError:
    print("ERROR: umap-learn not installed.  pip install umap-learn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
OUT_DIR    = Path(__file__).parent.parent / 'logs' / 'umap_paper_figure'
SEED       = 42

SOC_MODEL  = 'rot_baseline_full_v1'
EOC_MODEL  = 'convnext_small_eoc_depression_flat_v1'

TRAIN_ANGLE   = 15.0
OVERLAY_ANGLES = [30.0, 45.0, 60.0]

# ColorBrewer Set1 — accessible on print
ANGLE_COLORS = {
    15.0: ('#bbbbbb', 'lightgray'),   # (scatter color, label color)
    30.0: ('#377eb8', '#377eb8'),
    45.0: ('#4daf4a', '#4daf4a'),
    60.0: ('#984ea3', '#984ea3'),
}


# ── DB ────────────────────────────────────────────────────────────────────────

def load_records(samples_per_class: int):
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

    rng = np.random.default_rng(SEED)
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r['target_type'], float(r['depression_angle']))].append(r)

    train, overlay = [], []
    for (veh, ang), recs in by_key.items():
        rng.shuffle(recs)
        subset = recs[:samples_per_class]
        if ang == TRAIN_ANGLE:
            train.extend(subset)
        elif ang in OVERLAY_ANGLES:
            overlay.extend(subset)

    print(f"  Train ({int(TRAIN_ANGLE)}°): {len(train):,}  "
          f"Overlay ({[int(a) for a in OVERLAY_ANGLES]}°): {len(overlay):,}")
    return train, overlay


# ── Dataset ───────────────────────────────────────────────────────────────────

class SARDataset(Dataset):
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


# ── Model loading + feature extraction ────────────────────────────────────────

def load_model(model_name: str, device):
    matches = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))
    if not matches:
        direct = MODELS_DIR / model_name
        matches = [direct] if direct.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"No model directory found for '{model_name}'")
    exp_dir = matches[-1]
    import json
    config  = json.loads((exp_dir / 'config.json').read_text())
    mod     = importlib.import_module(f'models.{config.get("model_type","convnext_small")}')
    model   = mod.build_model(num_classes=len(config['class_names'])).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded: {exp_dir.name}")
    return model, mod.get_transform()


def extract_features(model, records, transform, device, batch_size=64):
    buf = {}
    def hook(m, inp, out): buf['v'] = out.detach().cpu()
    handle = model.classifier[1].register_forward_hook(hook)

    loader = DataLoader(SARDataset(records, transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
            feats.append(buf['v'].numpy())
    handle.remove()
    return np.vstack(feats)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def fit_umap(features):
    print(f"  Fitting UMAP on {features.shape[0]:,} × {features.shape[1]}...")
    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    emb = reducer.fit_transform(features)
    print(f"  Done → {emb.shape}")
    return emb, reducer


def transform_umap(reducer, features):
    print(f"  Transforming {features.shape[0]:,} overlay points...")
    return reducer.transform(features)


# ── Per-model pipeline ────────────────────────────────────────────────────────

def run_model(model_name: str, train_recs, overlay_recs, device,
              cache_prefix: str, recompute: bool):
    train_emb_file   = OUT_DIR / f'{cache_prefix}_train_emb.npy'
    overlay_emb_file = OUT_DIR / f'{cache_prefix}_overlay_emb.npy'

    if not recompute and train_emb_file.exists() and overlay_emb_file.exists():
        print(f"  Loading cached embeddings for {model_name}...")
        return np.load(train_emb_file), np.load(overlay_emb_file)

    print(f"\nLoading model: {model_name}")
    model, transform = load_model(model_name, device)

    print("  Extracting train features...")
    train_feats = extract_features(model, train_recs, transform, device)

    print("  Extracting overlay features...")
    overlay_feats = extract_features(model, overlay_recs, transform, device)

    train_emb, reducer = fit_umap(train_feats)
    overlay_emb = transform_umap(reducer, overlay_feats)

    np.save(train_emb_file,   train_emb)
    np.save(overlay_emb_file, overlay_emb)
    print(f"  Cached: {cache_prefix}_{{train,overlay}}_emb.npy")
    return train_emb, overlay_emb


# ── Figure ────────────────────────────────────────────────────────────────────

def draw_panel(ax, train_emb, overlay_emb, train_recs, overlay_recs, label):
    # 15° base — light gray, small, very transparent
    ax.scatter(train_emb[:, 0], train_emb[:, 1],
               c='#cccccc', s=3, alpha=0.30, zorder=1, rasterized=True)

    # Overlay angles — colored, larger, more opaque
    for ang in OVERLAY_ANGLES:
        color = ANGLE_COLORS[ang][0]
        idx = np.where([float(r['depression_angle']) == ang
                        for r in overlay_recs])[0]
        if len(idx):
            ax.scatter(overlay_emb[idx, 0], overlay_emb[idx, 1],
                       c=color, s=7, alpha=0.65, zorder=2, rasterized=True,
                       label=f'{int(ang)}°')

    # Panel label inside top-left corner
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')
    ax.axis('off')


def build_figure(soc_train, soc_overlay, eoc_train, eoc_overlay,
                 train_recs, overlay_recs):
    # IEEE two-column full-width = 7.16"; height ~3.2" for a clean 2:1 ratio
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.3))

    draw_panel(axes[0], soc_train, soc_overlay, train_recs, overlay_recs, '(a)')
    draw_panel(axes[1], eoc_train, eoc_overlay, train_recs, overlay_recs, '(b)')

    # Shared legend — bottom center
    legend_handles = [
        mpatches.Patch(color='#cccccc', label='15° reference (train)'),
        mpatches.Patch(color=ANGLE_COLORS[30.0][0], label='30° (test)'),
        mpatches.Patch(color=ANGLE_COLORS[45.0][0], label='45° (test)'),
        mpatches.Patch(color=ANGLE_COLORS[60.0][0], label='60° (test)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.08,
                        wspace=0.08)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',   type=int, default=100,
                        help='Images per (vehicle, angle) for each set (default 100)')
    parser.add_argument('--recompute', action='store_true',
                        help='Ignore cached embeddings and recompute')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    print(f"\nLoading records ({args.samples} per vehicle per angle)...")
    train_recs, overlay_recs = load_records(args.samples)

    soc_train, soc_overlay = run_model(
        SOC_MODEL, train_recs, overlay_recs, device, 'soc', args.recompute)
    eoc_train, eoc_overlay = run_model(
        EOC_MODEL, train_recs, overlay_recs, device, 'eoc', args.recompute)

    print("\nBuilding figure...")
    fig = build_figure(soc_train, soc_overlay, eoc_train, eoc_overlay,
                       train_recs, overlay_recs)

    out = OUT_DIR / 'soc_eoc_overlay.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
