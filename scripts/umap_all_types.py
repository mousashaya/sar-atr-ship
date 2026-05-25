"""
scripts/umap_all_types.py
==========================
UMAP of ALL ship types in the test set, colored by ship type.
Uses dual-pol model (ship_dualpol_weighted_v1) features.

Rare types (n < 20) drawn as larger markers on top so they're visible.

Output: logs/ship_errors/umap_all_types.png
"""

import sys
import math
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

try:
    import tifffile
except ImportError:
    print("ERROR: pip install tifffile"); sys.exit(1)
try:
    import umap as umap_lib
except ImportError:
    print("ERROR: pip install umap-learn"); sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR     = Path(__file__).parent.parent / 'models' / 'saved'
OUT_DIR        = Path(__file__).parent.parent / 'logs' / 'ship_errors'
PATCH_CAL_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')
SEED           = 42
CLASSES_TRAIN  = ['Cargo', 'Tanker', 'Other Type']

DB_MIN, DB_MAX = -30.0, 10.0

# Color palette — 3 main classes bold, rare types muted
TYPE_STYLE = {
    'Cargo':           {'color': '#2563eb', 'marker': 'o', 'size': 12,  'alpha': 0.45, 'zorder': 2},
    'Tanker':          {'color': '#dc2626', 'marker': 'o', 'size': 12,  'alpha': 0.45, 'zorder': 2},
    'Other Type':      {'color': '#16a34a', 'marker': 'o', 'size': 12,  'alpha': 0.45, 'zorder': 2},
    'Fishing':         {'color': '#f59e0b', 'marker': 'D', 'size': 55,  'alpha': 0.85, 'zorder': 4},
    'Dredging':        {'color': '#8b5cf6', 'marker': 'D', 'size': 55,  'alpha': 0.85, 'zorder': 4},
    'Passenger':       {'color': '#ec4899', 'marker': 'D', 'size': 55,  'alpha': 0.85, 'zorder': 4},
    'Tug':             {'color': '#06b6d4', 'marker': '*', 'size': 120, 'alpha': 0.95, 'zorder': 5},
    'Pilot Vessel':    {'color': '#f97316', 'marker': '*', 'size': 120, 'alpha': 0.95, 'zorder': 5},
    'High speed craft':{'color': '#84cc16', 'marker': '*', 'size': 120, 'alpha': 0.95, 'zorder': 5},
    'Law Enforcement': {'color': '#fbbf24', 'marker': 'P', 'size': 150, 'alpha': 0.95, 'zorder': 6},
    'Search':          {'color': '#a78bfa', 'marker': 'P', 'size': 150, 'alpha': 0.95, 'zorder': 6},
    'Wing in ground':  {'color': '#34d399', 'marker': 'X', 'size': 180, 'alpha': 0.95, 'zorder': 6},
    'Port Tender':     {'color': '#fb923c', 'marker': 'X', 'size': 180, 'alpha': 0.95, 'zorder': 6},
    'Pleasure Craft':  {'color': '#f472b6', 'marker': 'X', 'size': 180, 'alpha': 0.95, 'zorder': 6},
    'Towing':          {'color': '#ffffff', 'marker': 'X', 'size': 200, 'alpha': 1.00, 'zorder': 7},
}
DEFAULT_STYLE = {'color': '#aaaaaa', 'marker': 's', 'size': 40, 'alpha': 0.7, 'zorder': 3}


def to_float_channel(arr_2d):
    db = 10.0 * np.log10(np.maximum(arr_2d, 1e-6))
    db = np.clip(db, DB_MIN, DB_MAX)
    return ((db - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


class AllTypesDataset(Dataset):
    def __init__(self, records):
        self.records   = records
        self.resize    = T.Resize((128, 128), antialias=True)
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            arr = tifffile.imread(str(rec['tif_path']))
            if arr.ndim == 2:
                arr = np.stack([arr, arr], axis=2)
            vv   = to_float_channel(arr[:, :, 0])
            vh   = to_float_channel(arr[:, :, 1])
            diff = np.clip(vv - vh + 0.5, 0.0, 1.0)
            tensor = torch.from_numpy(np.stack([vv, vh, diff], axis=0))
        except Exception:
            tensor = torch.zeros(3, 37, 37)
        tensor = self.resize(tensor)
        tensor = self.normalize(tensor)
        return tensor, idx


def load_records():
    """Load unique chips for all ship types in test set."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT ON (scene_id, ship_type,
                                SPLIT_PART(image_id, '_VV', 1))
                   scene_id, ship_type, image_id
            FROM defense.opensar_images
            WHERE polarization = 'VV'
              AND split = 'test'
              AND local_path IS NOT NULL
            ORDER BY scene_id, ship_type,
                     SPLIT_PART(image_id, '_VV', 1)
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    out = []
    for r in rows:
        parts = r['image_id'].split('__', 1)
        if len(parts) < 2:
            continue
        stem     = parts[1].rsplit('_VV', 1)[0]
        tif_path = PATCH_CAL_ROOT / r['scene_id'] / 'Patch_Cal' / f'{stem}.tif'
        if tif_path.exists():
            out.append({'ship_type': r['ship_type'], 'tif_path': tif_path})
    return out


def load_model(device):
    import importlib
    exp_dir = MODELS_DIR / 'ship_dualpol_weighted_v1'
    mod     = importlib.import_module('models.convnext_small')
    model   = mod.build_model(num_classes=len(CLASSES_TRAIN)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    return model


def extract_features(model, records, device):
    buf = {}
    def hook(m, inp, out): buf['v'] = out.detach().cpu()
    handle = model.classifier[1].register_forward_hook(hook)

    ds     = AllTypesDataset(records)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    feats  = np.zeros((len(records), 768), dtype=np.float32)

    with torch.no_grad():
        for imgs, idxs in loader:
            model(imgs.to(device))
            feats[idxs.numpy()] = buf['v'].numpy()

    handle.remove()
    return feats


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("Loading test records (all ship types)...")
    records = load_records()
    types   = [r['ship_type'] for r in records]
    counts  = {}
    for t in types: counts[t] = counts.get(t, 0) + 1
    print(f"  {len(records):,} chips across {len(counts)} ship types:")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<22}: {n:>4}")

    print("\nLoading model + extracting features...")
    model  = load_model(device)
    feats  = extract_features(model, records, device)

    print(f"\nFitting UMAP on {feats.shape[0]:,} × {feats.shape[1]}...")
    reducer   = umap_lib.UMAP(n_neighbors=20, min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(feats)
    print("Done.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Draw in z-order: common types first, rare on top
    all_types_ordered = sorted(counts.keys(), key=lambda t: -counts[t])

    legend_handles = []
    for ship_type in all_types_ordered:
        idx   = [i for i, t in enumerate(types) if t == ship_type]
        style = TYPE_STYLE.get(ship_type, DEFAULT_STYLE)
        ax.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=style['color'], marker=style['marker'],
            s=style['size'], alpha=style['alpha'],
            zorder=style['zorder'],
            edgecolors='none' if style['size'] < 50 else '#222',
            linewidths=0.5,
        )
        legend_handles.append(mpatches.Patch(
            color=style['color'],
            label=f"{ship_type}  (n={counts[ship_type]})"
        ))

    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.15,
              facecolor='#1a1a1a', edgecolor='#444',
              labelcolor='white', loc='upper right',
              ncol=2 if len(counts) > 8 else 1)

    ax.set_title('Feature space — all ship types (test set)\n'
                 'dual-pol ConvNeXt-Small, 768-dim UMAP projection',
                 fontsize=11, color='white', pad=10)
    ax.axis('off')
    fig.tight_layout()

    out = OUT_DIR / 'umap_all_types.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
