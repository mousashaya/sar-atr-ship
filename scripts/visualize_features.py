"""
scripts/visualize_features.py
==============================
UMAP visualization of ConvNeXt feature space.

Extracts the 768-dimensional feature vector from the layer before the
classification head, reduces to 2D with UMAP, and plots three views:

  View 1: colored by 4 main classes (Bus/Car/Truck/Special)
  View 2: colored by depression angle (15/30/45/60)
  View 3: colored by depression angle for ONE vehicle at a time

Reveals how the model organizes its internal representations — whether
class structure or angle structure dominates the feature space.

Usage:
    python3 scripts/visualize_features.py
    python3 scripts/visualize_features.py --samples-per-class 100
    python3 scripts/visualize_features.py --model rot_baseline_full_v1
    python3 scripts/visualize_features.py --vehicle "Buick_GL8"
    python3 scripts/visualize_features.py --samples-per-class 50 --vehicle "FAW_J6P"
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import umap
except ImportError:
    print("ERROR: umap-learn not installed. Run: pip install umap-learn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
SEED       = 42

MAIN_CLASS_MAP = {
    'Buick_Excelle_GT': 'Car',    'Buick_GL8': 'Car',
    "Chang'an_CS75_Plus": 'Car',  "Chang'an_Starlight_4500": 'Car',
    'Chery_Arrizo 5': 'Car',      'Chery_qq3': 'Car',
    'Chevrolet_Blazer_1998': 'Car','Great_Wall_Voleex_C50': 'Car',
    'Great_Wall_poer': 'Car',     'Hawtai_EV160B': 'Car',
    'Hongqi_CA7180A3E': 'Car',    'Hongqi_h5': 'Car',
    'Jeep_Patriot': 'Car',        'Lincoln_MKC': 'Car',
    'Mitsubishi_Outlander_2003': 'Car',
    'Dongfeng_Forthing_Lingzhi': 'Bus', 'Dongfeng_EQ6608LTV': 'Bus',
    'FAW_Jiabao_T51': 'Bus',      'JINBEI_SY5033XJH': 'Bus',
    'MAXUS_V80': 'Bus',           'Wuling_Rongguang_V': 'Bus',
    'Yangzi_YZK6590XCA': 'Bus',   'Yutong_ZK6120HY1': 'Bus',
    'CNHTC_HOWO': 'Truck',        'Changlin_8228-5': 'Truck',
    'Dongfeng_Tianjin_DFH2200B': 'Truck','Dongfeng_Tianjin_KR230': 'Truck',
    'Dongfeng_Duolika': 'Truck',  'FAW_J6P': 'Truck',
    'Foton_BJ1045V9JB5-54': 'Truck','Huanghai_N1': 'Truck',
    'JAC_Junling': 'Truck',       'Lveco_Proud_2009': 'Truck',
    'SHACMAN_DeLong_M3000': 'Truck','SHACMAN_DeLong_X3000': 'Truck',
    'WAW_Aochi_1800': 'Truck',    'WAW_Aochi_Hongrui': 'Truck',
    'Changfeng_Cheetah_CFA6473C': 'Special','Hyundai_HLF25_II': 'Special',
    'SDLG_ZL40F': 'Special',      'WAW_Aochi_Hongrui': 'Truck',
}

ANGLE_COLORS  = {15.0: '#e41a1c', 30.0: '#377eb8', 45.0: '#4daf4a', 60.0: '#984ea3'}
CLASS4_COLORS = {'Car': '#e41a1c', 'Bus': '#377eb8', 'Truck': '#4daf4a', 'Special': '#984ea3'}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(samples_per_class: int, vehicle_filter: str = None,
                 train_angle: float = None, overlay_angles: list = None) -> tuple:
    """
    Returns (train_records, overlay_records).

    train_angle:    if set, only load images at this depression angle as 'train'
    overlay_angles: if set, also load these angles separately as 'overlay'
                    (used for Plot 4/5 — show where unlabeled angles land)
    If neither is set, loads all angles and overlay_records = []
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        sql = """
            SELECT id, local_path, target_type, main_class,
                   depression_angle, band, polarization, azimuth_angle
            FROM defense.atrnet_images
            WHERE experiment_setting = 'SOC_40classes'
              AND local_path IS NOT NULL
              AND target_type IS NOT NULL AND target_type != ''
              AND depression_angle IS NOT NULL
        """
        params = []
        if vehicle_filter:
            sql += " AND target_type = %s"
            params.append(vehicle_filter)
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    rng = np.random.default_rng(SEED)
    from collections import defaultdict

    def sample(records, n):
        by_class_angle = defaultdict(list)
        for r in records:
            key = (r['target_type'], float(r['depression_angle']))
            by_class_angle[key].append(r)
        out = []
        for key, recs in by_class_angle.items():
            rng.shuffle(recs)
            out.extend(recs[:n])
        return out

    if train_angle is not None:
        train_rows   = [r for r in rows if float(r['depression_angle']) == train_angle]
        overlay_rows = [r for r in rows if float(r['depression_angle']) in (overlay_angles or [])]
        train_recs   = sample(train_rows,   samples_per_class)
        overlay_recs = sample(overlay_rows, samples_per_class)
        print(f"  Train angle {int(train_angle)}°: {len(train_recs):,} images")
        if overlay_recs:
            print(f"  Overlay angles {overlay_angles}: {len(overlay_recs):,} images")
        return train_recs, overlay_recs
    else:
        all_recs = sample(rows, samples_per_class)
        print(f"  Loaded {len(all_recs):,} images (all angles)")
        return all_recs, []


# ── Dataset ───────────────────────────────────────────────────────────────────

class FeatureDataset(Dataset):
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


# ── Model + feature extraction ────────────────────────────────────────────────

def load_model_and_extractor(model_name: str, device):
    matches = sorted(MODELS_DIR.glob(f'exp_*_{model_name}'))
    if not matches:
        # Fall back to direct directory name match
        direct = MODELS_DIR / model_name
        if direct.is_dir():
            matches = [direct]
    if not matches:
        raise FileNotFoundError(f"No model found for '{model_name}'")
    exp_dir     = matches[-1]
    config      = json.loads((exp_dir / 'config.json').read_text())
    model_type  = config.get('model_type', 'convnext_small')
    class_names = config['class_names']
    mod         = importlib.import_module(f'models.{model_type}')
    model       = mod.build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded: {exp_dir.name}  (val={config.get('best_val_acc','?')}%)")

    # Hook to extract 768-dim features before the classification head
    features_out = {}
    def hook(module, input, output):
        features_out['vec'] = output.detach().cpu()

    # ConvNeXt-Small: classifier[1] is the AdaptiveAvgPool → flatten → Linear
    # We hook the layer just before the final Linear (classifier[1] = Flatten output)
    model.classifier[1].register_forward_hook(hook)

    return model, features_out, mod.get_transform()


def extract_features(model, features_out, records, transform, device,
                     batch_size=64) -> np.ndarray:
    ds     = FeatureDataset(records, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    all_feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            _ = model(imgs.to(device))
            all_feats.append(features_out['vec'].numpy())
    return np.vstack(all_feats)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(features: np.ndarray, reducer=None):
    """Fit UMAP (or reuse fitted reducer) and return (embedding, reducer)."""
    if reducer is None:
        print(f"  Fitting UMAP on {features.shape[0]:,} × {features.shape[1]} features...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
        embedding = reducer.fit_transform(features)
    else:
        print(f"  Transforming {features.shape[0]:,} overlay points with fitted UMAP...")
        embedding = reducer.transform(features)
    return embedding, reducer


# ── Plotting ──────────────────────────────────────────────────────────────────

BAND_COLORS  = {'X': '#e41a1c', 'Ku': '#377eb8', 'Ka': '#4daf4a', 'W': '#984ea3'}
POL_COLORS   = {'HH': '#e41a1c', 'HV': '#377eb8', 'VH': '#4daf4a', 'VV': '#984ea3'}
AZ_BIN_EDGES = [0, 45, 90, 135, 180, 225, 270, 315, 360]
AZ_BIN_LABELS = ['0–45°', '45–90°', '90–135°', '135–180°',
                 '180–225°', '225–270°', '270–315°', '315–360°']
AZ_BIN_COLORS = dict(zip(AZ_BIN_LABELS,
                         [plt.cm.hsv(i / 8) for i in range(8)]))


def _azimuth_bin(az):
    """Map azimuth (0-360) to an 8-sector bin label."""
    az = float(az) % 360
    for i in range(len(AZ_BIN_EDGES) - 1):
        if AZ_BIN_EDGES[i] <= az < AZ_BIN_EDGES[i + 1]:
            return AZ_BIN_LABELS[i]
    return AZ_BIN_LABELS[-1]


def _attribute_palette(attribute: str):
    """Return the (value→color) dict for a given attribute name."""
    return {
        'angle':      ANGLE_COLORS,
        'main_class': CLASS4_COLORS,
        'band':       BAND_COLORS,
        'polarization': POL_COLORS,
        'azimuth_bin':  AZ_BIN_COLORS,
    }[attribute]


def _record_value(r: dict, attribute: str):
    """Extract the display value for a record given the attribute name."""
    if attribute == 'angle':
        return float(r['depression_angle'])
    if attribute == 'main_class':
        return MAIN_CLASS_MAP.get(r['target_type'], 'Special')
    if attribute == 'band':
        raw = r.get('band') or 'unknown'
        # Normalize DB uppercase (KU, KA) to title case (Ku, Ka)
        return {'KU': 'Ku', 'KA': 'Ka'}.get(raw, raw)
    if attribute == 'polarization':
        return r.get('polarization') or 'unknown'
    if attribute == 'azimuth_bin':
        az = r.get('azimuth_angle')
        return _azimuth_bin(az) if az is not None else 'unknown'
    raise ValueError(f"Unknown attribute: {attribute}")


ATTR_TITLE = {
    'angle':        'Depression Angle',
    'main_class':   'Main Class (4-way)',
    'band':         'Radar Band',
    'polarization': 'Polarization',
    'azimuth_bin':  'Azimuth Bin (45° sectors)',
}


def plot_by_attribute(embedding, records, attribute: str, out_path,
                      angle_tag: str = ''):
    """Generic scatter plot colored by any acquisition attribute."""
    palette = _attribute_palette(attribute)
    fig, ax = plt.subplots(figsize=(12, 10))
    plotted = False
    for val, color in palette.items():
        idx = np.where([_record_value(r, attribute) == val
                        for r in records])[0]
        if len(idx):
            label = f'{int(val)}°' if attribute == 'angle' else str(val)
            ax.scatter(embedding[idx, 0], embedding[idx, 1],
                       c=[color] if not isinstance(color, str) else color,
                       s=8, alpha=0.55, label=label)
            plotted = True
    if not plotted:
        print(f"  WARNING: no data found for attribute '{attribute}'")
        plt.close()
        return
    title_attr = ATTR_TITLE.get(attribute, attribute)
    angle_note = f' — {angle_tag.strip("_")} images only' if angle_tag else ''
    ax.legend(markerscale=3, fontsize=12, title=title_attr)
    ax.set_title(f'Feature Space — colored by {title_attr}{angle_note}', fontsize=13)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_view1_4class(embedding, records, out_path):
    """View 1: colored by 4 main classes."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for cls, color in CLASS4_COLORS.items():
        mask = [MAIN_CLASS_MAP.get(r['target_type'], 'Special') == cls
                for r in records]
        idx  = np.where(mask)[0]
        if len(idx):
            ax.scatter(embedding[idx, 0], embedding[idx, 1],
                      c=color, s=8, alpha=0.5, label=cls)

    ax.legend(markerscale=3, fontsize=12)
    ax.set_title('Feature Space — colored by Main Class (4-way)\n'
                 'Well-separated clusters = good class structure', fontsize=13)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_view2_angle(embedding, records, out_path):
    """View 2: colored by depression angle."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for angle, color in ANGLE_COLORS.items():
        mask = [float(r['depression_angle']) == angle for r in records]
        idx  = np.where(mask)[0]
        if len(idx):
            ax.scatter(embedding[idx, 0], embedding[idx, 1],
                      c=color, s=8, alpha=0.5, label=f'{int(angle)}°')

    ax.legend(markerscale=3, fontsize=12, title='Depression angle')
    ax.set_title('Feature Space — colored by Depression Angle\n'
                 'Mixed colors = angle-invariant features  |  '
                 'Separated colors = angle dominates', fontsize=13)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_view3_single_vehicle(embedding, records, vehicle: str, out_path):
    """View 3: one vehicle, colored by depression angle."""
    mask = [r['target_type'] == vehicle for r in records]
    idx  = np.where(mask)[0]

    if len(idx) == 0:
        print(f"  WARNING: no records found for vehicle '{vehicle}'")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    angles_present = sorted(set(float(records[i]['depression_angle']) for i in idx))
    for angle in angles_present:
        color    = ANGLE_COLORS.get(angle, 'gray')
        sub_idx  = [i for i in idx
                    if float(records[i]['depression_angle']) == angle]
        ax.scatter(embedding[sub_idx, 0], embedding[sub_idx, 1],
                  c=color, s=20, alpha=0.7, label=f'{int(angle)}°')

    # Also plot everything else faintly for context
    other_idx = np.where([not m for m in mask])[0]
    ax.scatter(embedding[other_idx, 0], embedding[other_idx, 1],
              c='lightgray', s=4, alpha=0.2, zorder=0)

    ax.legend(markerscale=2, fontsize=12, title='Depression angle')
    ax.set_title(f'Feature Space — {vehicle} only\n'
                 f'Tight clusters per angle = angle causes separation  |  '
                 f'Mixed = good generalization', fontsize=12)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_view4_vehicles(embedding, records, train_angle: float, out_path):
    """View 4: angle-filtered images colored by vehicle (up to 40 classes)."""
    vehicles = sorted(set(r['target_type'] for r in records))
    # Two distinct colormaps blended for up to 40 vehicles
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    vehicle_colors = {v: palette[i % len(palette)] for i, v in enumerate(vehicles)}

    fig, ax = plt.subplots(figsize=(14, 11))
    for v in vehicles:
        idx = np.where([r['target_type'] == v for r in records])[0]
        if len(idx):
            ax.scatter(embedding[idx, 0], embedding[idx, 1],
                      c=[vehicle_colors[v]], s=10, alpha=0.65, label=v)

    ax.legend(markerscale=2, fontsize=7, ncol=2,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.set_title(f'Feature Space ({int(train_angle)}° only) — colored by Vehicle\n'
                 'Tight per-vehicle clusters = identity drives the space', fontsize=13)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_view5_overlay(embedding_train, train_records,
                       embedding_overlay, overlay_records,
                       train_angle: float, out_path):
    """View 5: UMAP fit on train_angle, then overlay other angles."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Base — 15° in light gray
    ax.scatter(embedding_train[:, 0], embedding_train[:, 1],
              c='lightgray', s=4, alpha=0.35, zorder=0,
              label=f'{int(train_angle)}° (train base)')

    # Overlaid angles in colour
    for angle, color in ANGLE_COLORS.items():
        if angle == train_angle:
            continue
        idx = np.where([float(r['depression_angle']) == angle
                        for r in overlay_records])[0]
        if len(idx):
            ax.scatter(embedding_overlay[idx, 0], embedding_overlay[idx, 1],
                      c=color, s=12, alpha=0.7, label=f'{int(angle)}° (overlay)')

    ax.legend(markerscale=2.5, fontsize=12)
    ax.set_title(
        f'Feature Space — {int(train_angle)}° base + other angles overlaid\n'
        'Overlaps with gray = angle-invariant features  |  '
        'Separate region = angle-specific', fontsize=13)
    ax.set_xlabel('UMAP dimension 1')
    ax.set_ylabel('UMAP dimension 2')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',             default='rot_baseline_full_v1')
    parser.add_argument('--samples-per-class', type=int, default=200,
                        help='Images per vehicle per angle (default: 200)')
    parser.add_argument('--vehicle',           default=None,
                        help='Vehicle for View 3 (default: first in records)')
    parser.add_argument('--batch-size',        type=int, default=64)
    parser.add_argument('--train-angle',       type=float, default=None,
                        help='Fit UMAP on this angle only, e.g. --train-angle 15  '
                             'Adds View 4 (per-vehicle) and View 5 (angle overlay)')
    parser.add_argument('--overlay-angles',    nargs='+', type=float, default=None,
                        help='Angles to overlay on View 5 (default: 30 45 60)')
    parser.add_argument('--color-by',          default=None,
                        choices=['angle', 'band', 'polarization',
                                 'azimuth_bin', 'main_class'],
                        help='Extra plot colored by this attribute '
                             '(angle/band/polarization/azimuth_bin/main_class)')
    args = parser.parse_args()

    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\nDevice: {device}")

    # ── Output directory: one folder per run ─────────────────────────────────
    angle_tag  = f'_15deg' if args.train_angle else ''
    run_name   = f'{args.model}{angle_tag}'
    out_dir    = Path(__file__).parent.parent / 'logs' / f'umap_{run_name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model, features_out, transform = load_model_and_extractor(args.model, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading images ({args.samples_per_class} per class per angle)...")
    overlay_angles = args.overlay_angles or [30.0, 45.0, 60.0]
    records, overlay_records = load_records(
        args.samples_per_class,
        vehicle_filter=args.vehicle,
        train_angle=args.train_angle,
        overlay_angles=overlay_angles if args.train_angle else None,
    )

    # ── Extract features (train set) ──────────────────────────────────────────
    print(f"\nExtracting features...")
    features = extract_features(model, features_out, records,
                                transform, device, args.batch_size)
    print(f"  Feature matrix: {features.shape[0]:,} × {features.shape[1]}")

    # ── UMAP (fit on train set) ────────────────────────────────────────────────
    print(f"\nReducing dimensions...")
    embedding, reducer = run_umap(features)

    # ── Plot View 1 — 4 main classes ──────────────────────────────────────────
    print(f"\nPlotting View 1 — 4 main classes...")
    plot_view1_4class(embedding, records, out_dir / '4class.png')

    # ── Plot View 2 — depression angle ────────────────────────────────────────
    print(f"\nPlotting View 2 — depression angle...")
    plot_view2_angle(embedding, records, out_dir / 'angle.png')

    # ── Plot View 3 — single vehicle ──────────────────────────────────────────
    vehicle = args.vehicle or records[0]['target_type']
    print(f"\nPlotting View 3 — single vehicle: {vehicle}...")
    plot_view3_single_vehicle(embedding, records, vehicle,
                              out_dir / f'vehicle_{vehicle}.png')

    # ── Views 4 + 5 only when --train-angle is set ────────────────────────────
    if args.train_angle is not None:
        print(f"\nPlotting View 4 — {int(args.train_angle)}° only, colored by vehicle...")
        plot_view4_vehicles(embedding, records, args.train_angle,
                            out_dir / 'vehicles_only.png')

        if overlay_records:
            print(f"\nExtracting features for overlay angles {overlay_angles}...")
            overlay_features = extract_features(model, features_out, overlay_records,
                                                transform, device, args.batch_size)
            overlay_embedding, _ = run_umap(overlay_features, reducer=reducer)

            print(f"\nPlotting View 5 — {int(args.train_angle)}° base + angle overlay...")
            plot_view5_overlay(embedding, records,
                               overlay_embedding, overlay_records,
                               args.train_angle,
                               out_dir / 'angle_overlay.png')

    # ── Extra --color-by plot ─────────────────────────────────────────────────
    if args.color_by:
        print(f"\nPlotting extra view — colored by {args.color_by}...")
        plot_by_attribute(embedding, records, args.color_by,
                          out_dir / f'{args.color_by}.png',
                          angle_tag=angle_tag)

    print(f"\nDone. All plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
