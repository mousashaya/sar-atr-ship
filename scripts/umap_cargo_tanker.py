"""
scripts/umap_cargo_tanker.py
=============================
Static UMAP of Cargo vs Tanker in VV feature space, colored by outcome:

  Cargo correct       (blue,   circle)
  Cargo → Tanker      (red,    X)       ← misclassified Cargo
  Tanker correct      (green,  circle)
  Tanker → Cargo      (orange, X)       ← misclassified Tanker

Output: logs/ship_errors/umap_cargo_tanker_vv.png
"""

import importlib, sys, json
from pathlib import Path
import numpy as np
import psycopg2, psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    import umap as umap_lib
except ImportError:
    print("ERROR: pip install umap-learn"); sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
OUT_DIR    = Path(__file__).parent.parent / 'logs' / 'ship_errors'
CLASSES    = ['Cargo', 'Tanker', 'Other Type']
SEED       = 42

COLORS = {
    ('Cargo',  True):  '#2563eb',   # blue   — Cargo correct
    ('Cargo',  False): '#dc2626',   # red    — Cargo misclassified
    ('Tanker', True):  '#16a34a',   # green  — Tanker correct
    ('Tanker', False): '#f97316',   # orange — Tanker misclassified
}


def load_records():
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT local_path, ship_type, image_id
            FROM defense.opensar_images
            WHERE ship_type IN ('Cargo','Tanker')
              AND polarization = 'VV'
              AND split = 'test'
              AND local_path IS NOT NULL
        """)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    print(f"  {len(rows):,} test chips (Cargo + Tanker, VV)")
    return rows


class DS(Dataset):
    def __init__(self, records, transform):
        self.records = records; self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        try:   img = Image.open(self.records[i]['local_path']).convert('RGB')
        except: img = Image.new('RGB', (37, 37), 64)
        return self.transform(img), i


def load_model(device):
    exp_dir = MODELS_DIR / 'ship_soc_VV_v1'
    config  = json.loads((exp_dir / 'config.json').read_text())
    mod     = importlib.import_module('models.convnext_small')
    model   = mod.build_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    return model, mod.get_transform()


def extract(model, records, transform, device):
    buf = {}
    def hook(m, inp, out): buf['v'] = out.detach().cpu()
    handle = model.classifier[1].register_forward_hook(hook)

    loader = DataLoader(DS(records, transform), batch_size=128,
                        shuffle=False, num_workers=0)
    feats, preds = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
            feats.append(buf['v'].numpy())
            preds.extend(model(imgs.to(device)).argmax(1).cpu().tolist())
    handle.remove()
    return np.vstack(feats), [CLASSES[p] for p in preds]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))

    print("Loading records...")
    records = load_records()

    print("Loading model + extracting features...")
    model, transform = load_model(device)
    features, preds  = extract(model, records, transform, device)

    print(f"Fitting UMAP on {features.shape[0]:,} × {features.shape[1]}...")
    reducer   = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(features)
    print("Done.")

    # ── Stats ─────────────────────────────────────────────────────────────────
    for cls in ['Cargo', 'Tanker']:
        sub   = [(r, p, i) for i, (r, p) in enumerate(zip(records, preds))
                 if r['ship_type'] == cls]
        right = sum(1 for r, p, _ in sub if p == cls)
        wrong = len(sub) - right
        print(f"  {cls:8s}: {right} correct, {wrong} misclassified "
              f"({wrong/len(sub)*100:.0f}% error rate)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_order = [
        ('Cargo',  True),
        ('Tanker', True),
        ('Cargo',  False),
        ('Tanker', False),
    ]

    for cls, correct in plot_order:
        idx = [i for i, (r, p) in enumerate(zip(records, preds))
               if r['ship_type'] == cls and (p == cls) == correct]
        if not idx: continue
        color  = COLORS[(cls, correct)]
        marker = 'o' if correct else 'X'
        size   = 12  if correct else 30
        alpha  = 0.4 if correct else 0.85
        other  = 'Tanker' if cls == 'Cargo' else 'Cargo'
        status = 'correct' if correct else f'→ {other}'
        mark   = 'correct' if correct else 'X'
        label  = f"{cls} {status} (n={len(idx)})"
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=color, marker=marker, s=size, alpha=alpha,
                   label=label, linewidths=0.3,
                   edgecolors='white' if correct else '#333',
                   zorder=3 if not correct else 2)

    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.9,
              loc='upper right')
    ax.set_title('Feature space — Cargo vs Tanker (VV, test set)\n'
                 'Circles = correct  |  ✗ = misclassified',
                 fontsize=11)
    ax.axis('off')
    fig.tight_layout()

    out = OUT_DIR / 'umap_cargo_tanker_vv.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
