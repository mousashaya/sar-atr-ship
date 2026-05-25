"""
scripts/inspect_ship_errors.py
================================
Runs VV model on test set, shows a grid of:
  - Row 1: Tanker chips correctly classified
  - Row 2: Tanker chips misclassified as Cargo
  - Row 3: Cargo chips misclassified as Tanker

Saved to logs/ship_errors/error_grid.png
"""

import importlib, sys
from pathlib import Path
import numpy as np
import psycopg2, psycopg2.extras
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
OUT_DIR    = Path(__file__).parent.parent / 'logs' / 'ship_errors'
CLASSES    = ['Cargo', 'Tanker', 'Other Type']
N_SHOW     = 12   # chips per row


def load_test_records():
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT local_path, ship_type, scene_id, image_id
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = 'VV'
              AND split = 'test'
              AND local_path IS NOT NULL
        """, CLASSES)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_model(device):
    exp_dir = MODELS_DIR / 'ship_soc_VV_v1'
    import json
    config  = json.loads((exp_dir / 'config.json').read_text())
    mod     = importlib.import_module('models.convnext_small')
    model   = mod.build_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(
        torch.load(exp_dir / 'weights.pt', map_location=device, weights_only=True))
    model.eval()
    transform = mod.get_transform()
    return model, transform


def predict_all(model, transform, records, device, batch_size=128):
    from torch.utils.data import Dataset, DataLoader

    class DS(Dataset):
        def __init__(self, recs, tf):
            self.recs = recs; self.tf = tf
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            try:   img = Image.open(self.recs[i]['local_path']).convert('RGB')
            except: img = Image.new('RGB', (37, 37), 64)
            return self.tf(img), i

    loader = DataLoader(DS(records, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    preds = [None] * len(records)
    with torch.no_grad():
        for imgs, idxs in loader:
            out = model(imgs.to(device)).argmax(1).cpu().tolist()
            for pred, idx in zip(out, idxs.tolist()):
                preds[idx] = CLASSES[pred]
    return preds


def chip_image(path):
    """Load chip as display-ready RGB PIL image (upscaled 6×)."""
    try:
        img = Image.open(path).convert('L')
        img = img.resize((img.width * 6, img.height * 6), Image.NEAREST)
        return img.convert('RGB')
    except:
        return Image.new('RGB', (222, 222), 64)


def make_grid(records, preds, true_cls, pred_cls, n=N_SHOW):
    """Return list of (image, title) for true_cls chips predicted as pred_cls."""
    items = [(r, p) for r, p in zip(records, preds)
             if r['ship_type'] == true_cls and p == pred_cls]
    np.random.default_rng(42).shuffle(items)
    return [(chip_image(r['local_path']),
             f"{r['scene_id'][:16]}")
            for r, p in items[:n]]


def plot_rows(rows_data, out_path):
    """
    rows_data: list of (row_title, [(img, subtitle), ...])
    """
    n_cols = N_SHOW
    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.4, n_rows * 1.8))
    if n_rows == 1: axes = [axes]

    for row_i, (row_title, items) in enumerate(rows_data):
        for col_i in range(n_cols):
            ax = axes[row_i][col_i]
            if col_i < len(items):
                img, subtitle = items[col_i]
                ax.imshow(img)
                ax.set_xlabel(subtitle, fontsize=5, labelpad=2)
            else:
                ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(row_title, fontsize=8, fontweight='bold',
                              rotation=0, ha='right', va='center', labelpad=4)

    fig.suptitle('Ship chip inspection — VV polarization, test set',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))

    print("Loading test records...")
    records = load_test_records()
    print(f"  {len(records):,} test chips")

    print("Loading model and running predictions...")
    model, transform = load_model(device)
    preds = predict_all(model, transform, records, device)

    # Summary
    correct = sum(r['ship_type'] == p for r, p in zip(records, preds))
    print(f"  Test accuracy: {correct/len(records)*100:.1f}%")
    for cls in CLASSES:
        sub = [(r,p) for r,p in zip(records,preds) if r['ship_type']==cls]
        ok  = sum(1 for r,p in sub if r['ship_type']==p)
        print(f"  {cls:12s}: {ok}/{len(sub)} correct ({ok/len(sub)*100:.0f}%)")

    rows_data = [
        ("Tanker\n✓ correct",    make_grid(records, preds, 'Tanker', 'Tanker')),
        ("Tanker\n→ Cargo",      make_grid(records, preds, 'Tanker', 'Cargo')),
        ("Cargo\n→ Tanker",      make_grid(records, preds, 'Cargo',  'Tanker')),
        ("Other Type\n→ Cargo",  make_grid(records, preds, 'Other Type', 'Cargo')),
    ]

    plot_rows(rows_data, OUT_DIR / 'error_grid_vv.png')


if __name__ == '__main__':
    main()
