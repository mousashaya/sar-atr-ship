"""
scripts/umap_ship.py
====================
Interactive UMAP of ship feature space — Cargo vs Tanker, VV and VH.

Produces a single HTML file with:
  - Two side-by-side UMAP plots (VV left, VH right)
  - 8 checkboxes to toggle any combination of:
      VV/VH × Cargo/Tanker × Correct/Misclassified

Hover shows: filename, true class, predicted class, ship dimensions.

Usage:
    python3 scripts/umap_ship.py
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

try:
    import umap as umap_lib
except ImportError:
    print("ERROR: umap-learn not installed.  pip install umap-learn")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed.  pip install plotly")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR    = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR      = Path(__file__).parent.parent / 'logs' / 'umap_ship'
SEED          = 42
CLASSES       = ['Cargo', 'Tanker']
TRAIN_CLASSES = ['Cargo', 'Tanker', 'Other Type']

MODEL_FOR_POL = {'VV': 'ship_soc_v1', 'VH': 'ship_soc_VH_v1'}

COLORS = {
    ('Cargo',  True):  '#2563eb',   # blue        — Cargo correct
    ('Cargo',  False): '#f97316',   # orange       — Cargo misclassified
    ('Tanker', True):  '#16a34a',   # green        — Tanker correct
    ('Tanker', False): '#9333ea',   # purple       — Tanker misclassified
}
MARKER_SYMBOL = {True: 'circle', False: 'x'}
MARKER_SIZE   = {True: 5,        False: 9}
MARKER_OPACITY= {True: 0.55,     False: 0.95}


# ── DB ────────────────────────────────────────────────────────────────────────

def load_records(pol: str) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT image_id, local_path, ship_type,
                   ship_length_m, ship_width_m, scene_id
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = %s
              AND local_path IS NOT NULL
        """, CLASSES + [pol])
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    print(f"  {pol}: {len(rows):,} images  "
          f"(Cargo {sum(1 for r in rows if r['ship_type']=='Cargo'):,}  "
          f"Tanker {sum(1 for r in rows if r['ship_type']=='Tanker'):,})")
    return rows


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShipDataset(Dataset):
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


# ── Model + features ──────────────────────────────────────────────────────────

def load_model(pol: str, device):
    mod   = importlib.import_module('models.convnext_small')
    model = mod.build_model(num_classes=len(TRAIN_CLASSES)).to(device)
    weights = MODELS_DIR / MODEL_FOR_POL[pol] / 'weights.pt'
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded {MODEL_FOR_POL[pol]}")
    return model, mod.get_transform()


def extract(model, records, transform, device, batch_size=64):
    feats_buf = {}

    def hook(m, inp, out):
        feats_buf['v'] = out.detach().cpu()

    handle = model.classifier[1].register_forward_hook(hook)
    loader = DataLoader(ShipDataset(records, transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=4, persistent_workers=True)
    all_feats, all_preds = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(device))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_feats.append(feats_buf['v'].numpy())
    handle.remove()

    features    = np.vstack(all_feats)
    pred_labels = [TRAIN_CLASSES[p] for p in all_preds]
    return features, pred_labels


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(features: np.ndarray) -> np.ndarray:
    print(f"  Fitting UMAP on {features.shape[0]:,} × {features.shape[1]}...")
    reducer   = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    embedding = reducer.fit_transform(features)
    print(f"  Done → {embedding.shape}")
    return embedding


# ── Build combined Plotly figure ───────────────────────────────────────────────

def build_figure(data_vv, data_vh):
    """
    data_vv / data_vh: dict with keys
      'records', 'embedding', 'pred_labels', 'pol'
    Returns (fig, trace_meta) where trace_meta is list of dicts
    describing each trace for the checkbox panel.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['VV Polarization', 'VH Polarization'],
        horizontal_spacing=0.04,
    )

    trace_meta = []   # [{pol, cls, correct, n, trace_idx}]
    trace_idx  = 0

    for col_idx, d in enumerate([data_vv, data_vh], start=1):
        records    = d['records']
        embedding  = d['embedding']
        pred_labels = d['pred_labels']
        pol        = d['pol']

        for cls in CLASSES:
            for correct in [True, False]:
                mask = [
                    r['ship_type'] == cls and (pred_labels[i] == r['ship_type']) == correct
                    for i, r in enumerate(records)
                ]
                idx = [i for i, m in enumerate(mask) if m]

                color   = COLORS[(cls, correct)]
                symbol  = MARKER_SYMBOL[correct]
                size    = MARKER_SIZE[correct]
                opacity = MARKER_OPACITY[correct]

                hover = [
                    f"<b>{records[i]['ship_type']}</b> → pred: {pred_labels[i]}<br>"
                    f"File: {Path(records[i]['local_path']).name}<br>"
                    f"Scene: {records[i].get('scene_id','?')}<br>"
                    f"Length: {records[i].get('ship_length_m','?')} m  "
                    f"Width: {records[i].get('ship_width_m','?')} m"
                    for i in idx
                ]

                trace = go.Scatter(
                    x=embedding[idx, 0] if idx else [],
                    y=embedding[idx, 1] if idx else [],
                    mode='markers',
                    name=f"{pol} {cls} {'✓' if correct else '✗'}",
                    marker=dict(color=color, symbol=symbol,
                                size=size, opacity=opacity,
                                line=dict(width=0 if correct else 0.5,
                                          color='#444' if not correct else color)),
                    text=hover,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                )
                fig.add_trace(trace, row=1, col=col_idx)
                trace_meta.append(dict(
                    pol=pol, cls=cls, correct=correct,
                    n=len(idx), trace_idx=trace_idx,
                    color=color,
                ))
                trace_idx += 1

    n_vv = len(data_vv['records'])
    n_vh = len(data_vh['records'])
    fig.update_layout(
        title=dict(
            text='Ship Feature Space — Cargo vs Tanker (VV and VH)',
            font_size=17,
        ),
        hovermode='closest',
        width=1300,
        height=700,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        margin=dict(t=80, b=20, l=20, r=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig, trace_meta


# ── HTML with custom checkbox panel ───────────────────────────────────────────

CHECKBOX_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #fff; }
  #controls {
    display: flex; gap: 32px; padding: 14px 24px 10px;
    background: #f1f5f9; border-bottom: 1px solid #e2e8f0;
    flex-wrap: wrap; align-items: flex-start;
  }
  .pol-group { display: flex; flex-direction: column; gap: 6px; }
  .pol-label {
    font-size: 12px; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: #64748b; margin-bottom: 2px;
  }
  .cb-row {
    display: flex; align-items: center; gap: 7px;
    cursor: pointer; user-select: none; font-size: 13px;
  }
  .cb-row input[type=checkbox] {
    width: 15px; height: 15px; cursor: pointer; accent-color: #334155;
  }
  .swatch { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .swatch.x { border-radius: 2px; }
  .cb-label { color: #1e293b; }
  .cb-count { color: #94a3b8; font-size: 11px; margin-left: 2px; }
  #btn-row { display: flex; gap: 8px; align-items: center;
             padding: 0 24px 8px; background: #f1f5f9;
             border-bottom: 1px solid #e2e8f0; }
  button {
    font-size: 12px; padding: 4px 12px; border-radius: 5px; cursor: pointer;
    border: 1px solid #cbd5e1; background: #fff; color: #334155;
  }
  button:hover { background: #e2e8f0; }
</style>
"""

def build_html(fig, trace_meta) -> str:
    import plotly.io as pio
    fig_json = pio.to_json(fig)

    # Build checkbox rows
    pol_groups = {'VV': [], 'VH': []}
    for tm in trace_meta:
        pol_groups[tm['pol']].append(tm)

    cb_html = ''
    for pol, items in pol_groups.items():
        cb_html += f'<div class="pol-group"><div class="pol-label">{pol}</div>'
        for tm in items:
            cid      = f'cb_{tm["pol"]}_{tm["cls"]}_{int(tm["correct"])}'
            lbl      = f'{tm["cls"]} {"correct" if tm["correct"] else "misclassified"}'
            br = '50%' if tm['correct'] else '3px'
            cb_html += (
                f'<label class="cb-row">'
                f'<input type="checkbox" id="{cid}" '
                f'data-trace="{tm["trace_idx"]}" checked '
                f'onchange="toggleTrace(this)">'
                f'<span class="swatch" style="background:{tm["color"]};border-radius:{br}"></span>'
                f'<span class="cb-label">{lbl}</span>'
                f'<span class="cb-count">n={tm["n"]:,}</span>'
                f'</label>\n'
            )
        cb_html += '</div>'

    js = """
<script>
var graphDiv;
document.addEventListener('DOMContentLoaded', function() {
  var figData = """ + fig_json + """;
  graphDiv = document.getElementById('plot');
  Plotly.newPlot(graphDiv, figData.data, figData.layout, {responsive: true});
});

function toggleTrace(cb) {
  var idx = parseInt(cb.dataset.trace);
  Plotly.restyle(graphDiv, {visible: cb.checked ? true : false}, [idx]);
}

function selectAll() {
  document.querySelectorAll('#controls input[type=checkbox]').forEach(function(cb) {
    cb.checked = true;
    Plotly.restyle(graphDiv, {visible: true}, [parseInt(cb.dataset.trace)]);
  });
}

function selectNone() {
  document.querySelectorAll('#controls input[type=checkbox]').forEach(function(cb) {
    cb.checked = false;
    Plotly.restyle(graphDiv, {visible: false}, [parseInt(cb.dataset.trace)]);
  });
}

function selectOnly(pol, cls, correct) {
  document.querySelectorAll('#controls input[type=checkbox]').forEach(function(cb) {
    var id = cb.id;
    var match = id === 'cb_' + pol + '_' + cls + '_' + correct;
    cb.checked = match;
    Plotly.restyle(graphDiv, {visible: match}, [parseInt(cb.dataset.trace)]);
  });
}
</script>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ship UMAP — Cargo vs Tanker</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
{CHECKBOX_CSS}
</head>
<body>
<div id="controls">
{cb_html}
</div>
<div id="btn-row">
  <button onclick="selectAll()">Select all</button>
  <button onclick="selectNone()">Select none</button>
  <button onclick="selectOnly('VV','Cargo','0')">VV misclassified only</button>
  <button onclick="selectOnly('VH','Cargo','0')">VH misclassified only</button>
</div>
<div id="plot"></div>
{js}
</body>
</html>
"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def process_pol(pol: str, device) -> dict:
    print(f"\n{'='*55}\n  {pol}\n{'='*55}")
    records = load_records(pol)
    print("Loading model...")
    model, transform = load_model(pol, device)
    print("Extracting features + predictions...")
    features, pred_labels = extract(model, records, transform, device)
    n_correct = sum(1 for i, r in enumerate(records)
                    if pred_labels[i] == r['ship_type'])
    print(f"  Accuracy on Cargo+Tanker: {n_correct/len(records)*100:.1f}%")
    print("Running UMAP...")
    embedding = run_umap(features)
    return dict(pol=pol, records=records, embedding=embedding,
                pred_labels=pred_labels)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = (torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    data_vv = process_pol('VV', device)
    data_vh = process_pol('VH', device)

    print("\nBuilding combined figure...")
    fig, trace_meta = build_figure(data_vv, data_vh)

    print("Writing HTML...")
    html = build_html(fig, trace_meta)
    out  = LOGS_DIR / 'umap_ship_combined.html'
    out.write_text(html, encoding='utf-8')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
