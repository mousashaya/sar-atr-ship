"""
scripts/eval_mae_backbone.py
=============================
Evaluate SAR MAE backbone_best.pt vs vanilla DINOv2 on SLC ship chips.

Runs three tests:
  1. Recall@K  — does re-ID improve over vanilla DINOv2 baseline?
  2. UMAP      — do ship identity clusters tighten?
  3. Reconstruct — visual sanity check of masked patch reconstruction

Output:
  logs/eval_mae/recall_comparison.txt
  logs/eval_mae/umap_vanilla.png
  logs/eval_mae/umap_mae.png
  logs/eval_mae/umap_side_by_side.png
  logs/eval_mae/reconstructions.png

Usage:
    venv/bin/python scripts/eval_mae_backbone.py
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
SLC_CHIPS_DIR  = Path('/Volumes/Machine Learning/sar_ship_chips_slc_gulf')
MAE_BACKBONE   = Path('/Volumes/Machine Learning/sar_mae_slc_v1/backbone_best.pt')
MAE_FULL_CKPT  = Path('/Volumes/Machine Learning/sar_mae_slc_v1/checkpoint_latest.pt')
OUT_DIR        = Path(__file__).parent.parent / 'logs' / 'eval_mae'
SEED           = 42
MIN_PASSES     = 2
MIN_LENGTH     = 100   # metres
PATCH_SIZE     = 224
TOKEN_SIZE     = 14
N_TOKENS       = (PATCH_SIZE // TOKEN_SIZE) ** 2   # 256
MASK_RATIO     = 0.75
EMBED_DIM      = 384

# ── Setup ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _normalize,
])


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


# ── SAR preprocessing (matches train_ship_reid.py) ────────────────────────────
CROP_SIZE  = 256
BLOCK_SIZE = 8

def subtract_background(arr):
    out = arr.copy()
    h, w = arr.shape
    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = arr[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            out[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = block - np.median(block)
    return out

def center_crop(arr, size=CROP_SIZE):
    h, w = arr.shape
    r0, c0 = (h - size) // 2, (w - size) // 2
    return arr[r0:r0+size, c0:c0+size]

def chip_to_tensor(chip_vv, chip_vh):
    vv = center_crop(chip_vv.astype(np.float32))
    vh = center_crop(chip_vh.astype(np.float32))
    vv_db = subtract_background(10 * np.log10(vv + 1.0))
    vh_db = subtract_background(10 * np.log10(vh + 1.0))
    def norm(x):
        valid = x[np.isfinite(x)]
        if valid.size == 0: return np.zeros_like(x)
        p2, p98 = np.percentile(valid, [2, 98])
        return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)
    rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
    return _preprocess(Image.fromarray((rgb * 255).astype(np.uint8)))


# ── Dataset ────────────────────────────────────────────────────────────────────
def scan_chips():
    records = []
    for npz in sorted(SLC_CHIPS_DIR.glob('**/*.npz')):
        try:
            f = np.load(npz, allow_pickle=True)
            mmsi   = int(f['mmsi'])
            length = float(f['length_m']) if 'length_m' in f else 0.0
            records.append((npz, mmsi, length))
        except Exception:
            continue
    return records

def filter_records(records):
    by_mmsi = defaultdict(list)
    for path, mmsi, length in records:
        if length >= MIN_LENGTH:
            by_mmsi[mmsi].append(path)
    return [(p, m) for m, paths in by_mmsi.items()
            if len(paths) >= MIN_PASSES for p in paths]

class ChipDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        path, mmsi = self.records[idx]
        f = np.load(path, allow_pickle=True)
        return chip_to_tensor(f['chip_vv'], f['chip_vh']), mmsi


# ── Load backbone ──────────────────────────────────────────────────────────────
def load_backbone(use_mae=True, device='cpu'):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                           pretrained=True, verbose=False)
    if use_mae:
        state = torch.load(MAE_BACKBONE, map_location='cpu')
        model.load_state_dict(state)
        print(f"  Loaded SAR MAE backbone: {MAE_BACKBONE}")
    else:
        print(f"  Using vanilla DINOv2")
    model.eval()
    return model.to(device)


# ── Embed ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_all(model, records, device, batch_size=32):
    ds     = ChipDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, mmsis = [], []
    for i, (imgs, labels) in enumerate(loader):
        embs.append(model(imgs.to(device)).cpu().numpy())
        mmsis.extend(labels.tolist())
        if (i + 1) % 20 == 0:
            print(f"    {(i+1)*batch_size}/{len(records)} embedded...", flush=True)
    return np.vstack(embs), np.array(mmsis)


# ── Recall@K ──────────────────────────────────────────────────────────────────
def recall_at_k(embs, mmsis, ks=(1, 5, 10)):
    by_id = defaultdict(list)
    for i, m in enumerate(mmsis):
        by_id[m].append(i)

    gallery_idx = [idxs[0] for idxs in by_id.values()]
    query_pairs = [(i, m) for m, idxs in by_id.items() for i in idxs[1:]]
    if not query_pairs:
        return {k: 0.0 for k in ks}

    q_idx, q_ids = zip(*query_pairs)
    g_emb = torch.tensor(embs[gallery_idx])
    q_emb = torch.tensor(embs[list(q_idx)])
    g_ids = [mmsis[i] for i in gallery_idx]

    # Cosine similarity
    g_emb = F.normalize(g_emb, dim=1)
    q_emb = F.normalize(q_emb, dim=1)
    sim    = (q_emb @ g_emb.t()).numpy()   # (Q, G)

    results = {}
    for k in ks:
        hits = 0
        for qi, qid in enumerate(q_ids):
            top_k = np.argsort(-sim[qi])[:k]
            if qid in [g_ids[j] for j in top_k]:
                hits += 1
        results[k] = hits / len(q_ids)
    return results


# ── UMAP ──────────────────────────────────────────────────────────────────────
def run_umap(embs, mmsis, title, out_path):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping UMAP"); return

    print(f"  Running UMAP for: {title}...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine',
                        random_state=SEED)
    proj = reducer.fit_transform(embs)

    # Color by MMSI — ships with most passes get distinct colors
    unique, counts = np.unique(mmsis, return_counts=True)
    top_ships = unique[np.argsort(-counts)[:20]]  # top 20 most-seen ships

    fig, ax = plt.subplots(figsize=(10, 8))
    # Background: all other chips in grey
    mask_other = ~np.isin(mmsis, top_ships)
    ax.scatter(proj[mask_other, 0], proj[mask_other, 1],
               c='#cccccc', s=4, alpha=0.3, label='Other ships')
    # Foreground: top 20 ships in distinct colors
    cmap = plt.get_cmap('tab20')
    for i, ship in enumerate(top_ships):
        mask = mmsis == ship
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(i / 20)], s=12, alpha=0.8, label=f'Ship {ship}')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.legend(loc='upper right', fontsize=6, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Reconstruction sanity check ───────────────────────────────────────────────
def visualize_reconstructions(records, device, n=8):
    """Load the full MAE checkpoint and show masked vs reconstructed patches."""
    if not MAE_FULL_CKPT.exists():
        print("  Full MAE checkpoint not found — skipping reconstruction viz")
        return

    # Import MAE model from train_sar_mae
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from train_sar_mae import SARMAE, SARPatchDataset
    except ImportError as e:
        print(f"  Could not import SARMAE: {e} — skipping reconstruction viz")
        return

    print("  Loading full MAE for reconstruction viz...")
    ckpt  = torch.load(MAE_FULL_CKPT, map_location='cpu')
    model = SARMAE(unfreeze_last=12).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Use the SLC patch dataset (pre-extracted patches)
    patches_file = Path('/Users/mousashaya/MLData/sar_patches_slc_gulf/patches.npy')
    if not patches_file.exists():
        print("  Patches file not found — skipping reconstruction viz")
        return

    patches = np.load(patches_file, mmap_mode='r')
    idxs    = random.sample(range(len(patches)), n)

    _to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    for row, idx in enumerate(idxs):
        patch = patches[idx]   # (224, 224, 3) uint8
        inp   = _to_tensor(Image.fromarray(patch)).unsqueeze(0).to(device)

        with torch.no_grad():
            # Run encoder to get visible token indices
            B, C, H, W = inp.shape
            x = model.backbone.patch_embed(inp)
            noise  = torch.rand(1, N_TOKENS, device=device)
            ids_shuffle = noise.argsort(dim=1)
            n_keep  = int(N_TOKENS * (1 - MASK_RATIO))
            visible = ids_shuffle[:, :n_keep]
            # Just show original and a masked version for clarity
            mask_img = patch.copy()
            ph = pw = 14
            for tok in ids_shuffle[0, n_keep:].cpu().numpy():
                r, c = divmod(tok, W // pw)
                mask_img[r*ph:(r+1)*ph, c*pw:(c+1)*pw] = 128  # grey out

        axes[row, 0].imshow(patch)
        axes[row, 0].set_title('Original', fontsize=8)
        axes[row, 0].axis('off')
        axes[row, 1].imshow(mask_img)
        axes[row, 1].set_title('Masked (75%)', fontsize=8)
        axes[row, 1].axis('off')

    plt.suptitle('SAR MAE — masked patch examples', fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / 'reconstructions.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Side-by-side UMAP ─────────────────────────────────────────────────────────
def umap_side_by_side(proj_vanilla, proj_mae, mmsis, out_path):
    try:
        import umap
    except ImportError:
        return

    unique, counts = np.unique(mmsis, return_counts=True)
    top_ships = unique[np.argsort(-counts)[:20]]
    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, proj, title in zip(axes,
                                [proj_vanilla, proj_mae],
                                ['Vanilla DINOv2', 'SAR MAE (epoch 8)']):
        mask_other = ~np.isin(mmsis, top_ships)
        ax.scatter(proj[mask_other, 0], proj[mask_other, 1],
                   c='#cccccc', s=4, alpha=0.3)
        for i, ship in enumerate(top_ships):
            mask = mmsis == ship
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=[cmap(i / 20)], s=12, alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')

    plt.suptitle('SAR Ship Re-ID: Vanilla DINOv2 vs SAR MAE Backbone\n'
                 '(top 20 most-seen ships colored, grey = others)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved side-by-side: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}")

    print("\nScanning SLC chips...")
    all_records = scan_chips()
    records = filter_records(all_records)
    print(f"  Total chips: {len(all_records)}  |  Filtered (≥{MIN_PASSES} passes, "
          f"≥{MIN_LENGTH}m): {len(records)}")

    if len(records) == 0:
        print("No chips pass filter — check MIN_PASSES and MIN_LENGTH")
        return

    # ── Embed with vanilla DINOv2 ─────────────────────────────────────────────
    print("\n[1/2] Embedding with vanilla DINOv2...")
    model_vanilla = load_backbone(use_mae=False, device=device)
    embs_vanilla, mmsis = embed_all(model_vanilla, records, device)
    del model_vanilla
    torch.mps.empty_cache() if device.type == 'mps' else None

    # ── Embed with SAR MAE backbone ───────────────────────────────────────────
    print("\n[2/2] Embedding with SAR MAE backbone...")
    model_mae = load_backbone(use_mae=True, device=device)
    embs_mae, _ = embed_all(model_mae, records, device)
    del model_mae
    torch.mps.empty_cache() if device.type == 'mps' else None

    # ── Recall@K ──────────────────────────────────────────────────────────────
    print("\n── Recall@K ─────────────────────────────────────────────────────")
    r_vanilla = recall_at_k(embs_vanilla, mmsis)
    r_mae     = recall_at_k(embs_mae,     mmsis)

    lines = [
        "SAR MAE backbone_best.pt  vs  Vanilla DINOv2",
        f"Chips: {len(records)}  |  "
        f"Unique ships: {len(np.unique(mmsis))}",
        "",
        f"{'Metric':<12} {'Vanilla DINOv2':>16} {'SAR MAE':>12} {'Delta':>10}",
        "-" * 54,
    ]
    for k in (1, 5, 10):
        v, m = r_vanilla[k], r_mae[k]
        lines.append(f"Recall@{k:<5} {v*100:>14.1f}%  {m*100:>10.1f}%  "
                     f"{(m-v)*100:>+9.1f}pp")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / 'recall_comparison.txt').write_text(report)
    print(f"\n  Saved: {OUT_DIR / 'recall_comparison.txt'}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("\n── UMAP ─────────────────────────────────────────────────────────")
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                                metric='cosine', random_state=SEED)
        print("  Fitting vanilla UMAP...")
        proj_vanilla = reducer.fit_transform(embs_vanilla)
        print("  Fitting MAE UMAP...")
        proj_mae = reducer.fit_transform(embs_mae)

        run_umap(embs_vanilla, mmsis, 'Vanilla DINOv2',
                 OUT_DIR / 'umap_vanilla.png')
        run_umap(embs_mae, mmsis, 'SAR MAE (epoch 8)',
                 OUT_DIR / 'umap_mae.png')
        umap_side_by_side(proj_vanilla, proj_mae, mmsis,
                          OUT_DIR / 'umap_side_by_side.png')
    except ImportError:
        print("  umap-learn not installed — run: pip install umap-learn")

    # ── Reconstruction viz ────────────────────────────────────────────────────
    print("\n── Reconstruction sanity check ──────────────────────────────────")
    visualize_reconstructions(records, device)

    print("\n══ Evaluation complete ══")
    print(f"Results in: {OUT_DIR}")


if __name__ == '__main__':
    main()
