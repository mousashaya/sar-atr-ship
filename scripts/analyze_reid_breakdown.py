"""
scripts/analyze_reid_breakdown.py
===================================
Per-category breakdown of ship re-ID results on the test set.
Loads the saved best model and reports rank-1/rank-5/mAP by:
  - Vessel type
  - Ship length bucket
  - Number of passes

Usage:
    venv/bin/python scripts/analyze_reid_breakdown.py
    venv/bin/python scripts/analyze_reid_breakdown.py --mae   # use MAE backbone model
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CHIPS_DIRS = [
    Path('/Volumes/Machine Learning/sar_ship_chips'),
    Path('/Volumes/Machine Learning/sar_ship_chips_gulf'),
]
MAE_BACKBONE  = Path('models/saved/sar_mae_v1/backbone.pt')
MODELS_DIR    = Path('models/saved/ship_reid_v1')
SEED          = 42
MIN_PASSES    = 4
MIN_LENGTH    = 100
EMBED_DIM     = 384
PROJ_DIM      = 128
CROP_SIZE     = 256
BLOCK_SIZE    = 8

import torch.nn as nn
import math


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
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return arr[r0:r0+size, c0:c0+size]


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def chip_to_tensor(chip_vv, chip_vh):
    vv = center_crop(chip_vv.astype(np.float32))
    vh = center_crop(chip_vh.astype(np.float32))
    vv_db = subtract_background(10 * np.log10(vv + 1.0))
    vh_db = subtract_background(10 * np.log10(vh + 1.0))

    def norm(x):
        valid = x[np.isfinite(x)]
        if valid.size == 0:
            return np.zeros_like(x)
        p2, p98 = np.percentile(valid, [2, 98])
        return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)

    rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
    return _preprocess(Image.fromarray((rgb * 255).astype(np.uint8)))


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        w = F.normalize(self.weight, dim=1)
        cos_theta = (embeddings @ w.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = cos_theta.acos()
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits = (1 - one_hot) * cos_theta + one_hot * (theta + self.m).clamp(0, math.pi).cos()
        return F.cross_entropy(logits * self.s, labels)


class ReIDModel(nn.Module):
    def __init__(self, num_classes, use_mae=False):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', pretrained=False, verbose=False
        )
        if use_mae and MAE_BACKBONE.exists():
            self.backbone.load_state_dict(torch.load(MAE_BACKBONE, map_location='cpu'))
        self.proj = nn.Sequential(
            nn.Linear(EMBED_DIM, PROJ_DIM, bias=False),
            nn.BatchNorm1d(PROJ_DIM),
        )
        self.arcface = ArcFaceHead(PROJ_DIM, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.proj(feats)
        return F.normalize(feats, dim=1)


def scan_chips():
    records = []
    for d in CHIPS_DIRS:
        if not d.exists():
            continue
        for npz in sorted(d.glob('**/*.npz')):
            if npz.name == 'embeddings.npz':
                continue
            try:
                f = np.load(npz, allow_pickle=True)
                mmsi        = int(f['mmsi'])
                length      = float(f['length_m']) if 'length_m' in f else 0.0
                vtype       = str(f['vessel_type']) if 'vessel_type' in f else 'unknown'
                records.append((npz, mmsi, length, vtype))
            except Exception:
                continue
    return records


def build_test_split(records):
    by_mmsi = defaultdict(list)
    for path, mmsi, length, vtype in records:
        if length >= MIN_LENGTH:
            by_mmsi[mmsi].append((path, length, vtype))

    qualified = {m: chips for m, chips in by_mmsi.items() if len(chips) >= MIN_PASSES}
    mmsis = sorted(qualified.keys())
    random.shuffle(mmsis)

    n      = len(mmsis)
    n_val  = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    test_mmsis = set(mmsis[n_val:n_val + n_test])

    test_recs = []
    for m in test_mmsis:
        for path, length, vtype in qualified[m]:
            test_recs.append((path, m, length, vtype, len(qualified[m])))
    return test_recs, len(qualified)


class ChipDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, mmsi, length, vtype, n_passes = self.records[idx]
        f = np.load(path, allow_pickle=True)
        t = chip_to_tensor(f['chip_vv'], f['chip_vh'])
        return t, mmsi, length, vtype, n_passes


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def extract_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    embs, mmsis, lengths, vtypes, passes = [], [], [], [], []
    for imgs, ms, ls, vs, ps in loader:
        embs.append(model(imgs.to(device)).cpu())
        mmsis.extend(ms.tolist())
        lengths.extend(ls.tolist())
        vtypes.extend(vs)
        passes.extend(ps.tolist())
    return torch.cat(embs), mmsis, lengths, vtypes, passes


def retrieval_per_ship(embs, mmsis):
    """Returns dict: mmsi → (rank1_hit, rank5_hit, ap)"""
    by_id = defaultdict(list)
    for i, m in enumerate(mmsis):
        by_id[m].append(i)

    gallery_idx = [idxs[0] for idxs in by_id.values()]
    g_emb  = embs[gallery_idx]
    g_ids  = [mmsis[i] for i in gallery_idx]

    results = {}
    for m, idxs in by_id.items():
        query_idxs = idxs[1:]
        if not query_idxs:
            continue
        q_emb = embs[query_idxs]
        dist  = torch.cdist(q_emb, g_emb)
        ranked = dist.argsort(dim=1)

        r1_hits, r5_hits, aps = 0, 0, []
        for qi in range(len(query_idxs)):
            correct = [g_ids[j] == m for j in ranked[qi].tolist()]
            if correct[0]: r1_hits += 1
            if any(correct[:5]): r5_hits += 1
            hits = [i + 1 for i, c in enumerate(correct) if c]
            if hits:
                ap = sum((k+1)/h for k, h in enumerate(hits)) / len(hits)
                aps.append(ap)
        nq = len(query_idxs)
        results[m] = (r1_hits / nq, r5_hits / nq, sum(aps)/len(aps) if aps else 0.0)
    return results


def print_breakdown(label, groups, per_ship_results, mmsi_meta):
    """groups: {group_name: [mmsi, ...]}"""
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  {'Category':<22} {'Ships':>6} {'Rank-1':>8} {'Rank-5':>8} {'mAP':>8}")
    print(f"  {'─'*50}")

    all_r1, all_r5, all_ap = [], [], []
    for gname, mmsi_list in sorted(groups.items()):
        r1s, r5s, aps = [], [], []
        for m in mmsi_list:
            if m in per_ship_results:
                r1, r5, ap = per_ship_results[m]
                r1s.append(r1); r5s.append(r5); aps.append(ap)
        if not r1s:
            continue
        print(f"  {gname:<22} {len(r1s):>6} {np.mean(r1s)*100:>7.1f}% {np.mean(r5s)*100:>7.1f}% {np.mean(aps)*100:>7.1f}%")
        all_r1.extend(r1s); all_r5.extend(r5s); all_ap.extend(aps)

    print(f"  {'─'*50}")
    print(f"  {'TOTAL':<22} {len(all_r1):>6} {np.mean(all_r1)*100:>7.1f}% {np.mean(all_r5)*100:>7.1f}% {np.mean(all_ap)*100:>7.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mae', action='store_true', help='Use MAE backbone model')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"Device: {device}")

    print("Scanning chips...")
    records = scan_chips()
    test_recs, n_qualified = build_test_split(records)
    print(f"  Qualified ships: {n_qualified}  Test ships: {len({m for _,m,_,_,_ in test_recs})}")

    # Need num_classes = n_train_ships (70% of qualified)
    n_train = n_qualified - 2 * max(1, int(n_qualified * 0.15))

    print(f"Loading model (MAE={'yes' if args.mae else 'no'})...")
    model = ReIDModel(num_classes=n_train, use_mae=args.mae).to(device)
    ckpt  = MODELS_DIR / 'best.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"  Loaded {ckpt}")

    print("Extracting embeddings...")
    ds   = ChipDataset(test_recs)
    embs, mmsis, lengths, vtypes, n_passes = extract_embeddings(model, ds, device)
    print(f"  {len(embs)} embeddings extracted")

    # Per-ship metadata (use first chip's metadata per ship)
    mmsi_meta = {}
    for m, l, v, p in zip(mmsis, lengths, vtypes, n_passes):
        if m not in mmsi_meta:
            mmsi_meta[m] = (l, v, p)

    per_ship = retrieval_per_ship(embs, mmsis)

    # ── Breakdown by vessel type ──
    def decode_vessel_type(v):
        try:
            code = int(float(str(v).strip()))
        except (ValueError, TypeError):
            code = -1
        if code == 0:                    return 'Unknown'
        if 20 <= code <= 29:             return 'WIG'
        if code == 30:                   return 'Fishing'
        if code in (31, 32):             return 'Towing'
        if code in (33, 34):             return 'Dredger/Diving'
        if code == 35:                   return 'Military'
        if code in (36, 37):             return 'Sailing/Pleasure'
        if code in (40, 41, 42, 43, 44, 45, 46, 47, 48, 49): return 'HSC'
        if code == 50:                   return 'Pilot'
        if code in (51, 52, 53, 54, 55): return 'Tug/Service'
        if code == 58:                   return 'Medical'
        if 60 <= code <= 69:             return 'Passenger'
        if 70 <= code <= 79:             return 'Cargo'
        if 80 <= code <= 89:             return 'Tanker'
        if 90 <= code <= 99:             return 'Other'
        return 'Unknown'

    by_type = defaultdict(list)
    for m, (l, v, p) in mmsi_meta.items():
        by_type[decode_vessel_type(v)].append(m)
    print_breakdown("By Vessel Type", by_type, per_ship, mmsi_meta)

    # ── Breakdown by length ──
    by_length = defaultdict(list)
    for m, (l, v, p) in mmsi_meta.items():
        if l < 150:       key = '100–150m'
        elif l < 200:     key = '150–200m'
        elif l < 250:     key = '200–250m'
        else:             key = '250m+'
        by_length[key].append(m)
    print_breakdown("By Ship Length", by_length, per_ship, mmsi_meta)

    # ── Breakdown by number of passes ──
    by_passes = defaultdict(list)
    for m, (l, v, p) in mmsi_meta.items():
        if p < 6:        key = '4–5 passes'
        elif p < 9:      key = '6–8 passes'
        elif p < 12:     key = '9–11 passes'
        else:            key = '12+ passes'
        by_passes[key].append(m)
    print_breakdown("By Number of Passes", by_passes, per_ship, mmsi_meta)

    # ── Top 10 best ships ──
    print(f"\n{'─'*55}")
    print(f"  Top 10 Best Ships")
    print(f"{'─'*55}")
    print(f"  {'MMSI':<12} {'Type':<16} {'Length':>8} {'Passes':>7} {'Rank-1':>8}")
    print(f"  {'─'*52}")
    ranked = sorted(per_ship.items(), key=lambda x: x[1][0], reverse=True)
    for m, (r1, r5, ap) in ranked[:10]:
        l, v, p = mmsi_meta.get(m, (0, 'unknown', 0))
        print(f"  {m:<12} {decode_vessel_type(v)[:15]:<16} {l:>7.0f}m {p:>6}  {r1*100:>7.1f}%")


if __name__ == '__main__':
    main()
