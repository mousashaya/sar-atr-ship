"""
scripts/train_ship_reid.py
===========================
SAR ship re-identification using ArcFace loss.

Architecture: SAR MAE Stage 1 backbone (or vanilla DINOv2) + projection head (384→128)
              + ArcFace head (128 → n_train_ships) during training only
Loss:         ArcFace — classification with angular margin (s=64, m=0.5)
Identity:     MMSI from AIS — no manual labeling required

Preprocessing: center crop 256×256 (ship region) + block-wise background subtraction
Split:   70/15/15 by MMSI (not by chip — prevents leakage)
Filter:  ships with ≥MIN_PASSES passes and length ≥MIN_LENGTH m

Usage:
    venv/bin/python scripts/train_ship_reid.py
    venv/bin/python scripts/train_ship_reid.py --epochs 30 --lr 1e-4
    venv/bin/python scripts/train_ship_reid.py --vanilla-dinov2
"""

import argparse
import json
import math
import random
import smtplib
import sys
import time
from collections import defaultdict
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

GRD_DIRS     = [
    Path('/Volumes/Machine Learning/sar_ship_chips'),
    Path('/Volumes/Machine Learning/sar_ship_chips_gulf'),
]
SLC_DIRS     = [Path('/Volumes/Machine Learning/sar_ship_chips_slc_gulf')]
MAE_BACKBONE = Path('/Volumes/Machine Learning/sar_mae_slc_v1/backbone_best.pt')
LABELS_PATH  = Path('data/resolved_labels.json')
SEED         = 42
MIN_PASSES   = 2
MIN_LENGTH   = 100      # metres
P            = 16       # ships per batch
K            = 4        # chips per ship per batch
PROJ_DIM     = 128      # projection head output

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxlvkprrmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"


def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = TO_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_USER, GMAIL_APP_PASS)
            s.send_message(msg)
        print(f"  Email sent: {subject}")
    except Exception as e:
        print(f"  Email failed: {e}")


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


# ── SAR preprocessing ─────────────────────────────────────────────────────────

CROP_SIZE   = 256   # center crop — ship region, discard outer ocean
BLOCK_SIZE  = 8     # background subtraction block size

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _normalize,
])

_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _normalize,
])


def subtract_background(arr):
    """Block-wise local median subtraction to suppress sea clutter."""
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


def chip_to_tensor_dinov2(chip_vv, chip_vh, augment=False):
    vv = center_crop(chip_vv.astype(np.float32))
    vh = center_crop(chip_vh.astype(np.float32))
    vv_db = 10 * np.log10(vv + 1.0)
    vh_db = 10 * np.log10(vh + 1.0)
    vv_db = subtract_background(vv_db)
    vh_db = subtract_background(vh_db)
    def norm(x):
        valid = x[np.isfinite(x)]
        if valid.size == 0: return np.zeros_like(x)
        p2, p98 = np.percentile(valid, [2, 98])
        return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)
    rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return (_aug if augment else _preprocess)(img)


def chip_to_tensor_croma(chip_vv, chip_vh, augment=False):
    sys.path.insert(0, 'models')
    from croma import chip_to_tensor as croma_chip
    vv = center_crop(chip_vv.astype(np.float32))
    vh = center_crop(chip_vh.astype(np.float32)) if chip_vh is not None else vv
    return croma_chip(vv, vh, augment=augment)


# ── Dataset ────────────────────────────────────────────────────────────────────

def scan_chips(chips_dirs, tanker_mmsis=None):
    """Return list of (npz_path, mmsi, length_m) for all valid chips."""
    records = []
    for d in chips_dirs:
        if not d.exists():
            continue
        for npz in sorted(d.glob('**/*.npz')):
            if npz.name == 'embeddings.npz':
                continue
            try:
                f = np.load(npz, allow_pickle=True)
                mmsi   = int(f['mmsi'])
                if tanker_mmsis is not None and mmsi not in tanker_mmsis:
                    continue
                length = float(f['length_m']) if 'length_m' in f else 0.0
                records.append((npz, mmsi, length))
            except Exception:
                continue
    return records


def build_splits(records):
    """Split MMSIs 70/15/15 after filtering by MIN_PASSES and MIN_LENGTH."""
    by_mmsi = defaultdict(list)
    for path, mmsi, length in records:
        if length >= MIN_LENGTH:
            by_mmsi[mmsi].append((path, length))

    qualified = {m: chips for m, chips in by_mmsi.items() if len(chips) >= MIN_PASSES}
    mmsis = sorted(qualified.keys())
    random.shuffle(mmsis)

    n      = len(mmsis)
    n_val  = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    val_mmsis  = set(mmsis[:n_val])
    test_mmsis = set(mmsis[n_val:n_val + n_test])
    train_mmsis = set(mmsis[n_val + n_test:])

    def subset(mmsi_set):
        out = []
        for m in mmsi_set:
            for path, _ in qualified[m]:
                out.append((path, m))
        return out

    return subset(train_mmsis), subset(val_mmsis), subset(test_mmsis), qualified


class ChipDataset(Dataset):
    def __init__(self, records, augment=False, tensor_fn=None):
        self.records   = records
        self.augment   = augment
        self.tensor_fn = tensor_fn or chip_to_tensor_dinov2

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, mmsi = self.records[idx]
        f = np.load(path, allow_pickle=True)
        chip_vh = f['chip_vh'] if 'chip_vh' in f else None
        t = self.tensor_fn(f['chip_vv'], chip_vh, augment=self.augment)
        return t, mmsi



# ── Model ──────────────────────────────────────────────────────────────────────

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
    def __init__(self, backbone_name='dinov2', unfreeze_blocks=6,
                 num_classes=None, arc_scale=64.0, arc_margin=0.5):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'croma':
            sys.path.insert(0, 'models')
            from croma import build_model as build_croma, FEATURE_DIM
            croma_model = build_croma(num_classes=1)  # num_classes unused, we strip head
            self.backbone = croma_model.encoder
            embed_dim = FEATURE_DIM  # 768
            for p in self.backbone.parameters():
                p.requires_grad = False
            # unfreeze last N transformer blocks in CROMA's ViT
            blocks = list(self.backbone.s1_encoder.transformer.layers)
            for block in blocks[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            self._forward_backbone = lambda x: self.backbone(SAR_images=x)['SAR_GAP']
        else:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False
            )
            if MAE_BACKBONE.exists():
                state = torch.load(MAE_BACKBONE, map_location='cpu')
                self.backbone.load_state_dict(state)
                print(f"  Loaded SAR MAE backbone from {MAE_BACKBONE}")
            else:
                print("  Using vanilla DINOv2 backbone")
            embed_dim = 384
            for p in self.backbone.parameters():
                p.requires_grad = False
            for block in list(self.backbone.blocks)[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            self._forward_backbone = lambda x: self.backbone(x)

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, PROJ_DIM, bias=False),
            nn.BatchNorm1d(PROJ_DIM),
        )
        self.arcface = ArcFaceHead(PROJ_DIM, num_classes, s=arc_scale, m=arc_margin) if num_classes else None

    def forward(self, x, labels=None):
        feats = self._forward_backbone(x)
        feats = self.proj(feats)
        feats = F.normalize(feats, dim=1)
        if labels is not None and self.arcface is not None:
            return self.arcface(feats, labels)
        return feats


# ── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, records, device, batch_size=64, tensor_fn=None):
    model.eval()
    ds = ChipDataset(records, augment=False, tensor_fn=tensor_fn)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, mmsis = [], []
    for imgs, labels in loader:
        embs.append(model(imgs.to(device)).cpu())
        mmsis.extend(labels.tolist())
    return torch.cat(embs), mmsis


def compute_metrics(embs, mmsis):
    """Gallery = first chip per ship. Query = rest. Returns rank1, rank5, mAP."""
    by_id = defaultdict(list)
    for i, m in enumerate(mmsis):
        by_id[m].append(i)

    gallery_idx = [idxs[0] for idxs in by_id.values()]
    query_idx   = [i for idxs in by_id.values() for i in idxs[1:]]

    if not query_idx:
        return 0.0, 0.0, 0.0

    g_emb  = embs[gallery_idx]               # (G, D)
    q_emb  = embs[query_idx]                 # (Q, D)
    g_ids  = [mmsis[i] for i in gallery_idx]
    q_ids  = [mmsis[i] for i in query_idx]

    # Pairwise distances (Q × G)
    dist = torch.cdist(q_emb, g_emb)         # (Q, G)
    ranked = dist.argsort(dim=1)             # (Q, G) indices into gallery

    rank1 = rank5 = 0
    aps = []
    for qi, q_id in enumerate(q_ids):
        correct = [g_ids[j] == q_id for j in ranked[qi].tolist()]
        if correct[0]:
            rank1 += 1
        if any(correct[:5]):
            rank5 += 1
        # AP: positions of correct hits
        hits = [i + 1 for i, c in enumerate(correct) if c]
        if hits:
            ap = sum((k + 1) / h for k, h in enumerate(hits)) / len(hits)
            aps.append(ap)

    n = len(q_ids)
    return rank1 / n, rank5 / n, (sum(aps) / len(aps) if aps else 0.0)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',            type=int,   default=30)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--unfreeze',          type=int,   default=6)
    parser.add_argument('--arc-scale',         type=float, default=64.0)
    parser.add_argument('--arc-margin',        type=float, default=0.5)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1)
    parser.add_argument('--backbone',          default='dinov2', choices=['dinov2', 'croma'])
    parser.add_argument('--tanker-only',       action='store_true',
                        help='Restrict to HIGH-confidence Tankers only')
    parser.add_argument('--slc',               action='store_true',
                        help='Use SLC chips instead of GRD')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    device = get_device()
    print(f"Device: {device}")

    chip_to_tensor = chip_to_tensor_croma if args.backbone == 'croma' else chip_to_tensor_dinov2

    run_tag = f"reid_{args.backbone}_{'slc' if args.slc else 'grd'}{'_tanker' if args.tanker_only else ''}"
    models_dir = Path(__file__).parent.parent / 'models' / 'saved' / run_tag
    logs_dir   = Path(__file__).parent.parent / 'logs' / run_tag
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Tanker filter ──
    tanker_mmsis = None
    if args.tanker_only:
        with open(LABELS_PATH) as f:
            resolved = json.load(f)
        tanker_mmsis = {
            int(mmsi) for mmsi, v in resolved.items()
            if v['label'] == 'Tanker' and v['confidence'] == 'HIGH'
        }
        print(f"Tanker-only filter: {len(tanker_mmsis)} HIGH-confidence Tanker MMSIs")

    # ── Data ──
    chips_dirs = SLC_DIRS if args.slc else GRD_DIRS
    print(f"Chip source: {'SLC' if args.slc else 'GRD'}")
    print("Scanning chips...")
    records = scan_chips(chips_dirs, tanker_mmsis=tanker_mmsis)
    train_recs, val_recs, test_recs, qualified = build_splits(records)

    n_train_ids = len({m for _, m in train_recs})
    n_val_ids   = len({m for _, m in val_recs})
    n_test_ids  = len({m for _, m in test_recs})
    print(f"  Qualified ships: {len(qualified)}  (≥{MIN_PASSES} passes, ≥{MIN_LENGTH}m)")
    print(f"  Train: {len(train_recs)} chips / {n_train_ids} ships")
    print(f"  Val:   {len(val_recs)} chips / {n_val_ids} ships")
    print(f"  Test:  {len(test_recs)} chips / {n_test_ids} ships")

    # Class index mapping: MMSI → 0..N-1 (needed for cross-entropy)
    train_mmsis   = sorted({m for _, m in train_recs})
    mmsi_to_idx   = {m: i for i, m in enumerate(train_mmsis)}
    n_train_ships = len(train_mmsis)

    loader = DataLoader(
        ChipDataset(train_recs, augment=True, tensor_fn=chip_to_tensor),
        batch_size=64, shuffle=True, num_workers=0, drop_last=True,
    )

    # ── Model ──
    print(f"Loading backbone ({args.backbone}, unfreeze_blocks={args.unfreeze})...")
    model = ReIDModel(
        backbone_name=args.backbone,
        unfreeze_blocks=args.unfreeze,
        num_classes=n_train_ships,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")
    print(f"  Training ships (classes): {n_train_ships}")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.proj.parameters()) + list(model.arcface.parameters())
    if backbone_params:
        backbone_lr = args.lr * args.backbone_lr_scale
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': args.lr},
        ], weight_decay=1e-4)
        print(f"  Differential LR — backbone: {backbone_lr:.2e}  head: {args.lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        print(f"  Backbone frozen — head LR: {args.lr:.2e}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    print(f"\n{'='*60}")
    print(f"  SAR Ship Re-ID  —  ArcFace")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}")
    print(f"  ArcFace s={args.arc_scale}  m={args.arc_margin}")
    print(f"{'='*60}\n")

    best_rank1  = 0.0
    best_epoch  = 0
    history     = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for imgs, mmsis in loader:
            imgs   = imgs.to(device)
            labels = torch.tensor(
                [mmsi_to_idx[m.item()] for m in mmsis], device=device
            )
            loss = model(imgs, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(losses) if losses else 0.0

        # Evaluate on val set
        val_embs, val_mmsis = extract_embeddings(model, val_recs, device, tensor_fn=chip_to_tensor)
        rank1, rank5, mAP   = compute_metrics(val_embs, val_mmsis)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:2d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"rank1={rank1*100:.1f}%  rank5={rank5*100:.1f}%  "
              f"mAP={mAP*100:.1f}%  ({elapsed:.0f}s)", flush=True)

        row = dict(epoch=epoch, loss=train_loss, rank1=rank1, rank5=rank5, mAP=mAP)
        history.append(row)

        if rank1 > best_rank1:
            best_rank1 = rank1
            best_epoch = epoch
            torch.save(model.state_dict(), models_dir / 'best.pt')
            print(f"    *** New best rank-1: {rank1*100:.1f}% ***")

    # ── Final test evaluation ──
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(models_dir / 'best.pt', map_location=device))
    test_embs, test_mmsis = extract_embeddings(model, test_recs, device, tensor_fn=chip_to_tensor)
    test_r1, test_r5, test_mAP = compute_metrics(test_embs, test_mmsis)

    print(f"\n{'='*60}")
    print(f"  Test results (best model from epoch {best_epoch})")
    print(f"  Rank-1: {test_r1*100:.1f}%")
    print(f"  Rank-5: {test_r5*100:.1f}%")
    print(f"  mAP:    {test_mAP*100:.1f}%")
    print(f"{'='*60}\n")

    results = {
        'args': vars(args),
        'best_epoch': best_epoch,
        'best_val_rank1': best_rank1,
        'test_rank1': test_r1,
        'test_rank5': test_r5,
        'test_mAP': test_mAP,
        'n_train_ships': n_train_ids,
        'n_val_ships': n_val_ids,
        'n_test_ships': n_test_ids,
        'history': history,
    }
    with open(logs_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), models_dir / 'final.pt')
    print(f"Results saved to {logs_dir}/results.json")

    send_email(
        f"Ship Re-ID done — Rank-1 {test_r1*100:.1f}%  mAP {test_mAP*100:.1f}%",
        f"Best epoch: {best_epoch}/{args.epochs}\n"
        f"Val rank-1:  {best_rank1*100:.1f}%\n"
        f"Test rank-1: {test_r1*100:.1f}%\n"
        f"Test rank-5: {test_r5*100:.1f}%\n"
        f"Test mAP:    {test_mAP*100:.1f}%\n\n"
        f"Ships — train: {n_train_ids}  val: {n_val_ids}  test: {n_test_ids}\n"
        f"Args: {vars(args)}"
    )


if __name__ == '__main__':
    main()
