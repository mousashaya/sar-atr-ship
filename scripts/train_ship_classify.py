"""
SAR ship type classification using SAR chips + AIS vessel_type labels.
DINOv2 ViT-S/14 backbone + linear head, cross-entropy loss.
4 classes: Tanker, Cargo, Tug/Special, Fishing

Usage:
    venv/bin/python scripts/train_ship_classify.py [--grd]
    venv/bin/python scripts/train_ship_classify.py        # SLC (default)
"""

import argparse
import glob
import json
import random
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--grd', action='store_true', help='Use GRD chips instead of SLC')
args = parser.parse_args()

if args.grd:
    CHIPS_DIR  = Path('/Volumes/Machine Learning/sar_ship_chips_gulf')
    MODELS_DIR = Path('models/saved/ship_classify_grd')
    LOGS_DIR   = Path('logs/ship_classify_grd')
    MODE       = 'GRD'
else:
    CHIPS_DIR  = Path('/Volumes/Machine Learning/sar_ship_chips_slc_gulf')
    MODELS_DIR = Path('models/saved/ship_classify_v2')
    LOGS_DIR   = Path('logs/ship_classify_v2')
    MODE       = 'SLC'

LABELS_PATH  = Path('data/resolved_labels.json')
SEED         = 42
EPOCHS       = 30
LR           = 1e-4
BACKBONE_LR_SCALE = 0.1
UNFREEZE     = 4
BATCH_SIZE   = 64
MIN_CHIPS    = 30
MIN_CONFIDENCE = 'MEDIUM'   # include MEDIUM confidence labels for full 4-class experiment

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Load resolved labels — keyed by MMSI string
with open(LABELS_PATH) as f:
    RESOLVED = json.load(f)

# Only keep vessels at or above MIN_CONFIDENCE
CONF_RANK = {'HIGH': 2, 'MEDIUM': 1}
MIN_RANK  = CONF_RANK[MIN_CONFIDENCE]
TRUSTED   = {
    mmsi: v for mmsi, v in RESOLVED.items()
    if v['label'] != 'Unresolved' and CONF_RANK.get(v['confidence'], 0) >= MIN_RANK
}
print(f"Trusted vessels ({MIN_CONFIDENCE}+ confidence): {len(TRUSTED)}")

CLASS_NAMES  = ['Tanker', 'Cargo', 'Tug/Special', 'Fishing']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _normalize,
])
_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _normalize,
])

def chip_to_tensor(chip_vv, chip_vh, augment=False):
    vv = chip_vv.astype(np.float32)
    vh = chip_vh.astype(np.float32) if chip_vh is not None else vv
    vv_db = 10 * np.log10(vv + 1.0)
    vh_db = 10 * np.log10(vh + 1.0)
    def norm(x):
        p2, p98 = np.percentile(x[np.isfinite(x)], [2, 98])
        return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)
    rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return (_aug if augment else _eval)(img)


class ChipDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records  # [(path, class_idx), ...]
        self.augment = augment

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        f = np.load(path, allow_pickle=True)
        chip_vh = f['chip_vh'] if 'chip_vh' in f else None
        t = chip_to_tensor(f['chip_vv'], chip_vh, augment=self.augment)
        return t, label


def scan_and_split():
    by_class = defaultdict(list)
    for npz in sorted(CHIPS_DIR.glob('**/*.npz')):
        try:
            f    = np.load(npz, allow_pickle=True)
            mmsi = str(int(f['mmsi']))
            info = TRUSTED.get(mmsi)
            if info is None:
                continue
            cls = info['label']
            if cls not in CLASS_TO_IDX:
                continue
            by_class[cls].append(npz)
        except Exception:
            continue

    print("Class distribution (resolved labels):")
    for cls in CLASS_NAMES:
        print(f"  {cls:15s}: {len(by_class[cls])} chips")

    # 70/15/15 stratified split by class
    train, val, test = [], [], []
    for cls, paths in by_class.items():
        if len(paths) < MIN_CHIPS:
            print(f"  Skipping {cls} — only {len(paths)} chips")
            continue
        random.shuffle(paths)
        n      = len(paths)
        n_val  = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        idx    = CLASS_TO_IDX[cls]
        val   += [(p, idx) for p in paths[:n_val]]
        test  += [(p, idx) for p in paths[n_val:n_val + n_test]]
        train += [(p, idx) for p in paths[n_val + n_test:]]

    return train, val, test


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_recs, val_recs, test_recs = scan_and_split()
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}  Test: {len(test_recs)}")

    # Model: DINOv2 + linear head
    backbone = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True, verbose=False
    )
    for p in backbone.parameters():
        p.requires_grad = False
    for block in list(backbone.blocks)[-UNFREEZE:]:
        for p in block.parameters():
            p.requires_grad = True

    head = nn.Linear(384, len(CLASS_NAMES))

    class ClassifyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            return self.head(self.backbone(x))

    model = ClassifyModel().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params     = list(head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR * BACKBONE_LR_SCALE},
        {'params': head_params,     'lr': LR},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(ChipDataset(train_recs, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ChipDataset(val_recs),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ChipDataset(test_recs),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    def evaluate(loader):
        model.eval()
        correct = total = 0
        per_class_correct = defaultdict(int)
        per_class_total   = defaultdict(int)
        with torch.no_grad():
            for imgs, labels in loader:
                preds = model(imgs.to(device)).argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total   += len(labels)
                for p, l in zip(preds.tolist(), labels.tolist()):
                    per_class_total[l]   += 1
                    per_class_correct[l] += int(p == l)
        oa = correct / total
        f1s = []
        for i, cls in enumerate(CLASS_NAMES):
            if per_class_total[i] > 0:
                f1s.append(per_class_correct[i] / per_class_total[i])
        return oa, np.mean(f1s), per_class_correct, per_class_total

    print(f"\n{'='*55}")
    print(f"  SAR Ship Classification — 4 classes — DINOv2 — {MODE}")
    print(f"{'='*55}\n")

    best_oa = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(model(imgs), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_oa, val_f1, _, _ = evaluate(val_loader)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={np.mean(losses):.4f}  "
              f"val_OA={val_oa*100:.1f}%  val_F1={val_f1*100:.1f}%  ({elapsed:.0f}s)", flush=True)

        if val_oa > best_oa:
            best_oa = val_oa
            torch.save(model.state_dict(), MODELS_DIR / 'best.pt')
            print(f"    *** New best OA: {val_oa*100:.1f}% ***")

        history.append(dict(epoch=epoch, loss=np.mean(losses), val_oa=val_oa, val_f1=val_f1))

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(MODELS_DIR / 'best.pt', map_location=device))
    test_oa, test_f1, per_correct, per_total = evaluate(test_loader)

    print(f"\n{'='*55}")
    print(f"  Test OA:  {test_oa*100:.1f}%")
    print(f"  Test F1:  {test_f1*100:.1f}%")
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        if per_total[i] > 0:
            print(f"    {cls:15s}: {per_correct[i]/per_total[i]*100:.1f}%  ({per_total[i]} chips)")
    print(f"{'='*55}\n")

    json.dump({'test_oa': test_oa, 'test_f1': test_f1, 'history': history},
              open(LOGS_DIR / 'results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
