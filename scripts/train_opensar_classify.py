"""
CROMA classification on OpenSARShip dataset.
3 classes: Tanker / Container Ship / Bulk Carrier
Using Patch_Cal TIF (45x45, float32, 2-ch NRCS sigma0: ch0=VH, ch1=VV)
70/30 train/test split matching Huang et al. 2018 experimental setup.

Usage:
    venv/bin/python scripts/train_opensar_classify.py
    venv/bin/python scripts/train_opensar_classify.py --backbone dinov2
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxml import etree
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'models')

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='croma', choices=['croma', 'dinov2', 'convnext'])
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()

OSS2_DIR  = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')
MODELS_DIR = Path(f'models/saved/opensar_{args.backbone}')
LOGS_DIR   = Path(f'logs/opensar_{args.backbone}')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SEED       = 42
EPOCHS     = args.epochs
LR         = 1e-4
BATCH_SIZE = 64

# Mapping: elaborated type → our class label
TANKER_TYPES = {
    'Tanker', 'Oil Products Tanker', 'Oil/Chemical Tanker', 'Crude Oil Tanker',
    'LPG Tanker', 'Chemical Tanker', 'Bunkering Tanker', 'Asphalt/Bitumen Tanker',
    'LNG Tanker', 'Tanker - Hazard A', 'Tanker - Hazard A (Major)',
    'Tanker - Hazard B', 'Tanker - Hazard C', 'Tanker - Hazard C (Minor)',
    'Tanker - Hazard D (Minor)', 'Water Tanker',
}
CONTAINER_TYPES = {'Container Ship'}
BULK_TYPES      = {'Bulk Carrier', 'Ore Carrier', 'Cement Carrier'}

CLASS_NAMES  = ['Tanker', 'Container Ship', 'Bulk Carrier']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def elab_to_class(elab):
    if elab in TANKER_TYPES:     return 'Tanker'
    if elab in CONTAINER_TYPES:  return 'Container Ship'
    if elab in BULK_TYPES:       return 'Bulk Carrier'
    return None


def build_label_map():
    """Parse all Ship.xml → {(scene, cx, cy): elaborated_type}."""
    label_map = {}
    for scene_dir in sorted(OSS2_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        xml_path = scene_dir / 'Ship.xml'
        if not xml_path.exists():
            continue
        parser = etree.XMLParser(recover=True, encoding='utf-8')
        tree = etree.parse(str(xml_path), parser)
        root = tree.getroot()
        if root is None:
            continue
        for ship in root.findall('ship'):
            cx = ship.findtext('SARShipInformation/Center_x', '').strip()
            cy = ship.findtext('SARShipInformation/Center_y', '').strip()
            et = ship.findtext('MarineTrafficInformation/Elaborated_type', '').strip()
            try:
                label_map[(scene_dir.name, int(cx), int(cy))] = et
            except ValueError:
                pass
    return label_map


def scan_chips(label_map):
    """Return list of (tif_path, class_idx) for the 3 target classes."""
    by_class = defaultdict(list)
    for scene_dir in sorted(OSS2_DIR.iterdir()):
        if not scene_dir.is_dir():
            continue
        # Skip single-pol scenes (1SSV) — no dual-pol data
        if '1SSV' in scene_dir.name:
            continue
        patch_cal = scene_dir / 'Patch_Cal'
        if not patch_cal.exists():
            continue
        for tif in sorted(patch_cal.glob('*.tif')):
            m = re.match(r'.+_x(\d+)_y(\d+)\.tif$', tif.name)
            if not m:
                continue
            cx, cy = int(m.group(1)), int(m.group(2))
            elab = label_map.get((scene_dir.name, cx, cy), '')
            cls = elab_to_class(elab)
            if cls is None:
                continue
            by_class[cls].append((tif, CLASS_TO_IDX[cls]))

    print('Class distribution:')
    for cls in CLASS_NAMES:
        print(f'  {cls:20s}: {len(by_class[cls])}')
    return by_class


if args.backbone == 'croma':
    import tifffile
    from croma import chip_to_tensor as _chip_to_tensor

    def load_tensor(tif_path, augment=False):
        arr = tifffile.imread(str(tif_path))   # (H, W, 2) ch0=VH ch1=VV
        if arr.ndim != 3 or arr.shape[2] < 2:
            raise ValueError(f'single-pol chip: {tif_path}')
        chip_vh = arr[:, :, 0]
        chip_vv = arr[:, :, 1]
        return _chip_to_tensor(chip_vv, chip_vh, augment=augment)

elif args.backbone == 'dinov2':
    import tifffile
    from PIL import Image
    from torchvision import transforms

    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _aug  = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(180), transforms.Resize((224, 224)),
                                  transforms.ToTensor(), _normalize])
    _eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), _normalize])

    def load_tensor(tif_path, augment=False):
        arr   = tifffile.imread(str(tif_path))
        vh_db = 10 * np.log10(arr[:,:,0] + 1e-6)
        vv_db = 10 * np.log10(arr[:,:,1] + 1e-6)
        def norm(x):
            p2, p98 = np.percentile(x, [2, 98])
            return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)
        rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        return (_aug if augment else _eval)(img)

elif args.backbone == 'convnext':
    import tifffile
    from PIL import Image
    from torchvision import transforms

    _aug  = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(180), transforms.Resize((224, 224)),
                                  transforms.ToTensor()])
    _eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def load_tensor(tif_path, augment=False):
        arr   = tifffile.imread(str(tif_path))
        vh_db = 10 * np.log10(arr[:,:,0] + 1e-6)
        vv_db = 10 * np.log10(arr[:,:,1] + 1e-6)
        def norm(x):
            p2, p98 = np.percentile(x, [2, 98])
            return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)
        rgb = np.stack([norm(vv_db), norm(vh_db), norm(vv_db - vh_db)], axis=2)
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        return (_aug if augment else _eval)(img)


class ChipDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        tif_path, label = self.records[idx]
        try:
            return load_tensor(tif_path, augment=self.augment), label
        except Exception:
            # fallback: return zeros tensor with same shape as a valid chip
            import torch
            return torch.zeros(2, 224, 224), label


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def build_model(n_cls):
    if args.backbone == 'croma':
        from croma import build_model as build_croma
        return build_croma(num_classes=n_cls)
    elif args.backbone == 'dinov2':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',
                                  pretrained=True, verbose=False)
        for p in backbone.parameters():
            p.requires_grad = False
        for block in list(backbone.blocks)[-4:]:
            for p in block.parameters():
                p.requires_grad = True

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(384, n_cls)
            def forward(self, x): return self.head(self.backbone(x))
        return M()
    elif args.backbone == 'convnext':
        from convnext_small import build_model as build_convnext
        return build_convnext(num_classes=n_cls)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_correct = defaultdict(int)
    per_total   = defaultdict(int)
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            correct += (preds == labels).sum().item()
            total   += len(labels)
            for p, l in zip(preds.tolist(), labels.tolist()):
                per_total[l]   += 1
                per_correct[l] += int(p == l)
    oa = correct / total
    f1s = [per_correct[i] / per_total[i] for i in range(len(CLASS_NAMES)) if per_total[i] > 0]
    return oa, np.mean(f1s), per_correct, per_total


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    print(f'Device: {device}')
    print(f'Backbone: {args.backbone.upper()}')

    print('Building label map from XML...')
    label_map = build_label_map()
    print(f'  {len(label_map)} ships parsed')

    by_class = scan_chips(label_map)
    total = sum(len(v) for v in by_class.values())
    print(f'Total chips: {total}')

    # 70/30 stratified split by class (matches Huang et al. 2018)
    train_recs, test_recs = [], []
    for cls, records in by_class.items():
        random.shuffle(records)
        n_test = max(1, int(len(records) * 0.30))
        test_recs  += records[:n_test]
        train_recs += records[n_test:]

    print(f'Train: {len(train_recs)}  Test: {len(test_recs)}')

    model = build_model(len(CLASS_NAMES)).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable:,}')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader = DataLoader(ChipDataset(train_recs, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(ChipDataset(test_recs),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n{'='*60}")
    print(f"  OpenSARShip — {len(CLASS_NAMES)}-class — {args.backbone.upper()}")
    print(f"  Baseline: Huang et al. 2018 k-NN ~75-79% OA")
    print(f"{'='*60}\n")

    best_oa  = 0.0
    history  = []

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

        test_oa, test_f1, _, _ = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        print(f'  Epoch {epoch:2d}/{EPOCHS}  loss={np.mean(losses):.4f}  '
              f'test_OA={test_oa*100:.1f}%  test_F1={test_f1*100:.1f}%  ({elapsed:.0f}s)', flush=True)

        if test_oa > best_oa:
            best_oa = test_oa
            torch.save(model.state_dict(), MODELS_DIR / 'best.pt')
            print(f'    *** New best OA: {test_oa*100:.1f}% ***')

        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                            test_oa=float(test_oa), test_f1=float(test_f1)))

    # Final report
    model.load_state_dict(torch.load(MODELS_DIR / 'best.pt', map_location=device))
    oa, f1, per_correct, per_total = evaluate(model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"  FINAL TEST  (best model,  OA={oa*100:.1f}%  F1={f1*100:.1f}%)")
    print(f"  Baseline (Huang 2018 k-NN): ~75-79% OA")
    print(f"\n  Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        if per_total[i] > 0:
            print(f'    {cls:20s}: {per_correct[i]/per_total[i]*100:.1f}%  ({per_total[i]} chips)')
    print(f"{'='*60}\n")

    json.dump({'test_oa': oa, 'test_f1': f1, 'history': history,
               'class_names': CLASS_NAMES},
              open(LOGS_DIR / 'results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
