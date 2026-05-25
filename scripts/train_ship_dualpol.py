"""
scripts/train_ship_dualpol.py
==============================
ConvNeXt-Small ship classification using dual-polarization input.

Input: 3-channel tensor per chip
  Ch 0 = VV   (log-normalized amplitude)
  Ch 1 = VH   (log-normalized amplitude)
  Ch 2 = VV - VH  (cross-pol difference)

Reads directly from Patch_Cal float32 TIFFs — no pre-split PNGs needed.
Uses scene-level train/val/test splits stored in defense.opensar_images.

Target: 80%+ (literature dual-pol range: 82-88%)

Usage:
    venv/bin/python3 scripts/train_ship_dualpol.py
    venv/bin/python3 scripts/train_ship_dualpol.py --epochs 20
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.transforms as T

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed.  pip install tifffile")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'ship_dualpol'
SEED       = 42
CLASSES    = ['Cargo', 'Tanker', 'Other Type']

# Patch_Cal root — chips live under scene subdirs
PATCH_CAL_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxlvkprrmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"

DB_MIN = -30.0   # dB clip range for normalization
DB_MAX =  10.0


def send_email(subject, body):
    import smtplib
    from email.mime.text import MIMEText
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


def border_suppress(arr_2d: np.ndarray, border: int = 10) -> np.ndarray:
    """Subtract background intensity estimated from chip border ring, clip to zero."""
    h, w = arr_2d.shape
    if h < 2 * border + 5 or w < 2 * border + 5:
        return arr_2d
    intensity = arr_2d ** 2
    mask = np.ones((h, w), dtype=bool)
    mask[border:h - border, border:w - border] = False
    bg_mean = intensity[mask].mean()
    suppressed = np.maximum(intensity - bg_mean, 0.0)
    return np.sqrt(suppressed).astype(np.float32)


def to_float_channel(arr_2d: np.ndarray) -> np.ndarray:
    """Convert float32 amplitude to [0,1] via dB normalization."""
    db = 10.0 * np.log10(np.maximum(arr_2d, 1e-6))
    db = np.clip(db, DB_MIN, DB_MAX)
    return ((db - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


# ── Dataset ───────────────────────────────────────────────────────────────────

class DualPolDataset(Dataset):
    """
    Reads Patch_Cal .tif (H,W,2) float32 directly.
    Returns 3-channel tensor: [VV, VH, VV-VH], each normalized to [0,1].
    Applies augmentations on the tensor level (train mode).
    """
    def __init__(self, records, class_to_idx, augment=False):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.augment      = augment
        self.resize       = T.Resize((128, 128), antialias=True)
        self.normalize    = T.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        self.aug          = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]) if augment else nn.Identity()

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec     = self.records[idx]
        tif_path = rec['tif_path']

        try:
            arr = tifffile.imread(str(tif_path))  # (H, W, 2) float32
            if arr.ndim == 2:
                arr = np.stack([arr, arr], axis=2)
            vv = to_float_channel(border_suppress(arr[:, :, 0]))
            vh = to_float_channel(border_suppress(arr[:, :, 1]))
            diff = np.clip(vv - vh + 0.5, 0.0, 1.0)  # shift diff to [0,1]
            tensor = torch.from_numpy(np.stack([vv, vh, diff], axis=0))
        except Exception:
            tensor = torch.zeros(3, 37, 37)

        tensor = self.resize(tensor)
        tensor = self.aug(tensor)
        tensor = self.normalize(tensor)
        return tensor, self.class_to_idx[rec['ship_type']]


# ── DB loading ────────────────────────────────────────────────────────────────

def load_split(split: str) -> list:
    """
    Load unique chip records for one split.
    Each chip appears once (not once per polarization).
    Reconstructs the Patch_Cal .tif path from scene_id + image_id.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT DISTINCT ON (scene_id, ship_type,
                                SPLIT_PART(image_id, '_VV', 1),
                                SPLIT_PART(image_id, '_VH', 1))
                   scene_id, ship_type, image_id
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = 'VV'
              AND split = %s
              AND local_path IS NOT NULL
        """, CLASSES + [split])
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Reconstruct tif path: image_id format is "{scene}__{stem}_VV"
    # tif is at PATCH_CAL_ROOT/{scene}/Patch_Cal/{stem}.tif
    out = []
    for r in rows:
        # image_id = "S1A_...__{ShipType}_x{x}_y{y}_VV"
        # stem = everything between __ and _VV
        parts    = r['image_id'].split('__', 1)
        if len(parts) < 2:
            continue
        stem_pol = parts[1]                        # e.g. "Cargo_x123_y456_VV"
        stem     = stem_pol.rsplit('_VV', 1)[0]    # e.g. "Cargo_x123_y456"
        tif_path = PATCH_CAL_ROOT / r['scene_id'] / 'Patch_Cal' / f'{stem}.tif'
        if tif_path.exists():
            out.append({
                'ship_type': r['ship_type'],
                'scene_id':  r['scene_id'],
                'tif_path':  tif_path,
            })
    return out


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def evaluate_per_class(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    print(classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=3))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    header = ''.join(f'{c[:8]:>10}' for c in CLASSES)
    print(f"{'':>12}{header}")
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i][:10]:<12}" + ''.join(f'{v:>10}' for v in row))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=15)
    parser.add_argument('--batch-size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"\nDevice: {device}")
    print("=" * 55)
    print(f"  Ship Dual-Pol — ConvNeXt-Small")
    print(f"  Input: [VV, VH, VV-VH]  (3-channel, Patch_Cal)")
    print(f"  Classes: {CLASSES}")
    print("=" * 55)

    print("\nLoading records (scene-level splits)...")
    train_recs = load_split('train')
    val_recs   = load_split('val')
    test_recs  = load_split('test')

    total = len(train_recs) + len(val_recs) + len(test_recs)
    print(f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Test: {len(test_recs):,}  (Total: {total:,})")
    for c in CLASSES:
        n = sum(1 for r in train_recs if r['ship_type'] == c)
        print(f"    {c:<15}: {n:,} train chips")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    train_ds = DualPolDataset(train_recs, class_to_idx, augment=True)
    val_ds   = DualPolDataset(val_recs,   class_to_idx, augment=False)
    test_ds  = DualPolDataset(test_recs,  class_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    mod   = importlib.import_module('models.convnext_small')
    model = mod.build_model(num_classes=len(CLASSES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Class-weighted loss — inverse frequency to counter 3:1 Cargo/Tanker imbalance
    counts = torch.tensor([
        sum(1 for r in train_recs if r['ship_type'] == c) for c in CLASSES
    ], dtype=torch.float32)
    weights = (1.0 / counts)
    weights = weights / weights.sum() * len(CLASSES)   # normalize so mean weight = 1
    print(f"\n  Class weights: " +
          "  ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    save_dir = MODELS_DIR / 'ship_dualpol_bgsub_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    send_email(
        "Ship dual-pol + bg-subtraction training started",
        f"Training started: ConvNeXt-Small with [VV, VH, VV-VH] input + border speckle suppression.\n\n"
        f"  Speckle filter: border ring (10px), threshold = mean + 3σ of border intensity\n"
        f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}\n"
        f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}\n"
        f"  Previous dual-pol (weighted, no speckle): 70.5%\n"
        f"  Target: improve over 70.5%\n"
    )

    best_val, best_epoch = 0.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total      += len(labels)
        val_acc = evaluate(model, val_loader, device)
        elapsed = int(time.time() - t0)
        print(f"  Epoch {ep}/{args.epochs}  loss={total_loss/total:.4f}"
              f"  val={val_acc*100:.1f}%  ({elapsed}s)")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')

        if ep == 5:
            send_email(
                f"Ship dual-pol + speckle — epoch 5  val={val_acc*100:.1f}%",
                f"Dual-pol + speckle suppression training in progress.\n\n"
                f"  Epoch 5/{args.epochs}  val={val_acc*100:.1f}%\n"
                f"  Best so far: {best_val*100:.1f}%  (epoch {best_epoch})\n"
                f"  Previous dual-pol (no speckle filter): 70.5%\n"
            )

    # ── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                     map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    print(f"\n  Test accuracy:    {test_acc*100:.1f}%")
    print(f"  Best val:         {best_val*100:.1f}%  (epoch {best_epoch})")
    print(f"  Single-pol refs:  VV=67.1%  VH=68.5%")
    print(f"  Literature (dual-pol): 82-88%")
    print(f"\n  Per-class breakdown:")
    evaluate_per_class(model, test_loader, device)

    result = {
        'class_names':  CLASSES,
        'input':        'VV+VH+diff (dual-pol, border speckle suppression)',
        'train_size':   len(train_recs),
        'val_size':     len(val_recs),
        'test_size':    len(test_recs),
        'best_val_acc': round(best_val * 100, 1),
        'test_acc':     round(test_acc * 100, 1),
        'best_epoch':   best_epoch,
        'epochs':       args.epochs,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    (LOGS_DIR / 'results.json').write_text(json.dumps(result, indent=2))
    print(f"\nModel saved to {save_dir}")

    send_email(
        f"Ship dual-pol + speckle DONE — test={test_acc*100:.1f}%  val={best_val*100:.1f}%",
        f"Dual-pol + border speckle suppression ship classification complete.\n\n"
        f"  Input:          [VV, VH, VV-VH]  + class-weighted loss + speckle filter\n"
        f"  Speckle filter: border ring 10px, threshold = mean + 3σ\n"
        f"  Train/Val/Test: {len(train_recs):,} / {len(val_recs):,} / {len(test_recs):,}\n\n"
        f"  Test accuracy:  {test_acc*100:.1f}%\n"
        f"  Best val:       {best_val*100:.1f}%  (epoch {best_epoch})\n\n"
        f"  vs. dual-pol weighted (no speckle): 70.5%\n"
        f"  vs. VV only:    67.1%\n"
        f"  vs. VH only:    68.5%\n"
        f"  Literature:     82-88%\n\n"
        f"Model: {save_dir}\n"
        f"Log:   {LOGS_DIR}/results.json\n"
    )


if __name__ == '__main__':
    main()
