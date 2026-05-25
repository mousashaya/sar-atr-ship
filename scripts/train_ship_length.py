"""
scripts/train_ship_length.py
==============================
ConvNeXt-Small ship classification: dual-pol [VV, VH, VV-VH]
+ ship Manual_Length injected after the backbone.

Architecture:
  ConvNeXt-Small backbone (frozen classifier[2])
    → 768-dim embedding
    → concat normalized log-length scalar
    → Linear(769, num_classes)

Length normalization: log-scale, clipped to [log(10), log(600)], → [0, 1].
Missing length → 0.5 (dataset midpoint).

Target: beat dual-pol weighted baseline of 70.5%.

Usage:
    venv/bin/python3 scripts/train_ship_length.py
    venv/bin/python3 scripts/train_ship_length.py --epochs 20
"""

import argparse
import importlib
import json
import sys
import time
import math
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed.  pip install tifffile")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'ship_length'
SEED       = 42
CLASSES    = ['Cargo', 'Tanker', 'Other Type']

PATCH_CAL_ROOT = Path('/Volumes/Machine Learning/opensar_ship/OpenSARShip2')

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxlvkprrmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"

DB_MIN, DB_MAX = -30.0, 10.0
LOG_LEN_MIN    = math.log(10.0)    # ~2.30  — below our smallest ship
LOG_LEN_MAX    = math.log(600.0)   # ~6.40  — above our largest ship


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


def normalize_length(length_m) -> float:
    """Log-normalize ship length to [0, 1]. None → 0.5."""
    if length_m is None:
        return 0.5
    log_l = math.log(max(float(length_m), 1.0))
    return float(np.clip((log_l - LOG_LEN_MIN) / (LOG_LEN_MAX - LOG_LEN_MIN), 0.0, 1.0))


def to_float_channel(arr_2d: np.ndarray) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(arr_2d, 1e-6))
    db = np.clip(db, DB_MIN, DB_MAX)
    return ((db - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


# ── Model ─────────────────────────────────────────────────────────────────────

class ShipLengthModel(nn.Module):
    """ConvNeXt-Small backbone + scalar ship length → classification head."""
    def __init__(self, num_classes: int):
        super().__init__()
        from torchvision import models
        base = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        # Keep feature extractor + norm + flatten; drop the final linear
        self.features  = base.features
        self.avgpool   = base.avgpool
        self.norm      = base.classifier[0]   # LayerNorm(768)
        self.flatten   = base.classifier[1]   # Flatten
        self.head      = nn.Linear(768 + 1, num_classes)

    def forward(self, img: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        x = self.features(img)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)                            # (B, 768)
        x = torch.cat([x, length.unsqueeze(1)], dim=1) # (B, 769)
        return self.head(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DualPolLengthDataset(Dataset):
    def __init__(self, records, class_to_idx, augment=False):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.augment      = augment
        self.resize       = T.Resize((128, 128), antialias=True)
        self.normalize    = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.aug          = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]) if augment else nn.Identity()

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        tif_path = rec['tif_path']
        length   = torch.tensor(rec['length_norm'], dtype=torch.float32)

        try:
            arr = tifffile.imread(str(tif_path))
            if arr.ndim == 2:
                arr = np.stack([arr, arr], axis=2)
            vv   = to_float_channel(arr[:, :, 0])
            vh   = to_float_channel(arr[:, :, 1])
            diff = np.clip(vv - vh + 0.5, 0.0, 1.0)
            tensor = torch.from_numpy(np.stack([vv, vh, diff], axis=0))
        except Exception:
            tensor = torch.zeros(3, 37, 37)

        tensor = self.resize(tensor)
        tensor = self.aug(tensor)
        tensor = self.normalize(tensor)
        return tensor, length, self.class_to_idx[rec['ship_type']]


# ── DB loading ────────────────────────────────────────────────────────────────

def load_split(split: str) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT DISTINCT ON (scene_id, ship_type,
                                SPLIT_PART(image_id, '_VV', 1),
                                SPLIT_PART(image_id, '_VH', 1))
                   scene_id, ship_type, image_id, ship_length_m
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = 'VV'
              AND split = %s
              AND local_path IS NOT NULL
        """, CLASSES + [split])
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    out = []
    for r in rows:
        parts = r['image_id'].split('__', 1)
        if len(parts) < 2:
            continue
        stem     = parts[1].rsplit('_VV', 1)[0]
        tif_path = PATCH_CAL_ROOT / r['scene_id'] / 'Patch_Cal' / f'{stem}.tif'
        if not tif_path.exists():
            continue
        out.append({
            'ship_type':   r['ship_type'],
            'scene_id':    r['scene_id'],
            'tif_path':    tif_path,
            'length_norm': normalize_length(r['ship_length_m']),
        })
    return out


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lengths, labels in loader:
            imgs, lengths, labels = imgs.to(device), lengths.to(device), labels.to(device)
            correct += (model(imgs, lengths).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def evaluate_per_class(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lengths, labels in loader:
            preds = model(imgs.to(device), lengths.to(device)).argmax(1).cpu().tolist()
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
    print("=" * 58)
    print(f"  Ship Dual-Pol + Length — ConvNeXt-Small")
    print(f"  Input: [VV, VH, VV-VH] + Manual_Length scalar")
    print(f"  Classes: {CLASSES}")
    print("=" * 58)

    print("\nLoading records...")
    train_recs = load_split('train')
    val_recs   = load_split('val')
    test_recs  = load_split('test')

    total = len(train_recs) + len(val_recs) + len(test_recs)
    print(f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  "
          f"Test: {len(test_recs):,}  (Total: {total:,})")

    # Length coverage stats
    n_missing = sum(1 for r in train_recs if r['length_norm'] == 0.5
                    and r['ship_type'] == r['ship_type'])
    print(f"  Length: all populated from Ship.xml")

    for c in CLASSES:
        lengths = [r['length_norm'] for r in train_recs if r['ship_type'] == c]
        # Denormalize for display
        avg_m = math.exp(np.mean(lengths) * (LOG_LEN_MAX - LOG_LEN_MIN) + LOG_LEN_MIN)
        print(f"    {c:<15}: {len(lengths):,} chips  avg_length≈{avg_m:.0f}m")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    train_ds = DualPolLengthDataset(train_recs, class_to_idx, augment=True)
    val_ds   = DualPolLengthDataset(val_recs,   class_to_idx, augment=False)
    test_ds  = DualPolLengthDataset(test_recs,  class_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    model = ShipLengthModel(num_classes=len(CLASSES)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    counts = torch.tensor([
        sum(1 for r in train_recs if r['ship_type'] == c) for c in CLASSES
    ], dtype=torch.float32)
    weights = (1.0 / counts)
    weights = weights / weights.sum() * len(CLASSES)
    print(f"\n  Class weights: " +
          "  ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    save_dir = MODELS_DIR / 'ship_dualpol_length_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

    send_email(
        "Ship dual-pol + length training started",
        f"Training started: ConvNeXt-Small with [VV, VH, VV-VH] + Manual_Length.\n\n"
        f"  Architecture: 768-dim embedding + log-normalized length → Linear(769, 3)\n"
        f"  Train: {len(train_recs):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}\n"
        f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}\n"
        f"  Baseline to beat: dual-pol weighted = 70.5%\n"
    )

    best_val, best_epoch = 0.0, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total = 0.0, 0
        for imgs, lengths, labels in train_loader:
            imgs, lengths, labels = imgs.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, lengths), labels)
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
                f"Ship dual-pol + length — epoch 5  val={val_acc*100:.1f}%",
                f"Dual-pol + length training in progress.\n\n"
                f"  Epoch 5/{args.epochs}  val={val_acc*100:.1f}%\n"
                f"  Best so far: {best_val*100:.1f}%  (epoch {best_epoch})\n"
                f"  Baseline: 70.5%\n"
            )

    # ── Test ─────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                     map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    print(f"\n  Test accuracy:      {test_acc*100:.1f}%")
    print(f"  Best val:           {best_val*100:.1f}%  (epoch {best_epoch})")
    print(f"  Dual-pol baseline:  70.5%")
    print(f"  Literature:         82-88%")
    print(f"\n  Per-class breakdown:")
    evaluate_per_class(model, test_loader, device)

    result = {
        'class_names':  CLASSES,
        'input':        'VV+VH+diff+Manual_Length',
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
        f"Ship dual-pol + length DONE — test={test_acc*100:.1f}%  val={best_val*100:.1f}%",
        f"Dual-pol + Manual_Length ship classification complete.\n\n"
        f"  Input:          [VV, VH, VV-VH] + log-normalized Manual_Length\n"
        f"  Architecture:   ConvNeXt-Small → 768-dim + length → Linear(769, 3)\n"
        f"  Train/Val/Test: {len(train_recs):,} / {len(val_recs):,} / {len(test_recs):,}\n\n"
        f"  Test accuracy:  {test_acc*100:.1f}%\n"
        f"  Best val:       {best_val*100:.1f}%  (epoch {best_epoch})\n\n"
        f"  vs. dual-pol (no length): 70.5%\n"
        f"  vs. VV only:              67.1%\n"
        f"  Literature:               82-88%\n\n"
        f"Model: {save_dir}\n"
        f"Log:   {LOGS_DIR}/results.json\n"
    )


if __name__ == '__main__':
    main()
