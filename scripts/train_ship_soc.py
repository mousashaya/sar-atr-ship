"""
scripts/train_ship_soc.py
==========================
ConvNeXt-Small baseline for OpenSARShip ship-type classification.

3 classes: Cargo, Tanker, Other Type  (VV polarization)
Train/val/test split: 70/15/15 stratified by class.

Literature reports 60-80% on this 3-class problem.
Cargo vs Tanker is the hard pair.

Usage:
    python3 scripts/train_ship_soc.py
    python3 scripts/train_ship_soc.py --epochs 20
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR   = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR     = Path(__file__).parent.parent / 'logs' / 'ship_soc'
SEED         = 42
CLASSES      = ['Cargo', 'Tanker', 'Other Type']
POLARIZATION = 'VV'

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxlvkprrmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"


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


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_records_by_split(split: str) -> list:
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT local_path, ship_type, scene_id
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = %s
              AND split = %s
              AND local_path IS NOT NULL
        """, CLASSES + [POLARIZATION, split])
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


class ShipDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), 64)
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[rec['ship_type']]


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def evaluate_per_class(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names, digits=3))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    header = ''.join(f'{c[:8]:>10}' for c in class_names)
    print(f"{'':>12}{header}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:10]:<12}" + ''.join(f'{v:>10}' for v in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=15)
    parser.add_argument('--batch-size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--polarization', type=str,   default='VV',
                        choices=['VV', 'VH', 'VV+VH'])
    args = parser.parse_args()
    global POLARIZATION
    POLARIZATION = args.polarization

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"\nDevice: {device}")
    print("=" * 55)
    print(f"  Ship SOC Baseline — ConvNeXt-Small")
    print(f"  Classes: {CLASSES}")
    print(f"  Polarization: {POLARIZATION}")
    print("=" * 55)

    # Scene-level splits assigned during ingest — no leakage
    train_recs = load_records_by_split('train')
    val_recs   = load_records_by_split('val')
    test_recs  = load_records_by_split('test')

    total = len(train_recs) + len(val_recs) + len(test_recs)
    print(f"\n  Total records: {total:,}")
    for c in CLASSES:
        n = sum(1 for r in train_recs + val_recs + test_recs if r['ship_type'] == c)
        print(f"    {c:<15}: {n:,}")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print(f"\n  Train: {len(train_recs):,}  Val: {len(val_recs):,}  Test: {len(test_recs):,}")

    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    model         = mod.build_model(num_classes=len(CLASSES)).to(device)

    train_loader = DataLoader(ShipDataset(train_recs, class_to_idx, transform),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader   = DataLoader(ShipDataset(val_recs,   class_to_idx, val_transform),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True)
    test_loader  = DataLoader(ShipDataset(test_recs,  class_to_idx, val_transform),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    pol_tag  = args.polarization.replace('+', '_')
    save_dir = MODELS_DIR / f'ship_soc_{pol_tag}_v1'
    save_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  Epoch {ep}/{args.epochs}  loss={total_loss/total:.4f}"
              f"  val={val_acc*100:.1f}%  ({int(time.time()-t0)}s)")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, ep
            torch.save(model.state_dict(), save_dir / 'weights.pt')
        if ep == 5:
            send_email(
                f"Ship SOC — epoch 5 check-in  val={val_acc*100:.1f}%",
                f"Ship SOC training in progress.\n\n"
                f"  Epoch 5/{args.epochs}  val={val_acc*100:.1f}%\n"
                f"  Best so far: {best_val*100:.1f}%\n"
                f"  Literature target: 60-80%\n")

    model.load_state_dict(torch.load(save_dir / 'weights.pt',
                                     map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    print(f"\n  Test accuracy:    {test_acc*100:.1f}%")
    print(f"  Best val:         {best_val*100:.1f}%  (epoch {best_epoch})")
    print(f"  Literature range: 60-80%")

    print(f"\n  Per-class breakdown:")
    evaluate_per_class(model, test_loader, device, CLASSES)

    result = {
        'class_names':   CLASSES,
        'polarization':  POLARIZATION,
        'train_size':    len(train_recs),
        'val_size':      len(val_recs),
        'test_size':     len(test_recs),
        'best_val_acc':  round(best_val * 100, 1),
        'test_acc':      round(test_acc * 100, 1),
        'epochs':        args.epochs,
    }
    (save_dir / 'config.json').write_text(json.dumps(result, indent=2))
    (LOGS_DIR / 'results.json').write_text(json.dumps(result, indent=2))
    print(f"\nModel saved to {save_dir}")

    send_email(
        f"Ship SOC done — test={test_acc*100:.1f}%  val={best_val*100:.1f}%",
        f"Ship type classification baseline complete.\n\n"
        f"  Classes:          {CLASSES}\n"
        f"  Polarization:     {POLARIZATION}\n"
        f"  Train/Val/Test:   {len(train_recs):,} / {len(val_recs):,} / {len(test_recs):,}\n\n"
        f"  Test accuracy:    {test_acc*100:.1f}%\n"
        f"  Best val:         {best_val*100:.1f}%  (epoch {best_epoch})\n"
        f"  Literature range: 60-80%\n\n"
        f"Results saved to {LOGS_DIR}/results.json\n"
        f"Model saved to   {save_dir}/\n"
    )


if __name__ == '__main__':
    main()
