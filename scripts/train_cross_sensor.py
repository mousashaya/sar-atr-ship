"""
scripts/train_cross_sensor.py
==============================
Cross-sensor generalization experiment: GF-3 (FUSAR) ↔ Sentinel-1 (OpenSARShip)

Three modes:
  baseline   — train and test within the same dataset (upper bound)
  cross      — train on source, test on target (cross-sensor gap)
  finetune   — pretrain on source, few-shot fine-tune on target, test on target

Aligned 4-class label set: Cargo, Tanker, Fishing, Tug

Usage:
    python scripts/train_cross_sensor.py --mode baseline --source fusar
    python scripts/train_cross_sensor.py --mode baseline --source opensar
    python scripts/train_cross_sensor.py --mode cross --source fusar
    python scripts/train_cross_sensor.py --mode cross --source opensar
    python scripts/train_cross_sensor.py --mode finetune --source fusar --shots 5
    python scripts/train_cross_sensor.py --mode finetune --source fusar --shots 20
    python scripts/train_cross_sensor.py --mode finetune --source fusar --shots 50
"""

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
import tifffile
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
LOGS_DIR   = Path(__file__).parent.parent / 'logs' / 'cross_sensor'
SEED       = 42

# 4 aligned classes present in both datasets
CLASSES = ['Cargo', 'Tanker', 'Fishing', 'Tug']

# Map OpenSARShip label variants to canonical names
OPENSAR_LABEL_MAP = {
    'Cargo':    'Cargo',
    'Tanker':   'Tanker',
    'Fishing':  'Fishing',
    'Tug':      'Tug',
}

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
    except Exception as e:
        print(f"  Email failed: {e}")


def get_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')


def load_fusar(split=None):
    """Load FUSAR records for the 4 aligned classes."""
    conn = psycopg2.connect(**DB_CONFIG)
    ph   = ','.join(['%s'] * len(CLASSES))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        sql = f"""
            SELECT id, local_path, category AS label
            FROM defense.fusar_images
            WHERE category IN ({ph})
              AND local_path IS NOT NULL
        """
        params = CLASSES[:]
        if split:
            sql += " AND split = %s"
            params.append(split)
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    # normalize label
    for r in rows:
        r['label'] = r['label'].strip()
    return rows


def load_opensar(split=None):
    """Load OpenSARShip records for the 4 aligned classes, VV polarization."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        ph  = ','.join(['%s'] * len(CLASSES))
        sql = f"""
            SELECT id, local_path, ship_type AS label
            FROM defense.opensar_images
            WHERE ship_type IN ({ph})
              AND polarization = 'VV'
              AND local_path IS NOT NULL
        """
        params = CLASSES[:]
        if split:
            sql += " AND split = %s"
            params.append(split)
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r['label'] = OPENSAR_LABEL_MAP.get(r['label'].strip(), r['label'].strip())
    return rows


def load_image(path):
    """Load SAR image from either TIFF (FUSAR) or standard format (OpenSARShip)."""
    path = str(path)
    try:
        if path.lower().endswith(('.tif', '.tiff')):
            arr = tifffile.imread(path)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            if arr.ndim == 2:
                img = Image.fromarray(arr).convert('RGB')
            else:
                img = Image.fromarray(arr[:, :, 0]).convert('RGB')
        else:
            img = Image.open(path).convert('RGB')
    except Exception:
        img = Image.new('RGB', (128, 128), 64)
    return img


class SARDataset(Dataset):
    def __init__(self, records, class_to_idx, transform=None):
        self.records      = records
        self.class_to_idx = class_to_idx
        self.transform    = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = load_image(rec['local_path'])
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[rec['label']]


def sample_few_shot(records, n_shots, seed=42):
    """Sample n_shots examples per class for few-shot fine-tuning."""
    rng = random.Random(seed)
    by_class = {c: [] for c in CLASSES}
    for r in records:
        if r['label'] in by_class:
            by_class[r['label']].append(r)
    sampled = []
    for c, recs in by_class.items():
        sampled.extend(rng.sample(recs, min(n_shots, len(recs))))
    return sampled


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += len(labels)
    return correct / total if total else 0.0


def evaluate_full(model, loader, device, class_names):
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
    return all_preds, all_labels


def train_epochs(model, loader, val_loader, optimizer, criterion,
                 device, epochs, label, save_path):
    best_val, best_epoch = 0.0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total      += len(labels)
        val_acc = evaluate(model, val_loader, device)
        print(f"  [{label}] Epoch {ep}/{epochs}  "
              f"loss={total_loss/total:.4f}  val={val_acc*100:.1f}%  "
              f"({int(time.time()-t0)}s)")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, ep
            torch.save(model.state_dict(), save_path)
    return best_val, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',    choices=['baseline', 'cross', 'finetune'],
                        default='baseline')
    parser.add_argument('--source',  choices=['fusar', 'opensar'], default='fusar')
    parser.add_argument('--epochs',  type=int, default=20)
    parser.add_argument('--ft-epochs', type=int, default=10,
                        help='Fine-tuning epochs (finetune mode only)')
    parser.add_argument('--shots',   type=int, default=20,
                        help='Shots per class for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--ft-lr',   type=float, default=1e-5,
                        help='Learning rate for fine-tuning')
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    device       = get_device()
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    target       = 'opensar' if args.source == 'fusar' else 'fusar'

    print(f"\nDevice: {device}")
    print("=" * 60)
    print(f"  Cross-Sensor Experiment")
    print(f"  Mode:    {args.mode}")
    print(f"  Source:  {args.source.upper()}")
    print(f"  Target:  {target.upper()}")
    print(f"  Classes: {CLASSES}")
    print("=" * 60)

    loader_fn = {'fusar': load_fusar, 'opensar': load_opensar}

    # Load source data
    src_train = loader_fn[args.source]('train')
    src_val   = loader_fn[args.source]('val') or loader_fn[args.source]('test')
    src_test  = loader_fn[args.source]('test')

    # Load target data
    tgt_train = loader_fn[target]('train')
    tgt_test  = loader_fn[target]('test')
    tgt_val   = loader_fn[target]('val') or tgt_test

    print(f"\n  Source ({args.source}) — train: {len(src_train):,}  "
          f"val: {len(src_val):,}  test: {len(src_test):,}")
    for c in CLASSES:
        n = sum(1 for r in src_train if r['label'] == c)
        print(f"    {c:<12}: {n:,}")

    print(f"\n  Target ({target}) — train: {len(tgt_train):,}  "
          f"test: {len(tgt_test):,}")
    for c in CLASSES:
        n = sum(1 for r in tgt_train + tgt_test if r['label'] == c)
        print(f"    {c:<12}: {n:,}")

    mod           = importlib.import_module('models.convnext_small')
    transform     = mod.get_train_transform()
    val_transform = mod.get_transform()
    criterion     = nn.CrossEntropyLoss()

    run_tag  = f"{args.mode}_{args.source}_to_{target}"
    save_dir = MODELS_DIR / f'cross_sensor_{run_tag}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Train on source ──────────────────────────────────────────
    print(f"\n── Phase 1: Training on {args.source.upper()} ──")
    model = mod.build_model(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        SARDataset(src_train, class_to_idx, transform),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        SARDataset(src_val, class_to_idx, val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_val, best_epoch = train_epochs(
        model, train_loader, val_loader, optimizer, criterion,
        device, args.epochs, args.source.upper(), save_dir / 'phase1_weights.pt')

    model.load_state_dict(torch.load(save_dir / 'phase1_weights.pt',
                                      map_location=device, weights_only=True))

    # Evaluate on source test (baseline upper bound)
    src_test_loader = DataLoader(
        SARDataset(src_test, class_to_idx, val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    src_test_acc = evaluate(model, src_test_loader, device)
    print(f"\n  Source test accuracy ({args.source.upper()}): {src_test_acc*100:.1f}%")

    # ── Evaluate directly on target (cross-sensor gap) ────────────────────
    tgt_test_loader = DataLoader(
        SARDataset(tgt_test, class_to_idx, val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=4)
    cross_acc = evaluate(model, tgt_test_loader, device)
    print(f"  Cross-sensor accuracy ({target.upper()}): {cross_acc*100:.1f}%")
    print(f"  Resolution gap penalty: {(src_test_acc - cross_acc)*100:.1f}%")

    print(f"\n  Per-class breakdown (cross-sensor):")
    evaluate_full(model, tgt_test_loader, device, CLASSES)

    result = {
        'mode':             args.mode,
        'source':           args.source,
        'target':           target,
        'classes':          CLASSES,
        'src_train_size':   len(src_train),
        'src_test_size':    len(src_test),
        'tgt_test_size':    len(tgt_test),
        'source_test_acc':  round(src_test_acc * 100, 1),
        'cross_acc':        round(cross_acc * 100, 1),
        'gap':              round((src_test_acc - cross_acc) * 100, 1),
    }

    # ── Phase 2: Few-shot fine-tuning (finetune mode only) ────────────────
    if args.mode == 'finetune':
        print(f"\n── Phase 2: Fine-tuning on {args.shots}-shot {target.upper()} ──")
        ft_records = sample_few_shot(tgt_train, args.shots)
        print(f"  Fine-tune samples: {len(ft_records)} "
              f"({args.shots} shots × {len(CLASSES)} classes)")

        ft_loader = DataLoader(
            SARDataset(ft_records, class_to_idx, transform),
            batch_size=min(args.batch_size, len(ft_records)),
            shuffle=True, num_workers=2)
        tgt_val_loader = DataLoader(
            SARDataset(tgt_val, class_to_idx, val_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=4)

        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr)
        ft_best, ft_epoch = train_epochs(
            model, ft_loader, tgt_val_loader, ft_optimizer, criterion,
            device, args.ft_epochs, f"FT-{args.shots}shot",
            save_dir / 'phase2_weights.pt')

        model.load_state_dict(torch.load(save_dir / 'phase2_weights.pt',
                                          map_location=device, weights_only=True))
        ft_acc = evaluate(model, tgt_test_loader, device)
        print(f"\n  Post-finetune accuracy ({target.upper()}): {ft_acc*100:.1f}%")
        print(f"  Recovery from fine-tuning: +{(ft_acc - cross_acc)*100:.1f}%")

        print(f"\n  Per-class breakdown (after fine-tuning):")
        evaluate_full(model, tgt_test_loader, device, CLASSES)

        result.update({
            'shots':        args.shots,
            'ft_acc':       round(ft_acc * 100, 1),
            'ft_recovery':  round((ft_acc - cross_acc) * 100, 1),
        })

    # Save results
    out_file = LOGS_DIR / f'results_{run_tag}.json'
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved → {out_file}")

    send_email(
        f"Cross-sensor [{args.mode}] {args.source}→{target}  "
        f"cross={cross_acc*100:.1f}%",
        f"Cross-sensor experiment complete.\n\n"
        f"  Mode:    {args.mode}\n"
        f"  Source:  {args.source.upper()} → Target: {target.upper()}\n"
        f"  Classes: {CLASSES}\n\n"
        f"  Source test acc:  {src_test_acc*100:.1f}%\n"
        f"  Cross-sensor acc: {cross_acc*100:.1f}%\n"
        f"  Resolution gap:   {(src_test_acc-cross_acc)*100:.1f}%\n"
        + (f"  After fine-tune:  {result.get('ft_acc','—')}%  "
           f"(+{result.get('ft_recovery','—')}%)\n"
           if args.mode == 'finetune' else '')
        + f"\nResults → {out_file}\n"
    )


if __name__ == '__main__':
    main()
