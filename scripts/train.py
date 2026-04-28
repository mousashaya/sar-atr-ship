"""
scripts/train.py
================
Trains a model on ATRNet-STAR and records everything to the DB.

Each run creates one row in defense.experiments with a unique ID.
Trained weights are saved to models/saved/exp_{ID}_{name}/weights.pt
All predictions on val and test sets are saved to defense.experiment_predictions.
Summary metrics saved to defense.experiment_metrics.

Usage:
    python3 scripts/train.py --model efficientnet_b2 --config 4class --name "baseline_4class_v1"
    python3 scripts/train.py --model efficientnet_b2 --config 40class --crop-bbox --name "40class_cropped_v1"
    python3 scripts/train.py --model dinov2 --config 4class --name "dinov2_4class_v1"

Arguments:
    --model          Model module name in models/ (e.g. efficientnet_b2, dinov2)
    --config         Classification level: 4class, 21class, 40class
    --setting        ATRNet experimental setting (default: SOC_40classes)
    --epochs         Number of training epochs (default: 10)
    --batch-size     Images per batch (default: 32)
    --lr             Learning rate (default: 0.001)
    --max-per-class  Cap images per class (default: no limit)
    --crop-bbox      Crop image to bounding box before training
    --augment        Apply rotation/flip augmentation to training set
    --use-atrnet-splits  Use ATRNet's predefined train/test split
    --num-workers    DataLoader worker processes (default: 4)
    --name           Short label for this run (required)
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Load .env from project root into os.environ (no external dependency needed)
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG
from dataset import build_datasets


# ── Email notification ────────────────────────────────────────────────────────

def send_email(to: str, subject: str, body: str):
    import smtplib, os
    from email.mime.text import MIMEText
    gmail_user     = os.environ.get('GMAIL_USER', '')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', '')
    if not gmail_user or not gmail_password:
        print("  Email skipped — GMAIL_USER / GMAIL_APP_PASSWORD not set in .env")
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = gmail_user
        msg['To']      = to
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(gmail_user, gmail_password)
            smtp.sendmail(gmail_user, to, msg.as_string())
        print(f"  Email sent to {to}")
    except Exception as e:
        print(f"  Email failed: {e}")


# ── Device selection ──────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── DB helpers ────────────────────────────────────────────────────────────────

def create_experiment(conn, args, model_config: dict, class_names: list) -> int:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO defense.experiments
                (name, model_type, config_name, epochs,
                 learning_rate, batch_size, crop_bbox, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            args.name,
            model_config.get('model_type'),
            args.config,
            args.epochs,
            args.lr,
            args.batch_size,
            args.crop_bbox,
            f"setting={args.setting}  classes={class_names}  "
            f"max_per_class={args.max_per_class}  "
            f"augment={args.augment}  crop_bbox={args.crop_bbox}  "
            f"use_atrnet_splits={args.use_atrnet_splits}",
        ))
        exp_id = cur.fetchone()[0]
    conn.commit()
    return exp_id


def save_epoch(conn, exp_id: int, epoch: int,
               train_loss: float, val_acc: float, elapsed: int):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO defense.experiment_epochs
                (experiment_id, epoch, train_loss, val_acc, elapsed_seconds)
            VALUES (%s, %s, %s, %s, %s)
        """, (exp_id, epoch, train_loss, val_acc, elapsed))
    conn.commit()


def save_weights_path(conn, exp_id: int, weights_path: str, config_path: str):
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE defense.experiments
            SET weights_path = %s, config_path = %s WHERE id = %s
        """, (weights_path, config_path, exp_id))
    conn.commit()


def save_predictions(conn, exp_id: int, split: str,
                     records: list, predictions: list, confidences: list):
    rows = [(exp_id, rec['image_id'], split,
             rec['true_class'], pred, float(conf))
            for rec, pred, conf in zip(records, predictions, confidences)]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO defense.experiment_predictions
                (experiment_id, image_id, split, true_class, predicted_class, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, rows)
    conn.commit()


def save_metrics(conn, exp_id: int, split: str,
                 y_true: list, y_pred: list, class_names: list):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    rows = []
    for cls in class_names:
        m = report.get(cls, {})
        rows.append((exp_id, split, cls,
                     m.get('precision'), m.get('recall'),
                     m.get('f1-score'), m.get('support'), None))
    rows.append((exp_id, split, None,
                 report.get('weighted avg', {}).get('precision'),
                 report.get('weighted avg', {}).get('recall'),
                 report.get('weighted avg', {}).get('f1-score'),
                 report.get('weighted avg', {}).get('support'),
                 report.get('accuracy')))
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO defense.experiment_metrics
                (experiment_id, split, class_name,
                 precision, recall, f1, support, accuracy)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, rows)
    conn.commit()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, class_names):
    model.eval()
    all_records, all_preds, all_confs, all_true = [], [], [], []
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images  = images.to(device)
            logits  = model(images)
            probs   = torch.softmax(logits, dim=1)
            confs, pred_idxs = probs.max(dim=1)
            for i in range(len(image_ids)):
                all_records.append({'image_id':   int(image_ids[i]),
                                    'true_class': class_names[labels[i]]})
                all_preds.append(class_names[pred_idxs[i]])
                all_confs.append(float(confs[i]))
                all_true.append(class_names[labels[i]])
    return all_records, all_preds, all_confs, all_true


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train an ATRNet-STAR classifier")
    parser.add_argument("--model",         required=True)
    parser.add_argument("--config",        default='4class',
                        choices=['4class', '21class', '40class'])
    parser.add_argument("--setting",       default='SOC_40classes',
                        help="ATRNet experimental setting (default: SOC_40classes)")
    parser.add_argument("--angle",         type=float, default=None,
                        help="Train on one depression angle only (e.g. 15, 30, 45, 60)")
    parser.add_argument("--parent-class",  default=None,
                        help="Cascade Level 2: restrict to images where main_class=this (e.g. Bus, Car, Special, Truck)")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--name",          required=True)
    parser.add_argument("--crop-bbox",     action="store_true",
                        help="Crop image to bounding box before training")
    parser.add_argument("--augment",       action="store_true",
                        help="Apply rotation/flip augmentation to training set")
    parser.add_argument("--use-atrnet-splits", action="store_true",
                        help="Use ATRNet predefined train/test split instead of random split")
    parser.add_argument("--cache",         action="store_true")
    parser.add_argument("--num-workers",   type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}  Setting: {args.setting}  crop_bbox={args.crop_bbox}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model_module = importlib.import_module(f"models.{args.model}")
    transform    = model_module.get_transform()
    model_config = model_module.get_config()

    train_transform = transform
    if args.augment and hasattr(model_module, 'get_train_transform'):
        train_transform = model_module.get_train_transform()
        print("Augmentation: ON")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data...")
    train_ds, val_ds, test_ds, class_names, split_records = build_datasets(
        config_name=args.config,
        max_per_class=args.max_per_class,
        transform=transform,
        train_transform=train_transform,
        cache=args.cache,
        use_atrnet_splits=args.use_atrnet_splits,
        crop_bbox=args.crop_bbox,
        setting=args.setting,
        angle=args.angle,
        parent_class=args.parent_class,
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    num_workers = 0 if args.cache else args.num_workers
    persistent  = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=persistent)

    # ── Build model ───────────────────────────────────────────────────────────
    model     = model_module.build_model(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── Register experiment ───────────────────────────────────────────────────
    conn   = psycopg2.connect(**DB_CONFIG)
    exp_id = create_experiment(conn, args, model_config, class_names)
    print(f"\nExperiment ID: {exp_id}")

    # ── Training loop ─────────────────────────────────────────────────────────
    log_path = Path(__file__).parent.parent / 'logs' / f'train_{args.name}.log'
    log_path.parent.mkdir(exist_ok=True)
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Experiment ID: {exp_id}")
    log(f"Training for {args.epochs} epochs...")

    best_val_acc = 0.0
    save_dir = Path(__file__).parent.parent / 'models' / 'saved' / f'exp_{exp_id:03d}_{args.name}'
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        log(f"Epoch {epoch}/{args.epochs} started at {datetime.now().strftime('%H:%M:%S')} ...")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        _, val_preds, _, val_true = evaluate(model, val_loader, device, class_names)
        val_acc = sum(p == t for p, t in zip(val_preds, val_true)) / len(val_true)

        elapsed = int(time.time() - start)
        log(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
            f"val_acc={val_acc*100:.1f}%  ({elapsed}s)")
        save_epoch(conn, exp_id, epoch, train_loss, val_acc, elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    log(f"\nBest val accuracy: {best_val_acc*100:.1f}%")

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        **model_config,
        'experiment_id': exp_id,
        'name':          args.name,
        'config_name':   args.config,
        'setting':       args.setting,
        'crop_bbox':     args.crop_bbox,
        'epochs':        args.epochs,
        'batch_size':    args.batch_size,
        'learning_rate': args.lr,
        'class_names':   class_names,
        'best_val_acc':  round(best_val_acc * 100, 1),
        'date':          datetime.now().isoformat(),
    }
    config_path = save_dir / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    save_weights_path(conn, exp_id, str(save_dir / 'weights.pt'), str(config_path))

    # ── Final evaluation ──────────────────────────────────────────────────────
    log("\nEvaluating on val and test sets...")
    model.load_state_dict(torch.load(save_dir / 'weights.pt', map_location=device))

    email_lines = [
        f"Experiment {exp_id} — {args.name}",
        f"Config: {args.config}  Setting: {args.setting}  crop_bbox: {args.crop_bbox}",
        f"Best val accuracy: {best_val_acc*100:.1f}%",
        "",
    ]

    for split, loader, records in [
        ('val',  val_loader,  split_records['val']),
        ('test', test_loader, split_records['test']),
    ]:
        recs, preds, confs, true = evaluate(model, loader, device, class_names)
        save_predictions(conn, exp_id, split, recs, preds, confs)
        save_metrics(conn, exp_id, split, true, preds, class_names)

        acc = sum(p == t for p, t in zip(preds, true)) / len(true)
        cm  = confusion_matrix(true, preds, labels=class_names)

        log(f"\n  {split.upper()} accuracy: {acc*100:.1f}%")
        log(f"  Confusion matrix (rows=true, cols=predicted):")
        header = ''.join(f"{c:>16}" for c in class_names)
        log(f"  {'':>16}{header}")
        for i, row_label in enumerate(class_names):
            row = ''.join(f"{cm[i,j]:>16}" for j in range(len(class_names)))
            log(f"  {row_label:>16}{row}")

        email_lines.append(f"{split.upper()} accuracy: {acc*100:.1f}%")

    conn.close()
    log(f"\nDone. Experiment {exp_id} saved.")
    log_file.close()

    send_email(
        to=os.environ.get('NOTIFY_EMAIL', os.environ.get('GMAIL_USER', '')),
        subject=f"Defense training done — exp {exp_id} ({args.name})",
        body='\n'.join(email_lines),
    )


if __name__ == "__main__":
    import os
    main()
