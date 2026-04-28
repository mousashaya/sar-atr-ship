"""
scripts/train_translator.py
============================
Trains a U-Net to translate SAR chips from one depression angle to 15°.

Uses SOC paired data: same vehicle, same azimuth, source angle + 15°.
No vehicle labels needed — purely image-to-image translation.

Train one translator per source angle (3 total):
    python3 scripts/train_translator.py --source-angle 30 --name translator_v1
    python3 scripts/train_translator.py --source-angle 45 --name translator_v1
    python3 scripts/train_translator.py --source-angle 60 --name translator_v1

Or all three in one call:
    python3 scripts/train_translator.py --all-angles --name translator_v1
"""

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import psycopg2
import psycopg2.extras
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

SOURCE_ANGLES = [30, 45, 60]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class PairedDataset(Dataset):
    """
    Each item: (source_chip, target_chip) where
      source_chip = SAR image at source_angle
      target_chip = same vehicle, same azimuth, at 15°
    """
    def __init__(self, pairs, transform=None):
        self.pairs     = pairs   # list of (source_path, target_path)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, tgt_path = self.pairs[idx]
        try:
            src = Image.open(src_path).convert('RGB')
        except Exception:
            src = Image.new('RGB', (128, 128), color=(64, 64, 64))
        try:
            tgt = Image.open(tgt_path).convert('RGB')
        except Exception:
            tgt = Image.new('RGB', (128, 128), color=(64, 64, 64))
        if self.transform:
            src = self.transform(src)
            tgt = self.transform(tgt)
        return src, tgt


def load_pairs(source_angle: int, seed: int = 42):
    """Query SOC paired chips: (source_angle chip, 15° chip) for same vehicle+azimuth."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT a.local_path AS src_path, b.local_path AS tgt_path
            FROM defense.atrnet_images a
            JOIN defense.atrnet_images b
              ON a.target_type   = b.target_type
             AND a.azimuth_angle = b.azimuth_angle
             AND a.band          = b.band
            WHERE a.experiment_setting = 'SOC_40classes'
              AND b.experiment_setting = 'SOC_40classes'
              AND a.depression_angle = %s
              AND b.depression_angle = 15
              AND a.band = 'X'
              AND a.split = 'train'
              AND b.split = 'train'
              AND a.local_path IS NOT NULL
              AND b.local_path IS NOT NULL
        """, (source_angle,))
        rows = cur.fetchall()
    conn.close()

    pairs = [(r['src_path'], r['tgt_path']) for r in rows]

    # 15% val split
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.15))
    return pairs[n_val:], pairs[:n_val]


def train_one_angle(source_angle: int, args, device, transform):
    print(f"\n{'='*60}")
    print(f"  Translator: {source_angle}° → 15°")
    print(f"{'='*60}")

    train_pairs, val_pairs = load_pairs(source_angle)
    print(f"  Train pairs: {len(train_pairs):,}  Val pairs: {len(val_pairs):,}")

    train_ds = PairedDataset(train_pairs, transform)
    val_ds   = PairedDataset(val_pairs,   transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    unet_module = importlib.import_module('models.unet')
    model       = unet_module.build_translator().to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion   = nn.L1Loss()

    save_dir = (Path(__file__).parent.parent / 'models' / 'saved' /
                f'translator_{args.name}_{source_angle}to15')
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = (Path(__file__).parent.parent / 'logs' /
                f'train_translator_{args.name}_{source_angle}to15.log')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Translator {source_angle}°→15°  —  {args.name}")
    log(f"Train: {len(train_pairs):,}  Val: {len(val_pairs):,}")

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(src)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                val_loss += criterion(model(src), tgt).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - start
        log(f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={total_loss/len(train_loader):.4f}  "
            f"val_loss={val_loss:.4f}  ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'weights.pt')

    # Save a few visual examples
    _save_examples(model, val_loader, device, save_dir, source_angle, n=8)

    config = {
        'name':          args.name,
        'task':          'depression_translation',
        'source_angle':  source_angle,
        'target_angle':  15,
        'best_val_loss': round(best_val_loss, 4),
        'date':          datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"Best val L1 loss: {best_val_loss:.4f}")
    log(f"Weights saved to {save_dir}")
    log_file.close()
    return best_val_loss


def _save_examples(model, val_loader, device, save_dir, source_angle, n=8):
    """Save side-by-side: source | translated | target."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        model.eval()
        src_batch, tgt_batch = next(iter(val_loader))
        src_batch = src_batch[:n].to(device)
        with torch.no_grad():
            pred_batch = model(src_batch)

        # Denormalize from [-1,1] to [0,1]
        def denorm(t):
            return (t * 0.5 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        srcs  = denorm(src_batch)
        preds = denorm(pred_batch)
        tgts  = denorm(tgt_batch[:n])

        fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
        for i in range(n):
            axes[i, 0].imshow(srcs[i],  cmap='gray')
            axes[i, 0].set_title(f'Input {source_angle}°', fontsize=8)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(preds[i], cmap='gray')
            axes[i, 1].set_title('Translated → 15°', fontsize=8)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(tgts[i],  cmap='gray')
            axes[i, 2].set_title('True 15°', fontsize=8)
            axes[i, 2].axis('off')
        plt.tight_layout()
        out = save_dir / f'examples_{source_angle}to15.png'
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"  Visual examples saved: {out}")
    except Exception as e:
        print(f"  Could not save examples: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',         required=True,
                        help='Base name, e.g. translator_v1')
    parser.add_argument('--source-angle', type=int, default=None,
                        choices=[30, 45, 60],
                        help='Which depression angle to translate FROM')
    parser.add_argument('--all-angles',   action='store_true',
                        help='Train all 3 translators (30→15, 45→15, 60→15) sequentially')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=0.0002)
    parser.add_argument('--num-workers',  type=int,   default=4)
    args = parser.parse_args()

    if args.source_angle is None and not args.all_angles:
        print("Error: specify --source-angle or --all-angles")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    unet_module = importlib.import_module('models.unet')
    transform   = unet_module.get_transform()

    angles = SOURCE_ANGLES if args.all_angles else [args.source_angle]
    summary = {}
    for angle in angles:
        loss = train_one_angle(angle, args, device, transform)
        summary[angle] = round(loss, 4)

    if len(angles) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — all translators")
        print(f"{'='*60}")
        for angle, loss in summary.items():
            print(f"  {angle}°→15°:  best val L1 = {loss:.4f}")


if __name__ == '__main__':
    main()
