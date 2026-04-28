"""
scripts/train_translator_gan.py
================================
Trains a GAN-based translator: source depression angle → 15°.

vs plain L1 translator:
  L1 alone → blurry outputs (safe pixel averages)
  GAN adds discriminator → sharper, more realistic 15° chips

Generator:  U-Net (same as train_translator.py)
Discriminator: small PatchGAN — judges 70x70 patches as real/fake
Loss: L1 (pixel fidelity) + adversarial (realism)

Quick diagnostic — 8 epochs to see if GAN improves over plain L1.

Usage:
    python3 scripts/train_translator_gan.py --source-angle 30 --name gan_v1
    python3 scripts/train_translator_gan.py --all-angles --name gan_v1 --epochs 8
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


# ── PatchGAN Discriminator ────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator — judges overlapping 70x70 patches as real/fake.
    More stable than full-image discriminator and better at local texture.
    Input: concatenated (source_chip, target_chip) — 6 channels
    Output: grid of real/fake scores (not a single number)
    """
    def __init__(self, in_channels=6):
        super().__init__()
        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64,  normalize=False),
            *block(64,  128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        return self.model(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairedDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs     = pairs
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
    rng   = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.15))
    return pairs[n_val:], pairs[:n_val]


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_angle(source_angle: int, args, device, transform):
    print(f"\n{'='*60}")
    print(f"  GAN Translator: {source_angle}° → 15°")
    print(f"{'='*60}")

    train_pairs, val_pairs = load_pairs(source_angle)
    print(f"  Train: {len(train_pairs):,}  Val: {len(val_pairs):,}")

    train_ds = PairedDataset(train_pairs, transform)
    val_ds   = PairedDataset(val_pairs,   transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, persistent_workers=True)

    unet_module = importlib.import_module('models.unet')
    generator   = unet_module.build_translator().to(device)
    discriminator = PatchDiscriminator(in_channels=6).to(device)

    opt_G = torch.optim.Adam(generator.parameters(),     lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_L1  = nn.L1Loss()
    criterion_GAN = nn.MSELoss()   # LSGAN — more stable than BCE

    save_dir = (Path(__file__).parent.parent / 'models' / 'saved' /
                f'translator_gan_{args.name}_{source_angle}to15')
    save_dir.mkdir(parents=True, exist_ok=True)

    log_path = (Path(__file__).parent.parent / 'logs' /
                f'train_gan_{args.name}_{source_angle}to15.log')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"GAN Translator {source_angle}°→15°  —  {args.name}")
    log(f"L1 weight: {args.lambda_l1}  (higher = more pixel fidelity vs realism)")

    best_val_l1 = float('inf')

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        generator.train()
        discriminator.train()

        total_G_loss = 0.0
        total_D_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            fake    = generator(src)

            # ── Train Discriminator ────────────────────────────────────────
            opt_D.zero_grad()
            # Real pair: (src, real_tgt) → should be 1
            pred_real = discriminator(src, tgt)
            loss_D_real = criterion_GAN(pred_real,
                                        torch.ones_like(pred_real))
            # Fake pair: (src, fake_tgt) → should be 0
            pred_fake = discriminator(src, fake.detach())
            loss_D_fake = criterion_GAN(pred_fake,
                                        torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ────────────────────────────────────────────
            opt_G.zero_grad()
            pred_fake_G = discriminator(src, fake)
            loss_G_adv  = criterion_GAN(pred_fake_G,
                                        torch.ones_like(pred_fake_G))
            loss_G_l1   = criterion_L1(fake, tgt) * args.lambda_l1
            loss_G      = loss_G_adv + loss_G_l1
            loss_G.backward()
            opt_G.step()

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()

        # Validation L1 (pixel quality)
        generator.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                val_l1  += criterion_L1(generator(src), tgt).item()
        val_l1 /= len(val_loader)

        elapsed = time.time() - start
        log(f"Epoch {epoch}/{args.epochs}  "
            f"G={total_G_loss/len(train_loader):.4f}  "
            f"D={total_D_loss/len(train_loader):.4f}  "
            f"val_L1={val_l1:.4f}  ({elapsed:.0f}s)")

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(generator.state_dict(), save_dir / 'weights.pt')

    # Save visual examples
    _save_examples(generator, val_loader, device, save_dir, source_angle)

    config = {
        'name':         args.name,
        'task':         'depression_translation_gan',
        'source_angle': source_angle,
        'target_angle': 15,
        'best_val_l1':  round(best_val_l1, 4),
        'lambda_l1':    args.lambda_l1,
        'date':         datetime.now().isoformat(),
    }
    (save_dir / 'config.json').write_text(json.dumps(config, indent=2))
    log(f"Best val L1: {best_val_l1:.4f}  (plain L1 baseline: ~0.116)")
    log(f"Weights saved to {save_dir}")
    log_file.close()
    return best_val_l1


def _save_examples(generator, val_loader, device, save_dir, source_angle, n=8):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        generator.eval()
        src_batch, tgt_batch = next(iter(val_loader))
        src_batch = src_batch[:n].to(device)
        with torch.no_grad():
            pred_batch = generator(src_batch)

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
            axes[i, 1].set_title('GAN → 15°', fontsize=8)
            axes[i, 1].axis('off')
            axes[i, 2].imshow(tgts[i],  cmap='gray')
            axes[i, 2].set_title('True 15°', fontsize=8)
            axes[i, 2].axis('off')
        plt.tight_layout()
        out = save_dir / f'examples_gan_{source_angle}to15.png'
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"  Visual examples saved: {out}")
    except Exception as e:
        print(f"  Could not save examples: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',         required=True)
    parser.add_argument('--source-angle', type=int, default=None,
                        choices=[30, 45, 60])
    parser.add_argument('--all-angles',   action='store_true')
    parser.add_argument('--epochs',       type=int,   default=8)
    parser.add_argument('--batch-size',   type=int,   default=16,
                        help='Smaller than L1 trainer — discriminator doubles memory')
    parser.add_argument('--lr',           type=float, default=0.0002)
    parser.add_argument('--lambda-l1',    type=float, default=100.0,
                        help='Weight of L1 loss vs adversarial (default 100 = pix2pix standard)')
    parser.add_argument('--num-workers',  type=int,   default=4)
    args = parser.parse_args()

    if args.source_angle is None and not args.all_angles:
        print("Error: specify --source-angle or --all-angles")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    unet_module = importlib.import_module('models.unet')
    transform   = unet_module.get_transform()

    angles  = SOURCE_ANGLES if args.all_angles else [args.source_angle]
    summary = {}
    for angle in angles:
        loss = train_one_angle(angle, args, device, transform)
        summary[angle] = round(loss, 4)

    if len(angles) > 1:
        print(f"\n{'='*60}")
        print(f"  SUMMARY — GAN translators")
        print(f"{'='*60}")
        print(f"  Plain L1 baseline val loss: ~0.116")
        for angle, loss in summary.items():
            better = '✓ better' if loss < 0.116 else '✗ worse'
            print(f"  {angle}°→15°:  val L1 = {loss:.4f}  {better}")


if __name__ == '__main__':
    main()
