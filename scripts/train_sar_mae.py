"""
scripts/train_sar_mae.py
=========================
Stage 1: SAR domain adaptation via Masked Autoencoder (MAE).

Fine-tunes DINOv2 ViT-S/14 on random patches extracted from raw Sentinel-1
Gulf of Mexico scenes — no labels, no ship identity. Forces the backbone to
learn SAR speckle statistics, sea clutter texture, and strong point scatterers.

Architecture:
  Encoder : DINOv2 ViT-S/14 (fully unfrozen) — sees only visible 25% of tokens
  Decoder : lightweight 2-block transformer → reconstruct masked 75% of tokens

Output: models/saved/sar_mae_v1/backbone.pt
  Load this instead of vanilla DINOv2 as the Stage 2 starting point.

Coverage note: scenes cover Aug–Dec 2024 only. Jan–May 2024 not yet downloaded.

Usage:
    venv/bin/python scripts/train_sar_mae.py
    venv/bin/python scripts/train_sar_mae.py --epochs 20 --patches-per-scene 300
"""

import argparse
import json
import random
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

SCENE_DIR    = Path('/Volumes/Machine Learning/sentinel1_slc_gulf')
PATCHES_FILE = Path('/Users/mousashaya/MLData/sar_patches_slc_gulf/patches.npy')
MODELS_DIR   = Path('/Volumes/Machine Learning/sar_mae_slc_v1')
LOGS_DIR     = Path(__file__).parent.parent / 'logs' / 'sar_mae_slc'
SEED       = 42
PATCH_SIZE = 224   # input to DINOv2
TOKEN_SIZE = 14    # DINOv2 ViT-S/14 patch size
N_TOKENS   = (PATCH_SIZE // TOKEN_SIZE) ** 2  # 256
MASK_RATIO = 0.75  # mask 75% of tokens
EMBED_DIM  = 384   # DINOv2 ViT-S/14

GMAIL_USER     = "mshaya01.ml@gmail.com"
GMAIL_APP_PASS = "pzxlvkprrmap wmqz".replace(' ', '')
TO_EMAIL       = "mousa.shaya@gmail.com"

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])


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


# ── SAR patch preprocessing ────────────────────────────────────────────────────

def norm_channel(x):
    valid = x[x > 0]
    if valid.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    return np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1).astype(np.float32)


def raw_to_tensor(vv_raw, vh_raw):
    vv_db = 10 * np.log10(vv_raw.astype(np.float32) + 1.0)
    vh_db = 10 * np.log10(vh_raw.astype(np.float32) + 1.0)
    rgb   = np.stack([norm_channel(vv_db),
                      norm_channel(vh_db),
                      norm_channel(vv_db - vh_db)], axis=2)
    img   = Image.fromarray((rgb * 255).astype(np.uint8))
    t     = transforms.ToTensor()(img)   # (3, 224, 224)  [0,1]
    return _normalize(t)


# ── Dataset: streams random patches from SAFE TIFFs ───────────────────────────

def find_safe_tiffs(scene_dir):
    """Return list of (vv_path, vh_path) for each IW subswath across all SAFE scenes.
    SLC scenes have 3 subswaths (iw1/iw2/iw3) — each treated as independent patch source.
    """
    pairs = []
    for safe in sorted(scene_dir.glob('*.SAFE')):
        meas = safe / 'measurement'
        for sw in ['iw1', 'iw2', 'iw3']:
            vv_files = sorted(meas.glob(f'*-{sw}-slc-vv-*.tiff'))
            vh_files = sorted(meas.glob(f'*-{sw}-slc-vh-*.tiff'))
            if vv_files and vh_files:
                pairs.append((vv_files[0], vh_files[0]))
    return pairs


class SARPatchDataset(Dataset):
    """Loads pre-extracted uint8 patches from internal SSD — fast sequential reads."""
    def __init__(self, patches_file, augment=True):
        self.augment = augment
        print(f"  Loading patches from {patches_file} ...")
        self.data = np.load(patches_file, mmap_mode='r')
        print(f"  Total patches: {len(self.data):,}  shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx].astype(np.float32) / 255.0  # uint8 → [0,1]
        t = _normalize(torch.from_numpy(patch))

        if self.augment:
            if random.random() > 0.5:
                t = torch.flip(t, dims=[2])
            if random.random() > 0.5:
                t = torch.flip(t, dims=[1])
            k = random.randint(0, 3)
            if k > 0:
                t = torch.rot90(t, k=k, dims=[1, 2])

        return t


# ── MAE model ─────────────────────────────────────────────────────────────────

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class SARMAE(nn.Module):
    def __init__(self, unfreeze_last=12):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14',
            pretrained=True, verbose=False
        )
        # Unfreeze last N transformer blocks (default: all 12)
        for p in self.backbone.parameters():
            p.requires_grad = False
        blocks = list(self.backbone.blocks)
        for block in blocks[-unfreeze_last:]:
            for p in block.parameters():
                p.requires_grad = True
        # Also unfreeze layer norm
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        # Lightweight MAE decoder
        decoder_dim = 256
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.decoder_proj = nn.Linear(EMBED_DIM, decoder_dim)
        self.decoder_pos  = nn.Embedding(N_TOKENS, decoder_dim)
        self.decoder_blocks = nn.Sequential(
            TransformerDecoderBlock(decoder_dim, 4),
            TransformerDecoderBlock(decoder_dim, 4),
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Reconstruct TOKEN_SIZE×TOKEN_SIZE×3 pixels per masked token
        self.pixel_head = nn.Linear(decoder_dim, TOKEN_SIZE * TOKEN_SIZE * 3)

        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask_ratio=MASK_RATIO):
        """
        x: (B, 3, 224, 224)
        Returns: loss (scalar)
        """
        B = x.shape[0]
        device = x.device

        # ── Patchify ground truth ──────────────────────────────────────────
        # Split image into TOKEN_SIZE×TOKEN_SIZE patches for reconstruction target
        # x: (B, 3, 224, 224) → (B, N_TOKENS, TOKEN_SIZE*TOKEN_SIZE*3)
        p = TOKEN_SIZE
        h = w = PATCH_SIZE // p        # 16
        x_patches = x.reshape(B, 3, h, p, w, p)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5)   # (B, h, w, 3, p, p)
        x_patches = x_patches.reshape(B, N_TOKENS, 3 * p * p)  # (B, 256, 588)

        # Normalize patch targets (per-patch mean/std as in original MAE)
        mean = x_patches.mean(dim=-1, keepdim=True)
        std  = x_patches.var(dim=-1, keepdim=True).sqrt() + 1e-6
        x_norm = (x_patches - mean) / std

        # ── Random masking ──────────────────────────────────────────────────
        n_mask    = int(N_TOKENS * mask_ratio)
        n_visible = N_TOKENS - n_mask
        noise     = torch.rand(B, N_TOKENS, device=device)
        ids_sort  = noise.argsort(dim=1)
        ids_restore = ids_sort.argsort(dim=1)
        visible_ids = ids_sort[:, :n_visible]   # (B, n_visible)
        mask        = torch.ones(B, N_TOKENS, device=device)
        mask.scatter_(1, visible_ids, 0)        # 1 = masked, 0 = visible

        # ── Encoder: DINOv2 on visible tokens only ──────────────────────────
        # We need to feed only visible patches to DINOv2.
        # DINOv2 doesn't natively support partial token input, so we
        # construct a masked image: replace masked patches with zeros.
        x_masked = x.clone()
        for b in range(B):
            for tok_idx in ids_sort[b, n_visible:].tolist():
                ri, ci = divmod(tok_idx, w)
                x_masked[b, :, ri*p:(ri+1)*p, ci*p:(ci+1)*p] = 0.0

        # Extract patch token features — forward_features returns dict with
        # 'x_norm_patchtokens': (B, N_TOKENS, 384), CLS already excluded
        feats = self.backbone.forward_features(x_masked)['x_norm_patchtokens']

        # ── Decoder ─────────────────────────────────────────────────────────
        # Fill masked positions with learned mask token
        mask_tokens = self.mask_token.expand(B, N_TOKENS, -1)
        # Replace visible positions with encoder output, masked with mask token
        vis_mask = (mask == 0).unsqueeze(-1).float()  # (B, N_TOKENS, 1)
        decoder_input = feats * vis_mask + mask_tokens * (1 - vis_mask)

        # Project to decoder dim + add position embeddings
        pos_ids = torch.arange(N_TOKENS, device=device).unsqueeze(0).expand(B, -1)
        decoder_input = self.decoder_proj(decoder_input) + self.decoder_pos(pos_ids)
        decoder_out   = self.decoder_blocks(decoder_input)
        decoder_out   = self.decoder_norm(decoder_out)
        pred          = self.pixel_head(decoder_out)   # (B, N_TOKENS, 588)

        # ── Loss: MSE on masked tokens only ─────────────────────────────────
        loss = (pred - x_norm) ** 2
        loss = loss.mean(dim=-1)               # (B, N_TOKENS)
        loss = (loss * mask).sum() / mask.sum()
        return loss


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--patches-per-scene', type=int,   default=500)
    parser.add_argument('--patience',          type=int,   default=3)
    parser.add_argument('--min-delta',         type=float, default=0.001)
    parser.add_argument('--batch-size',        type=int,   default=16)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--unfreeze-last',     type=int,   default=12,
                        help='Number of ViT blocks to unfreeze (default: all 12)')
    parser.add_argument('--resume-backbone',   type=str,   default=None,
                        help='Backbone filename in MODELS_DIR to warm-start from (e.g. backbone_epoch03.pt)')
    parser.add_argument('--start-epoch',       type=int,   default=None,
                        help='Epoch number to start from when using --resume-backbone')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ── Data ──
    if not PATCHES_FILE.exists():
        print(f"ERROR: Patches file not found: {PATCHES_FILE}")
        print(f"Run first: venv/bin/python scripts/extract_slc_patches.py")
        return

    dataset = SARPatchDataset(PATCHES_FILE)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0, pin_memory=False)
    print(f"Batches per epoch: {len(loader):,}")

    # ── Model ──
    print(f"Building SAR MAE (unfreeze_last={args.unfreeze_last})...")
    model = SARMAE(unfreeze_last=args.unfreeze_last).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05
    )
    # last_epoch=-1 is default (fresh); set to start_epoch-1 when warm-starting
    # so the cosine schedule is at the right phase without calling step() early
    sched_last = (args.start_epoch - 2) if (args.resume_backbone and args.start_epoch) else -1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=sched_last
    )

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_epoch = 1
    history     = []
    checkpoint  = MODELS_DIR / 'checkpoint_latest.pt'
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        history     = ckpt['history']
        print(f"  Resumed from epoch {ckpt['epoch']}  (loss={ckpt['loss']:.4f})")
    elif args.resume_backbone:
        # Warm-start from a backbone-only checkpoint (no optimizer state)
        backbone_path = MODELS_DIR / args.resume_backbone
        if not backbone_path.exists():
            print(f"ERROR: --resume-backbone file not found: {backbone_path}")
            return
        state = torch.load(backbone_path, map_location=device)
        model.backbone.load_state_dict(state)
        if args.start_epoch is not None:
            start_epoch = args.start_epoch
            # scheduler already initialised at the right last_epoch above
        print(f"  Warm-started backbone from {args.resume_backbone}, "
              f"resuming at epoch {start_epoch} (fresh optimizer)")

    print(f"\n{'='*60}")
    print(f"  SAR MAE — Stage 1 Domain Adaptation")
    print(f"  Patches: {len(dataset):,}  (pre-extracted, internal SSD)")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}")
    print(f"  Mask ratio: {MASK_RATIO:.0%}  LR: {args.lr}")
    print(f"{'='*60}\n")

    best_loss      = min((h['loss'] for h in history), default=float('inf'))
    patience_count = 0
    # After a warm-start (backbone-only resume), skip patience for 2 epochs
    # while the fresh optimizer rebuilds momentum
    warmup_epochs  = (start_epoch + 1) if args.resume_backbone else 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0     = time.time()
        losses = []

        for step, batch in enumerate(loader, 1):
            batch = batch.to(device)
            loss  = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 1000 == 0:
                pct = 100 * step / len(loader)
                ts  = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"    {ts}  [{step}/{len(loader)}  {pct:.0f}%  "
                      f"loss={np.mean(losses):.4f}]", flush=True)

        scheduler.step()
        avg_loss = np.mean(losses)
        elapsed  = time.time() - t0
        print(f"  Epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  ({elapsed:.0f}s)", flush=True)
        history.append({'epoch': epoch, 'loss': avg_loss})

        # Save resumable checkpoint + backbone weights each epoch
        torch.save({
            'epoch':     epoch,
            'loss':      avg_loss,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history':   history,
        }, MODELS_DIR / 'checkpoint_latest.pt')
        torch.save(model.backbone.state_dict(),
                   MODELS_DIR / f'backbone_epoch{epoch:02d}.pt')

        # Early stopping (skip patience check during warm-start settling period)
        if epoch <= warmup_epochs:
            print(f"  Warm-start grace period — patience paused (epoch {epoch}/{warmup_epochs})",
                  flush=True)
            if avg_loss < best_loss:
                best_loss = avg_loss  # still track best, just don't penalise
        elif avg_loss < best_loss - args.min_delta:
            best_loss      = avg_loss
            patience_count = 0
            torch.save(model.backbone.state_dict(), MODELS_DIR / 'backbone_best.pt')
        else:
            patience_count += 1
            print(f"  No improvement for {patience_count}/{args.patience} epochs "
                  f"(best={best_loss:.4f})", flush=True)
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} — best loss {best_loss:.4f}")
                break

    # Save final backbone
    torch.save(model.backbone.state_dict(), MODELS_DIR / 'backbone.pt')
    print(f"\nBackbone saved to {MODELS_DIR}/backbone.pt")
    print("Use this as Stage 2 starting point instead of vanilla DINOv2.")

    with open(LOGS_DIR / 'results.json', 'w') as f:
        json.dump({'args': vars(args), 'history': history,
                   'n_patches': len(dataset)}, f, indent=2)

    send_email(
        f"SAR MAE Stage 1 done — final loss {history[-1]['loss']:.4f}",
        f"Patches: {len(dataset):,}\n"
        f"Epochs: {args.epochs}\n"
        f"Final loss: {history[-1]['loss']:.4f}\n"
        f"Backbone saved to: {MODELS_DIR}/backbone.pt\n\n"
        f"NOTE: Jan–Apr 2024 scenes not included — download before final paper run.\n"
        f"NOTE: Gulf of Mexico only — add global scenes for geographic diversity."
    )


if __name__ == '__main__':
    main()
