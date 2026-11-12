"""
models/saratrx.py
==================
SARATR-X backbone wrapper — mirrors convnext_small.py interface.

Architecture: HiViT-Base (Hierarchical Vision Transformer)
Feature dim:  512
Pretrained:   SAR images (180K unlabeled SAR targets)
License:      CC BY-NC 4.0 (research only)

Weights: downloaded from https://huggingface.co/waterdisappear/SARATR-X
  File:  models/saratrx_arch/checkpoint-800.pth (800-epoch MAE checkpoint,
         186K_all run — see project memory `atrnet_star_rewrite` for provenance)

Usage:
    from models.saratrx import build_model, get_transform, FEATURE_DIM
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

# Add SARATR-X architecture to path
ARCH_DIR = Path(__file__).parent / 'saratrx_arch'
sys.path.insert(0, str(ARCH_DIR))

WEIGHTS_PATH = ARCH_DIR / 'checkpoint-800.pth'
FEATURE_DIM  = 512


def get_transform():
    """Inference transform — SAR amplitude images, no color jitter."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # SAR is single-channel
        transforms.ToTensor(),
        # SAR-specific normalization (mean/std from amplitude distribution)
        # Using ImageNet values as fallback until SAR-specific values confirmed
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class SARATRXClassifier(nn.Module):
    """
    HiViT encoder + classification head.
    Wraps the MAE encoder for classification/feature extraction.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        from models_hivit import HiViT
        self.encoder = HiViT(
            img_size=224, patch_size=16, in_chans=3,
            num_classes=0,          # no head — we add our own
            embed_dim=512, depths=[2, 2, 20], num_heads=8,
            stem_mlp_ratio=3., mlp_ratio=4.,
        )
        # Classification head equivalent to ConvNeXt's classifier[2]
        self.classifier = nn.Sequential(
            nn.Identity(),          # [0] placeholder
            nn.Identity(),          # [1] feature layer (hook target)
            nn.Linear(FEATURE_DIM, num_classes),  # [2] head
        )
        self._feature_dim = FEATURE_DIM

    def forward_features(self, x):
        """Extract 512-dim feature vector."""
        features = self.encoder.forward_features(x)  # (B, N, 512)
        # Mean-pool patch tokens → (B, 512)
        pooled = self.encoder.fc_norm(features.mean(dim=1))
        return pooled

    def forward(self, x):
        feat = self.forward_features(x)
        # Pass through identity layers so hook on classifier[1] works
        feat = self.classifier[0](feat)
        feat = self.classifier[1](feat)
        return self.classifier[2](feat)


def build_model(num_classes: int) -> nn.Module:
    model = SARATRXClassifier(num_classes)

    if not WEIGHTS_PATH.exists():
        print(f"  [SARATR-X] WARNING: weights not found at {WEIGHTS_PATH}")
        print(f"  Download from https://huggingface.co/waterdisappear/SARATR-X")
        print(f"  Running with random encoder weights (for testing only)")
        return model

    print(f"  [SARATR-X] Loading pretrained weights from {WEIGHTS_PATH}")
    # Checkpoint stores an argparse.Namespace of training args alongside the
    # weights (standard for this MAE training script) — not loadable under
    # PyTorch 2.6's default weights_only=True without allowlisting it.
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    ckpt = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)

    # MAE checkpoints store encoder weights under 'model' key
    state = ckpt.get('model', ckpt)

    # Keep only encoder weights, drop decoder
    encoder_state = {k.replace('encoder.', ''): v
                     for k, v in state.items()
                     if not k.startswith('decoder')}

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    print(f"  Encoder loaded successfully")

    # All parameters trainable (fine-tune everything)
    for param in model.parameters():
        param.requires_grad = True

    return model


def get_config() -> dict:
    return {
        'model_type':  'saratrx',
        'backbone':    'hivit_base',
        'pretrained':  'SAR_180K',
        'frozen':      False,
        'feature_dim': FEATURE_DIM,
    }
