"""
Extract DINOv2 embeddings from SAR ship chips.

Reads all .npz chip files, runs each through DINOv2 ViT-S/14,
saves a single embeddings.npz with:
  - embeddings: (N, 384) float32
  - mmsi: (N,) int64
  - vessel_name: (N,) str
  - vessel_type: (N,) int
  - length_m: (N,) float
  - sog: (N,) float
  - scene_id: (N,) str
  - acq_time: (N,) str
  - chip_path: (N,) str  -- for loading the chip image later
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

CHIPS_DIRS = [
    "/Volumes/Machine Learning/sar_ship_chips",
    "/Volumes/Machine Learning/sar_ship_chips_gulf",
    "/Volumes/Machine Learning/sar_ship_chips_slc_gulf",
]
OUT_PATH = "/Volumes/Machine Learning/sar_ship_chips/embeddings.npz"

# SAR preprocessing: amplitude -> dB -> normalize -> 3-channel
def chip_to_tensor(chip_vv, chip_vh=None):
    # dB scale, clip negative values
    vv_db = 10 * np.log10(chip_vv.astype(np.float32) + 1.0)
    if chip_vh is not None:
        vh_db = 10 * np.log10(chip_vh.astype(np.float32) + 1.0)
    else:
        vh_db = vv_db * 0.0

    # Per-chip percentile normalization to [0, 1]
    def norm(x):
        p2, p98 = np.percentile(x[x > 0], [2, 98]) if x.max() > 0 else (0, 1)
        x = np.clip((x - p2) / max(p98 - p2, 1e-6), 0, 1)
        return x.astype(np.float32)

    vv_n = norm(vv_db)
    vh_n = norm(vh_db)
    ratio = norm(vv_db - vh_db)  # VV/VH ratio (dB difference = log ratio)

    # Stack as 3-channel image: VV, VH, ratio
    rgb = np.stack([vv_n, vh_n, ratio], axis=2)  # (256, 256, 3)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return img


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    print("Loading DINOv2 ViT-S/14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    model.eval()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    print(f"  Model on: {device}")
    return model, device


def main():
    chip_files = []
    for d in CHIPS_DIRS:
        chip_files += sorted(glob.glob(os.path.join(d, '**/*.npz'), recursive=True))
    chip_files = [f for f in chip_files if 'embeddings' not in f]
    print(f"Found {len(chip_files)} chip files")

    if len(chip_files) == 0:
        print("No chips found. Run chip_ships_sar.py first.")
        return

    model, device = load_model()

    embeddings = []
    metas = {
        'mmsi': [], 'vessel_name': [], 'vessel_type': [], 'length_m': [],
        'sog': [], 'heading': [], 'scene_id': [], 'acq_time': [], 'chip_path': []
    }

    BATCH = 32
    batch_imgs, batch_paths = [], []

    def flush_batch():
        if not batch_imgs:
            return
        t = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            feats = model(t).cpu().numpy()
        embeddings.extend(feats)
        batch_imgs.clear()
        batch_paths.clear()

    for i, path in enumerate(chip_files):
        try:
            d = np.load(path, allow_pickle=True)
            chip_vv = d['chip_vv']
            chip_vh = d['chip_vh'] if 'chip_vh' in d else None
            img = chip_to_tensor(chip_vv, chip_vh)
            batch_imgs.append(img)
            batch_paths.append(path)
            metas['mmsi'].append(int(d['mmsi']))
            metas['vessel_name'].append(str(d['vessel_name']))
            metas['vessel_type'].append(int(d['vessel_type']))
            metas['length_m'].append(float(d['length_m']))
            metas['sog'].append(float(d['sog']))
            metas['heading'].append(float(d['heading']))
            metas['scene_id'].append(str(d['scene_id']))
            metas['acq_time'].append(str(d['acq_time']))
            metas['chip_path'].append(path)
        except Exception as e:
            print(f"  Skip {os.path.basename(path)}: {e}")
            continue

        if len(batch_imgs) >= BATCH:
            flush_batch()
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(chip_files)} chips embedded...")

    flush_batch()
    print(f"  {len(chip_files)}/{len(chip_files)} done")

    emb_array = np.array(embeddings, dtype=np.float32)
    print(f"\nEmbedding matrix: {emb_array.shape}")

    np.savez_compressed(
        OUT_PATH,
        embeddings=emb_array,
        **{k: np.array(v) for k, v in metas.items()}
    )
    print(f"Saved: {OUT_PATH}")

    # Quick stats
    mmsis = np.array(metas['mmsi'])
    unique, counts = np.unique(mmsis, return_counts=True)
    print(f"\nUnique ships: {len(unique)}")
    print(f"Ships with 5+ chips: {(counts >= 5).sum()}")
    print(f"Ships with 10+ chips: {(counts >= 10).sum()}")
    print(f"Max chips per ship: {counts.max()}")


if __name__ == '__main__':
    main()
