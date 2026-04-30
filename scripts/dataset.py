"""
scripts/dataset.py
==================
Data layer for ATRNet-STAR experiments.

Reads images from the defense.atrnet_images table, applies class mapping,
splits into train/val/test, and returns PyTorch datasets ready for training.

ATRNet has 3 classification levels you can target via --config:
  4class       -- 4 main vehicle categories
  21class      -- 21 subclasses
  40class      -- 40 specific target types (hardest)

Usage (standalone test):
    python3 scripts/dataset.py --config 4class
"""

import argparse
import signal
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG


def _load_image_safe(path: str, timeout_secs: int = 10):
    """Load image with timeout. Returns grey placeholder on failure."""
    def _handler(signum, frame):
        raise TimeoutError(f"Image load timed out: {path}")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_secs)
    try:
        img = Image.open(path).convert('RGB')
        signal.alarm(0)
        return img
    except Exception:
        return Image.new('RGB', (128, 128), color=(64, 64, 64))
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ── Config → class field mapping ──────────────────────────────────────────────

CONFIG_FIELD = {
    '4class':  'main_class',
    '21class': 'subclass',
    '40class': 'target_type',
}


def get_images(conn, config_name: str, split_filter: str = None,
               setting: str = 'SOC_40classes',
               angle: float = None,
               parent_class: str = None,
               subclass: str = None,
               polarization: str = None,
               band: str = None,
               azimuth_min: float = None,
               azimuth_max: float = None) -> list[dict]:
    """
    Return list of {image_id, local_path, true_class} dicts.
    split_filter: 'train' or 'test' to use ATRNet's predefined splits.
                  None = return everything (you handle the split).
    setting: experiment_setting to filter on (default: SOC_40classes).
    angle: depression_angle to filter on (e.g. 15, 30, 45, 60). None = all angles.
    parent_class: restrict to images where main_class = this value (cascade Level 2).
    polarization: restrict to images with this polarization (e.g. HH, HV, VH, VV).
    band: restrict to images with this radar band (e.g. X, KU).
    azimuth_min/azimuth_max: restrict to azimuth_angle in [min, max).
    """
    class_field = CONFIG_FIELD.get(config_name)
    if not class_field:
        raise ValueError(f"Unknown config '{config_name}'. "
                         f"Choose from: {list(CONFIG_FIELD.keys())}")

    # class_field validated against allowed list to prevent SQL injection
    allowed = {'main_class', 'subclass', 'target_type'}
    if class_field not in allowed:
        raise ValueError(f"class_field '{class_field}' not in allowed set")

    sql = f"""
        SELECT id AS image_id, local_path, {class_field} AS true_class,
               target_type, band, depression_angle, azimuth_angle,
               bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
        FROM defense.atrnet_images
        WHERE local_path IS NOT NULL
          AND {class_field} IS NOT NULL
          AND {class_field} != ''
          AND experiment_setting = %s
    """
    params = [setting]
    if angle is not None:
        sql += " AND depression_angle = %s"
        params.append(angle)
    if parent_class is not None:
        sql += " AND main_class = %s"
        params.append(parent_class)
    if subclass is not None:
        sql += " AND subclass = %s"
        params.append(subclass)
    if polarization is not None:
        sql += " AND polarization = %s"
        params.append(polarization)
    if band is not None:
        sql += " AND band = %s"
        params.append(band)
    if azimuth_min is not None:
        sql += " AND azimuth_angle >= %s"
        params.append(azimuth_min)
    if azimuth_max is not None:
        sql += " AND azimuth_angle < %s"
        params.append(azimuth_max)
    if split_filter:
        sql += " AND split = %s"
        params.append(split_filter)

    sql += f" ORDER BY {class_field}, id"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{'image_id':    r['image_id'],
             'local_path':  r['local_path'],
             'true_class':  r['true_class'],
             'acq_key':     (r['target_type'], r['band'],
                             r['depression_angle'], r['azimuth_angle']),
             'bbox':        (r['bbox_xmin'], r['bbox_ymin'],
                             r['bbox_xmax'], r['bbox_ymax'])
                            if r.get('bbox_xmin') is not None else None,
             } for r in rows]


# ── Train/val/test split ──────────────────────────────────────────────────────

def split_images(images: list[dict], train_frac=0.70, val_frac=0.15, seed=42):
    """
    Acquisition-safe stratified split (prevents cross-polarization leakage).

    Step 1 — split ACQUISITION GROUPS into pool vs test.
      An acquisition group is all images sharing (target_type, band,
      depression_angle, azimuth_angle) — i.e. the same physical measurement
      captured across HH/HV/VH/VV channels simultaneously. Keeping all
      channels of one acquisition on the same side prevents the leakage
      identified in Gao et al. (IEEE GRSL 2026) for ATRNet-STAR.
      3,546 groups at 15° → clean stratified split with no domain shift.

    Step 2 — carve val from the pool at IMAGE level.
      Val is for epoch monitoring only (never reported), so image-level
      splitting within the pool is fine and guarantees non-empty val.
    """
    rng = np.random.default_rng(seed)
    pool_frac   = train_frac + val_frac        # e.g. 0.85
    val_of_pool = val_frac / pool_frac         # fraction of pool → val

    # Group by class → acquisition key → images
    by_class_acq: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    no_key: list[dict] = []

    for img in images:
        key = img.get('acq_key')
        if key is not None and all(v is not None for v in key):
            by_class_acq[img['true_class']][key].append(img)
        else:
            no_key.append(img)

    pool: list[dict] = []
    test: list[dict] = []

    # Step 1: acquisition-level pool vs test split (stratified by class)
    for cls, acqs in by_class_acq.items():
        acq_list = list(acqs.items())
        rng.shuffle(acq_list)
        n_pool = max(1, int(len(acq_list) * pool_frac))
        for _, imgs in acq_list[:n_pool]:
            pool.extend(imgs)
        for _, imgs in acq_list[n_pool:]:
            test.extend(imgs)

    # Fallback: images missing acquisition metadata split at image level
    if no_key:
        by_class: dict[str, list] = defaultdict(list)
        for img in no_key:
            by_class[img['true_class']].append(img)
        for cls, items in by_class.items():
            items = list(items)
            rng.shuffle(items)
            n_pool = max(1, int(len(items) * pool_frac))
            pool.extend(items[:n_pool])
            test.extend(items[n_pool:])

    # Step 2: image-level val carve from pool (for monitoring only)
    by_class_pool: dict[str, list] = defaultdict(list)
    for img in pool:
        by_class_pool[img['true_class']].append(img)

    train: list[dict] = []
    val:   list[dict] = []
    for cls, items in by_class_pool.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_of_pool))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    return train, val, test


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ATRNetDataset(Dataset):
    """
    PyTorch Dataset for ATRNet-STAR images.
    128×128 SAR chips, single-channel amplitude (loaded as RGB for model compatibility).
    """

    def __init__(self, records: list[dict], class_names: list[str],
                 transform=None, cache: bool = False, crop_bbox: bool = False):
        self.records      = records
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = transform
        self.crop_bbox    = crop_bbox
        self.cached       = None

        if cache:
            print(f"  Caching {len(records)} images into RAM...", flush=True)
            self.cached = [self._load(r) for r in records]
            print(f"  Done.", flush=True)

    def _load(self, rec):
        img = _load_image_safe(rec['local_path'])
        if self.crop_bbox and rec.get('bbox'):
            xmin, ymin, xmax, ymax = rec['bbox']
            # Add 4px padding, clamp to image bounds
            pad = 4
            w, h = img.size
            img = img.crop((max(0, xmin - pad), max(0, ymin - pad),
                            min(w, xmax + pad), min(h, ymax + pad)))
        return img

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.cached[idx] if self.cached else self._load(rec)
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[rec['true_class']]
        return img, label, rec['image_id']


# ── Public entry point ────────────────────────────────────────────────────────

def build_datasets(config_name: str,
                   max_per_class: int = None,
                   train_frac=0.70, val_frac=0.15,
                   transform=None, train_transform=None,
                   cache: bool = False, seed=42,
                   use_atrnet_splits: bool = False,
                   crop_bbox: bool = False,
                   setting: str = 'SOC_40classes',
                   angle: float = None,
                   parent_class: str = None,
                   subclass: str = None,
                   polarization: str = None,
                   band: str = None,
                   azimuth_min: float = None,
                   azimuth_max: float = None):
    """
    Main entry point called by train.py.

    use_atrnet_splits=True  — use the train/test split provided by ATRNet-STAR authors
                              (recommended for reproducible benchmarking)
    use_atrnet_splits=False — random stratified split (default, allows val set)

    Returns:
        train_ds, val_ds, test_ds — ATRNetDataset instances
        class_names               — sorted list of class names
        split_records             — {'train': [...], 'val': [...], 'test': [...]}
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        if use_atrnet_splits:
            # WARNING: the official ATRNet-STAR train/test split is known to have
            # cross-polarization data leakage (Gao et al., IEEE GRSL 2026).
            # Use the default scene-based random split for leak-free evaluation.
            print("WARNING: --use-atrnet-splits uses the official ATRNet-STAR split "
                  "which has known cross-polarization data leakage. "
                  "See Gao et al. (IEEE GRSL 2026) for details.")
            train_imgs = get_images(conn, config_name, split_filter='train', setting=setting, angle=angle, parent_class=parent_class, subclass=subclass, polarization=polarization, band=band, azimuth_min=azimuth_min, azimuth_max=azimuth_max)
            test_imgs  = get_images(conn, config_name, split_filter='test',  setting=setting, angle=angle, parent_class=parent_class, subclass=subclass, polarization=polarization, band=band, azimuth_min=azimuth_min, azimuth_max=azimuth_max)
            # Carve a val set out of train (15% of training data)
            rng = np.random.default_rng(seed)
            by_class = defaultdict(list)
            for img in train_imgs:
                by_class[img['true_class']].append(img)
            train_recs, val_recs = [], []
            for cls, items in by_class.items():
                items = list(items)
                rng.shuffle(items)
                n_val = max(1, int(len(items) * val_frac))
                val_recs.extend(items[:n_val])
                train_recs.extend(items[n_val:])
            test_recs = test_imgs
        else:
            all_images = get_images(conn, config_name, setting=setting, angle=angle, parent_class=parent_class, subclass=subclass, polarization=polarization, band=band, azimuth_min=azimuth_min, azimuth_max=azimuth_max)
            if max_per_class:
                by_class = defaultdict(list)
                for img in all_images:
                    by_class[img['true_class']].append(img)
                all_images = []
                for cls, items in by_class.items():
                    all_images.extend(items[:max_per_class])
            train_recs, val_recs, test_recs = split_images(
                all_images, train_frac, val_frac, seed)
    finally:
        conn.close()

    if not train_recs:
        raise ValueError(f"No images found for config='{config_name}'. "
                         f"Have you run scripts/ingest_atrnet.py?")

    class_names = sorted(set(img['true_class'] for img in train_recs + val_recs + test_recs))

    eff_train_transform = train_transform if train_transform is not None else transform
    train_ds = ATRNetDataset(train_recs, class_names, eff_train_transform, cache=cache, crop_bbox=crop_bbox)
    val_ds   = ATRNetDataset(val_recs,   class_names, transform,           cache=cache, crop_bbox=crop_bbox)
    test_ds  = ATRNetDataset(test_recs,  class_names, transform,           cache=cache, crop_bbox=crop_bbox)

    split_records = {'train': train_recs, 'val': val_recs, 'test': test_recs}
    return train_ds, val_ds, test_ds, class_names, split_records


# ── Standalone test ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the ATRNet data layer")
    parser.add_argument("--config", default='4class',
                        choices=list(CONFIG_FIELD.keys()),
                        help="Classification level (default: 4class)")
    args = parser.parse_args()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        images = get_images(conn, args.config)
    finally:
        conn.close()

    if not images:
        print("No images found — run scripts/ingest_atrnet.py first.")
        return

    from collections import Counter
    counts = Counter(img['true_class'] for img in images)
    print(f"\nConfig: {args.config}  ({len(images):,} images, {len(counts)} classes)")
    print(f"\nPer-class counts:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls:<30} {n:>6,}")

    train_ds, val_ds, test_ds, class_names, _ = build_datasets(args.config)
    print(f"\nSplit sizes:  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")


if __name__ == '__main__':
    main()
