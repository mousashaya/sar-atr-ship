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


def get_images(conn, config_name: str, split_filter: str = None) -> list[dict]:
    """
    Return list of {image_id, local_path, true_class} dicts.
    split_filter: 'train' or 'test' to use ATRNet's predefined splits.
                  None = return everything (you handle the split).
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
        SELECT id AS image_id, local_path, {class_field} AS true_class
        FROM defense.atrnet_images
        WHERE local_path IS NOT NULL
          AND {class_field} IS NOT NULL
          AND {class_field} != ''
    """
    params = []
    if split_filter:
        sql += " AND split = %s"
        params.append(split_filter)

    sql += f" ORDER BY {class_field}, id"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{'image_id': r['image_id'],
             'local_path': r['local_path'],
             'true_class': r['true_class']} for r in rows]


# ── Train/val/test split ──────────────────────────────────────────────────────

def split_images(images: list[dict], train_frac=0.70, val_frac=0.15, seed=42):
    """Stratified random split by class."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for img in images:
        by_class[img['true_class']].append(img)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ATRNetDataset(Dataset):
    """
    PyTorch Dataset for ATRNet-STAR images.
    128×128 SAR chips, single-channel amplitude (loaded as RGB for model compatibility).
    """

    def __init__(self, records: list[dict], class_names: list[str],
                 transform=None, cache: bool = False):
        self.records      = records
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = transform
        self.cached       = None

        if cache:
            print(f"  Caching {len(records)} images into RAM...", flush=True)
            self.cached = [Image.open(r['local_path']).convert('RGB') for r in records]
            print(f"  Done.", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = self.cached[idx] if self.cached else _load_image_safe(rec['local_path'])
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
                   use_atrnet_splits: bool = False):
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
            train_imgs = get_images(conn, config_name, split_filter='train')
            test_imgs  = get_images(conn, config_name, split_filter='test')
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
            all_images = get_images(conn, config_name)
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
    train_ds = ATRNetDataset(train_recs, class_names, eff_train_transform, cache=cache)
    val_ds   = ATRNetDataset(val_recs,   class_names, transform, cache=cache)
    test_ds  = ATRNetDataset(test_recs,  class_names, transform, cache=cache)

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
