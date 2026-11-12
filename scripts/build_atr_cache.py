"""
Cache chips for the background/target masking experiments (paper 2) as compact arrays.

  mstar_{train,test}        : MSTAR SOC, 17 deg train / 15 deg test, centre 128x128 log10 magnitude (float16)
  atr_soc_{train,test}      : ATRNet-STAR official SOC_40classes split (train capped at --soc-cap chips/class)
  atr_scene_{train,test}    : ATRNet-STAR official EOC_scene split (test 'group' = test_factory/test_urban/test_woodland)
ATRNet-STAR chips are 128x128 uint8 tifs; stored as-is. Each record keeps the target box centre (cx, cy).

Output: /Volumes/Machine Learning/atr_cache/<name>.npz  (X, y, cx, cy, group, classes)

Usage:
    venv/bin/python scripts/build_atr_cache.py [--soc-cap 800]
"""
import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG

OUT = Path('/Volumes/Machine Learning/atr_cache')


def load_tif(p):
    try:
        a = np.array(Image.open(p))
        return a if a.shape == (128, 128) else None
    except Exception:
        return None


def atr(setting, split_like, cap=None, seed=0):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT local_path, target_type, split, bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax
                       FROM defense.atrnet_images WHERE experiment_setting=%s AND split LIKE %s
                       AND target_type IS NOT NULL AND target_type != '' ORDER BY id""", [setting, split_like])
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    if split_like == 'test%':                     # keep only the per-condition sub-splits, not the union 'test'
        sub = [r for r in rows if r['split'] != 'test']
        rows = sub if sub else rows
    if cap:
        rng = np.random.default_rng(seed); by = defaultdict(list)
        for r in rows: by[r['target_type']].append(r)
        rows = [r for k in sorted(by) for r in rng.permutation(np.array(by[k], dtype=object))[:cap]]
    return rows


def save_atr(name, rows, classes):
    with Pool(8) as pool:
        imgs = pool.map(load_tif, [r['local_path'] for r in rows], chunksize=64)
    keep = [i for i, a in enumerate(imgs) if a is not None]
    X = np.stack([imgs[i] for i in keep]).astype(np.uint8)
    R = [rows[i] for i in keep]
    np.savez(OUT / f'{name}.npz', X=X, y=np.array([classes.index(r['target_type']) for r in R]),
             cx=np.array([(r['bbox_xmin'] + r['bbox_xmax']) / 2 for r in R]),
             cy=np.array([(r['bbox_ymin'] + r['bbox_ymax']) / 2 for r in R]),
             group=np.array([r['split'] for r in R]), classes=np.array(classes))
    print(f'{name:16s} {len(R):6d} chips ({len(rows) - len(R)} unreadable)', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--soc-cap', type=int, default=800); a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    from ingest_mstar import read_mstar
    from eval_clip_mstar import records, CLASSES, centre
    for name, dep in (('mstar_train', 17.0), ('mstar_test', 15.0)):
        rows = records(dep)
        X = np.stack([centre(np.log10(np.clip(read_mstar(Path(r['local_path']))[1], 1e-6, None))) for r in rows])
        np.savez(OUT / f'{name}.npz', X=X.astype(np.float16), y=np.array([CLASSES.index(r['target_short']) for r in rows]),
                 cx=np.full(len(rows), 64.0), cy=np.full(len(rows), 64.0), group=np.array(['all'] * len(rows)),
                 classes=np.array(CLASSES))
        print(f'{name:16s} {len(rows):6d} chips', flush=True)

    for setting, tag, cap in (('SOC_40classes', 'atr_soc', a.soc_cap), ('EOC_scene', 'atr_scene', None)):
        tr = atr(setting, 'train', cap)
        classes = sorted({r['target_type'] for r in tr})
        save_atr(f'{tag}_train', tr, classes)
        te = [r for r in atr(setting, 'test%') if r['target_type'] in classes]
        save_atr(f'{tag}_test', te, classes)


if __name__ == '__main__':
    main()
