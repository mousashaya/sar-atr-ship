"""
scripts/eval_clip_mstar.py
==========================
Do CLIP-style models (vanilla OpenAI CLIP ViT-L/14, SARCLIP ViT-L/14) know MSTAR vehicles?
Same protocol as eval_sarclip.py (ships), on the US benchmark:

  SOC: train 17 deg -> test 15 deg, 10 classes, all serial numbers (same records as train_mstar_soc.py)
  - zero-shot with vehicle-name prompts
  - frozen logistic-regression probe on pooled image features: full train set and k-shot (k=1,5,10; 5 draws)

Usage:
    venv/bin/python scripts/eval_clip_mstar.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from db_config import DB_CONFIG
from ingest_mstar import read_mstar, mag_to_uint8
from eval_sarclip import load_model, encode_images, encode_texts

OUT = Path('/Volumes/Machine Learning/sarclip_eval/mstar')
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 128
SEED = 42

CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']
NAMES = {
    '2S1':    ['a 2S1 self-propelled howitzer', 'a self-propelled artillery vehicle', 'a tracked howitzer'],
    'BMP2':   ['a BMP-2 infantry fighting vehicle', 'an infantry fighting vehicle', 'a tracked armored vehicle'],
    'BRDM2':  ['a BRDM-2 armored scout car', 'an armored reconnaissance vehicle', 'a wheeled armored car'],
    'BTR60':  ['a BTR-60 armored personnel carrier', 'an eight-wheeled armored personnel carrier', 'a wheeled APC'],
    'BTR70':  ['a BTR-70 armored personnel carrier', 'an armored personnel carrier', 'a wheeled armored vehicle'],
    'D7':     ['a D7 bulldozer', 'a bulldozer', 'a tracked construction vehicle'],
    'T62':    ['a T-62 tank', 'a main battle tank', 'a tank'],
    'T72':    ['a T-72 tank', 'a main battle tank', 'a tank with a long gun'],
    'ZIL131': ['a ZIL-131 military truck', 'a military cargo truck', 'a six-wheeled truck'],
    'ZSU234': ['a ZSU-23-4 anti-aircraft gun', 'a self-propelled anti-aircraft gun', 'a tracked air defense vehicle'],
}


def tmpl(n):
    return [f'this is an SAR image of {n} .', f'a SAR image of {n}.', f'a radar image of {n}.']


DB_NAME = {'BTR70': 'BTR70_TRAN', 'BRDM2': 'BRDM2_TRUC', 'ZSU234': 'ZSU23-4_GU', 'ZIL131': 'ZIL131_TRU'}
FROM_DB = {DB_NAME.get(c, c): c for c in CLASSES}


def records(dep):
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT local_path, target_short FROM defense.mstar_images
                       WHERE desired_depression = %s AND target_short = ANY(%s)
                       ORDER BY local_path""", [dep, list(FROM_DB)])
        rows = [dict(r, target_short=FROM_DB[r['target_short']]) for r in cur.fetchall()]
    conn.close()
    return rows


def centre(img, s=SIZE):
    h, w = img.shape
    if h < s or w < s:
        img = np.pad(img, ((max(0, s - h) // 2 + 1,) * 2, (max(0, s - w) // 2 + 1,) * 2))
        h, w = img.shape
    t, l = (h - s) // 2, (w - s) // 2
    return img[t:t + s, l:l + s]


def load(rows):
    u8, y = [], []
    for r in rows:
        _, mag = read_mstar(Path(r['local_path']))
        u8.append(centre(mag_to_uint8(mag)))
        y.append(CLASSES.index(r['target_short']))
    return np.stack(u8), np.array(y)


def probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    p = clf.predict(sc.transform(Xte))
    return accuracy_score(yte, p), f1_score(yte, p, average='macro')


def main():
    tr, te = records(17.0), records(15.0)
    Utr, ytr = load(tr)
    Ute, yte = load(te)
    print(f'train 17deg: {len(ytr)}  test 15deg: {len(yte)}', flush=True)
    res = {'n_train': len(ytr), 'n_test': len(yte)}
    rng = np.random.default_rng(SEED)
    for name in ['openai', 'sarclip_gelu']:
        model, tok = load_model(name)
        with torch.no_grad():
            Ptr, Ztr = encode_images(model, Utr)
            Pte, Zte = encode_images(model, Ute)
            W = encode_texts(model, tok, [sum((tmpl(n) for n in NAMES[c]), []) for c in CLASSES])
        zp = (Zte @ W.T).argmax(1)
        r = {'zeroshot_acc': accuracy_score(yte, zp),
             'zeroshot_f1': f1_score(yte, zp, average='macro'),
             'zeroshot_pred_dist': np.bincount(zp, minlength=len(CLASSES)).tolist()}
        r['probe_full_acc'], r['probe_full_f1'] = probe(Ptr, ytr, Pte, yte)
        for k in (1, 5, 10):
            accs = []
            for _ in range(5):
                idx = np.concatenate([rng.choice(np.where(ytr == c)[0], k, replace=False) for c in range(len(CLASSES))])
                accs.append(probe(Ptr[idx], ytr[idx], Pte, yte)[0])
            r[f'probe_{k}shot_acc'] = float(np.mean(accs))
            r[f'probe_{k}shot_std'] = float(np.std(accs))
        res[name] = r
        print(f'-- {name}', flush=True)
        for kk, v in r.items():
            print(f'   {kk:22s} {v if isinstance(v, list) else round(float(v), 3)}', flush=True)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    json.dump(res, open(OUT / 'eval.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
