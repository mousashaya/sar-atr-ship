"""
scripts/eval_orient.py
======================
Does rotating each ship chip so its hull lies vertical help vessel-type classification?

Data: our Sentinel-1 Gulf GRD chips (radar geometry), resolved AIS labels.
Hull angle in image (deg from +row toward +col, mod 180):
  - AIS:   heading - track, track = 350 (ascending, ~00 UTC) / 190 (descending, ~12 UTC)   [validated ~5 deg]
  - image: long axis of the bright blob at chip centre (second moments)    -> no AIS needed
Bow/stern is ambiguous in SAR, so every rotated view is averaged with its +180 twin.

Conditions (frozen OpenAI CLIP ViT-L/14 pooled features + logistic regression, 5-fold MMSI-grouped CV):
  base        no rotation
  base_sym    no rotation, averaged with 180-deg twin   (fair control for the twin-averaging)
  ais         AIS angle for train and test          (oracle upper bound; needs AIS at test)
  ais_img     AIS angle for train, image angle for test (deployable)
  img         image angle for train and test        (fully AIS-free)

Usage:
    venv/bin/python scripts/eval_orient.py extract
    venv/bin/python scripts/eval_orient.py eval
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from eval_sarclip import to_u8, load_model, encode_images, setting_rows, LABELS, CHIPS_DIR, SEED

OUT = Path('/Volumes/Machine Learning/sarclip_eval/orient')
OUT.mkdir(parents=True, exist_ok=True)
CROP = 128
MODEL = 'openai'


def ais_angle(heading, scene_id):
    if not (0 < heading < 360):          # 511 = unavailable; 0 = missing placeholder in chip_ships_gulf.py
        return None
    hr = int(scene_id.split('_')[4][9:11])
    track = 190.0 if 6 <= hr < 18 else 350.0
    return (heading - track) % 180


def image_angle(vv, half=24, min_ratio=2.0):
    c = vv.shape[0] // 2
    db = 10 * np.log10(vv[c - half:c + half, c - half:c + half].astype(np.float64) + 1)
    lab, n = ndimage.label(db > np.percentile(db, 97))
    if n == 0:
        return None
    k = lab[half, half] or (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(lab == k)
    if len(xs) < 6:
        return None
    w, v = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    if w[1] < min_ratio * max(w[0], 1e-6):
        return None
    return np.degrees(np.arctan2(v[0, 1], v[1, 1])) % 180


def rotated_crop(u8, angle):
    """Rotate so a hull at `angle` (deg from +row toward +col) becomes vertical; centre crop."""
    c = u8.shape[0] // 2
    big = u8[c - 128:c + 128, c - 128:c + 128]
    r = ndimage.rotate(big, -angle, reshape=False, order=1, mode='constant', cval=0) if angle else big
    h = CROP // 2
    return r[128 - h:128 + h, 128 - h:128 + h]


def stage_extract():
    meta_all = json.load(open(Path('/Volumes/Machine Learning/sarclip_eval/meta.json')))
    labels = json.load(open(LABELS))
    rows, _, _ = setting_rows(meta_all, labels, ['Tanker', 'Cargo', 'Tug/Special', 'Fishing'], 'MEDIUM')
    views = {k: [] for k in ['base', 'base180', 'ais', 'ais180', 'img', 'img180']}
    info = []
    for n, i in enumerate(rows):
        m = meta_all[i]
        f = np.load(m['path'], allow_pickle=True)
        vv = f['chip_vv']
        u8 = to_u8(vv)
        a_img = image_angle(vv)
        a_ais = ais_angle(m['heading'], m['scene'])
        a_ais_f = a_ais if a_ais is not None else (a_img or 0.0)
        a_img_f = a_img if a_img is not None else 0.0
        views['base'].append(rotated_crop(u8, 0)); views['base180'].append(rotated_crop(u8, 180))
        views['ais'].append(rotated_crop(u8, a_ais_f)); views['ais180'].append(rotated_crop(u8, a_ais_f + 180))
        views['img'].append(rotated_crop(u8, a_img_f)); views['img180'].append(rotated_crop(u8, a_img_f + 180))
        info.append(dict(idx=int(i), a_ais=a_ais, a_img=a_img))
        if n % 1000 == 0:
            print(f'{n}/{len(rows)}', flush=True)
    json.dump(info, open(OUT / 'info.json', 'w'))
    # sanity: after AIS rotation the measured hull axis should be ~vertical (0 deg)
    chk = [image_angle(np.asarray(views['ais'][j], np.float64) ** 2 + 1) for j in range(0, len(info), 7)
           if info[j]['a_ais'] is not None]
    chk = np.array([((a + 90) % 180) - 90 for a in chk if a is not None])
    print(f'post-rotation axis |deviation from vertical|: median {np.median(np.abs(chk)):.1f} deg (n={len(chk)})', flush=True)
    model, _ = load_model(MODEL)
    with torch.no_grad():
        for k, v in views.items():
            pooled, _ = encode_images(model, np.stack(v))
            np.save(OUT / f'feat_{k}.npy', pooled)
            print('extracted', k, pooled.shape, flush=True)


def stage_eval():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    meta_all = json.load(open(Path('/Volumes/Machine Learning/sarclip_eval/meta.json')))
    labels = json.load(open(LABELS))
    info = json.load(open(OUT / 'info.json'))
    pos = {d['idx']: j for j, d in enumerate(info)}
    F = {k: np.load(OUT / f'feat_{k}.npy') for k in ['base', 'base180', 'ais', 'ais180', 'img', 'img180']}
    sym = lambda a, b: (F[a] + F[b]) / 2
    feats = {'base': F['base'], 'base_sym': sym('base', 'base180'), 'ais': sym('ais', 'ais180'), 'img': sym('img', 'img180')}
    print(f"hull angle from image found for {np.mean([d['a_img'] is not None for d in info])*100:.0f}% of chips; "
          f"AIS angle for {np.mean([d['a_ais'] is not None for d in info])*100:.0f}%")
    report = {}
    for sname, (classes, conf) in {'3cls_HIGH': (['Tanker', 'Cargo', 'Fishing'], 'HIGH'),
                                   '4cls_MEDIUM+': (['Tanker', 'Cargo', 'Tug/Special', 'Fishing'], 'MEDIUM')}.items():
        rows, y, g = setting_rows(meta_all, labels, classes, conf)
        sel = np.array([pos[i] for i in rows])
        maj = np.bincount(y).max() / len(y)
        print(f'\n## {sname}: {len(y)} chips, {len(set(g))} vessels, majority-class OA {maj*100:.1f}%')
        for cond, (trk, tek) in {'base': ('base', 'base'), 'base_sym': ('base_sym', 'base_sym'),
                                 'ais': ('ais', 'ais'), 'ais_img': ('ais', 'img'), 'img': ('img', 'img')}.items():
            Xtr_all, Xte_all = feats[trk][sel], feats[tek][sel]
            oas, f1s = [], []
            for seed in (SEED, SEED + 1, SEED + 2):
                pred = np.zeros_like(y)
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr_all, y, g):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))
                    clf.fit(Xtr_all[tr], y[tr]); pred[te] = clf.predict(Xte_all[te])
                oas.append(accuracy_score(y, pred)); f1s.append(f1_score(y, pred, average='macro'))
            report[f'{sname}|{cond}'] = dict(oa=float(np.mean(oas)), oa_sd=float(np.std(oas)),
                                             f1=float(np.mean(f1s)), f1_sd=float(np.std(f1s)), majority=float(maj))
            print(f'  {cond:9s} OA {np.mean(oas)*100:5.1f} ±{np.std(oas)*100:.1f}   macroF1 {np.mean(f1s)*100:5.1f} ±{np.std(f1s)*100:.1f}')
    json.dump(report, open(OUT / 'eval.json', 'w'), indent=2)


if __name__ == '__main__':
    {'extract': stage_extract, 'eval': stage_eval}[sys.argv[1]]()
