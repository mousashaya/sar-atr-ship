"""
SARCLIP (Jiang et al., ISPRS 2025) check on the Gulf Sentinel-1 GRD ship chips.

Weights: HF mirror BiliSakura/SARCLIP-ViT-L-14 @ a2a8912 (third-party). The mirror is only
half-converted (text tower in HF naming, vision tower + projections still OpenCLIP naming),
so we convert the OpenCLIP keys ourselves and load with strict=True.
Control: openai/clip-vit-large-patch14.

Stages:
  cache    read chips once -> centre/corner crops (uint8) + metadata   (slow, drive I/O)
  sanity   label-free check: ship-centred crops vs empty-water corner crops
  extract  image embeddings for all labelled chips
  eval     zero-shot vessel type + frozen linear probe (MMSI-grouped vs chip-level CV)

Usage:
  venv/bin/python scripts/eval_sarclip.py cache
  venv/bin/python scripts/eval_sarclip.py sanity
  venv/bin/python scripts/eval_sarclip.py extract
  venv/bin/python scripts/eval_sarclip.py eval
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHIPS_DIR = Path('/Volumes/Machine Learning/sar_ship_chips_gulf')
PRETRAINED = Path('/Volumes/Machine Learning/pretrained')
OUT = Path('/Volumes/Machine Learning/sarclip_eval')
LABELS = Path('data/resolved_labels.json')
CROP = 128          # 128 px * 10 m = 1.28 km centred on the AIS position
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
SEED = 42


def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ----------------------------------------------------------------------------- cache
def to_u8(chip):
    db = 10 * np.log10(chip.astype(np.float32) + 1.0)
    p2, p98 = np.percentile(db[np.isfinite(db)], [50, 99.9])  # median->99.9%: dark sea, bright ship; fairest to SARCLIP (sanity2)
    return (np.clip((db - p2) / max(p98 - p2, 1e-6), 0, 1) * 255).astype(np.uint8)


def stage_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CHIPS_DIR.glob('**/*.npz'))
    h = CROP // 2
    centre, corner, full, meta = [], [], [], []
    for i, p in enumerate(files):
        try:
            f = np.load(p, allow_pickle=True)
            vv = to_u8(f['chip_vv'])
        except Exception as e:
            print('skip', p.name, e); continue
        c = vv.shape[0] // 2
        centre.append(vv[c - h:c + h, c - h:c + h])
        corner.append(vv[:CROP, :CROP])
        # full chip downsampled to 224 for the "whole chip" view
        full.append(np.asarray(torch.nn.functional.interpolate(
            torch.from_numpy(vv)[None, None].float(), size=224, mode='area')[0, 0].byte()))
        meta.append(dict(path=str(p), mmsi=str(int(f['mmsi'])), length_m=float(f['length_m']),
                         sog=float(f['sog']), heading=float(f['heading']), scene=str(f['scene_id']),
                         clean=bool(f['clean']), n_other=int(f['n_other_ships'])))
        if i % 1000 == 0:
            print(f'{i}/{len(files)}', flush=True)
    np.save(OUT / 'centre.npy', np.stack(centre))
    np.save(OUT / 'corner.npy', np.stack(corner))
    np.save(OUT / 'full.npy', np.stack(full))
    json.dump(meta, open(OUT / 'meta.json', 'w'))
    print('cached', len(meta))


# ----------------------------------------------------------------------------- models
def convert_openclip_visual(sd):
    """Map OpenCLIP vision-tower keys (as left in the mirror) to HF CLIPModel keys."""
    out = {}
    for k, v in sd.items():
        if not k.startswith('vision_model.') and k != 'text_projection':
            out[k] = v; continue
        if k == 'text_projection':                    # OpenCLIP: x @ P  -> HF Linear weight = P.T
            out['text_projection.weight'] = v.t().contiguous(); continue
        r = k[len('vision_model.'):]
        if r == 'conv1.weight':          out['vision_model.embeddings.patch_embedding.weight'] = v
        elif r == 'class_embedding':     out['vision_model.embeddings.class_embedding'] = v
        elif r == 'positional_embedding':out['vision_model.embeddings.position_embedding.weight'] = v
        elif r.startswith('ln_pre.'):    out['vision_model.pre_layrnorm.' + r.split('.')[1]] = v
        elif r.startswith('ln_post.'):   out['vision_model.post_layernorm.' + r.split('.')[1]] = v
        elif r == 'proj':                out['visual_projection.weight'] = v.t().contiguous()
        elif r.startswith('transformer.resblocks.'):
            parts = r.split('.')
            i, rest = parts[2], '.'.join(parts[3:])
            base = f'vision_model.encoder.layers.{i}.'
            if rest in ('attn.in_proj_weight', 'attn.in_proj_bias'):
                kind = 'weight' if rest.endswith('weight') else 'bias'
                q, kk, vv = v.chunk(3, dim=0)
                out[base + f'self_attn.q_proj.{kind}'] = q
                out[base + f'self_attn.k_proj.{kind}'] = kk
                out[base + f'self_attn.v_proj.{kind}'] = vv
            else:
                m = {'attn.out_proj.weight': 'self_attn.out_proj.weight', 'attn.out_proj.bias': 'self_attn.out_proj.bias',
                     'ln_1.weight': 'layer_norm1.weight', 'ln_1.bias': 'layer_norm1.bias',
                     'ln_2.weight': 'layer_norm2.weight', 'ln_2.bias': 'layer_norm2.bias',
                     'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                     'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias'}
                out[base + m[rest]] = v
        else:
            raise KeyError(k)
    return out


def load_model(name):
    """name: openai | sarclip_gelu | sarclip_quickgelu"""
    from transformers import CLIPModel, CLIPConfig, CLIPTokenizer
    from safetensors.torch import load_file
    tok = CLIPTokenizer.from_pretrained(PRETRAINED / 'openai_clip_vitl14')
    if name == 'openai':
        model = CLIPModel.from_pretrained(PRETRAINED / 'openai_clip_vitl14')
    else:
        cfg = CLIPConfig.from_pretrained(PRETRAINED / 'openai_clip_vitl14')
        act = 'gelu' if name == 'sarclip_gelu' else 'quick_gelu'
        cfg.text_config.hidden_act = act
        cfg.vision_config.hidden_act = act
        model = CLIPModel(cfg)
        sd = convert_openclip_visual(load_file(PRETRAINED / 'sarclip_vitl14' / 'model.safetensors'))
        sd = {k: v for k, v in sd.items() if k != 'text_model.embeddings.position_ids'}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith('position_ids')]
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.eval().to(device()), tok


@torch.no_grad()
def encode_images(model, u8, bs=64):
    """u8: (N,H,W) uint8 grayscale -> (pooled 1024, projected+normalised 768)."""
    dev = device()
    pooled, proj = [], []
    for i in range(0, len(u8), bs):
        x = torch.from_numpy(u8[i:i + bs]).float().div(255)[:, None].repeat(1, 3, 1, 1)
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
        x = ((x - MEAN) / STD).to(dev)
        o = model.vision_model(pixel_values=x)
        p = o.pooler_output
        pooled.append(p.cpu())
        proj.append(F.normalize(model.visual_projection(p), dim=-1).cpu())
    return torch.cat(pooled).numpy(), torch.cat(proj).numpy()


@torch.no_grad()
def encode_texts(model, tok, prompts_per_class):
    """prompts_per_class: list[list[str]] -> (C,768) averaged, normalised."""
    dev = device()
    W = []
    for prompts in prompts_per_class:
        t = tok(prompts, padding=True, return_tensors='pt').to(dev)
        pooled = model.text_model(input_ids=t['input_ids'], attention_mask=t['attention_mask']).pooler_output
        e = F.normalize(model.text_projection(pooled), dim=-1).mean(0)
        W.append(F.normalize(e, dim=0).cpu())
    return torch.stack(W).numpy()


def tmpl(c):
    return [f'this is an SAR image of {c} .', f'an SAR image shows {c} .',
            f'the SAR image contains a scene with {c} .']


# ----------------------------------------------------------------------------- sanity
def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def stage_sanity():
    meta = json.load(open(OUT / 'meta.json'))
    centre, corner, full = np.load(OUT / 'centre.npy'), np.load(OUT / 'corner.npy'), np.load(OUT / 'full.npy')
    rng = np.random.default_rng(SEED)
    # clean open-water chips with no other ship in the chip -> corner crop should be empty sea
    idx = [i for i, m in enumerate(meta) if m['clean'] and m['n_other'] == 0 and m['length_m'] >= 50]
    idx = rng.choice(idx, size=min(1000, len(idx)), replace=False)
    objects = ['a ship', 'an aircraft', 'storage tanks', 'vehicles', 'a bridge', 'buildings']
    scenes = ['water surfaces', 'urban areas', 'arable lands', 'forests']
    results = {}
    for name in ['openai', 'sarclip_gelu', 'sarclip_quickgelu']:
        model, tok = load_model(name)
        W_obj = encode_texts(model, tok, [tmpl(c) for c in objects])
        W_ship_water = encode_texts(model, tok, [tmpl('a ship') + ['one solitary marine craft is visible in the center region .'],
                                                 tmpl('water surfaces')])
        W_scene = encode_texts(model, tok, [tmpl(c) for c in scenes])
        _, e_c = encode_images(model, centre[idx])
        _, e_k = encode_images(model, corner[idx])
        _, e_f = encode_images(model, full[idx])
        s = lambda e: (e @ W_ship_water.T) @ np.array([1, -1])
        r = {
            'AUC ship-centre vs empty-corner (128px)': auc(s(e_c), s(e_k)),
            'top1=ship among objects, centre crop': float(((e_c @ W_obj.T).argmax(1) == 0).mean()),
            'top1=ship among objects, full chip': float(((e_f @ W_obj.T).argmax(1) == 0).mean()),
            'top1=water among scenes, corner crop': float(((e_k @ W_scene.T).argmax(1) == 0).mean()),
            'mean cos centre vs corner (lower = more distinct)': float((e_c * e_k).sum(1).mean()),
        }
        results[name] = r
        print(f'\n== {name}  (n={len(idx)})')
        for k, v in r.items():
            print(f'  {k:52s} {v:.3f}')
        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    json.dump(results, open(OUT / 'sanity.json', 'w'), indent=2)


# ----------------------------------------------------------------------------- extract
def stage_extract(names=('openai', 'sarclip_gelu')):
    centre = np.load(OUT / 'centre.npy')
    for name in names:
        model, _ = load_model(name)
        pooled, proj = encode_images(model, centre)
        np.save(OUT / f'feat_{name}_pooled.npy', pooled)
        np.save(OUT / f'feat_{name}_proj.npy', proj)
        print('extracted', name, pooled.shape, flush=True)
        del model


# ----------------------------------------------------------------------------- eval
ZS_NAMES = {'Tanker': ['a tanker', 'an oil tanker', 'a tanker ship'],
            'Cargo': ['a cargo ship', 'a container ship', 'a bulk carrier'],
            'Fishing': ['a fishing boat', 'a fishing vessel', 'a trawler'],
            'Tug/Special': ['a tugboat', 'a tug', 'a supply vessel']}


def setting_rows(meta, labels, classes, min_conf):
    rank = {'HIGH': 2, 'MEDIUM': 1}
    rows, y, g = [], [], []
    for i, m in enumerate(meta):
        L = labels.get(m['mmsi'])
        if not L or L['label'] not in classes or rank.get(L['confidence'], 0) < rank[min_conf]:
            continue
        rows.append(i); y.append(classes.index(L['label'])); g.append(m['mmsi'])
    return np.array(rows), np.array(y), np.array(g)


def stage_eval():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    meta = json.load(open(OUT / 'meta.json'))
    labels = json.load(open(LABELS))
    settings = {'3cls_HIGH (paper headline)': (['Tanker', 'Cargo', 'Fishing'], 'HIGH'),
                '4cls_MEDIUM+': (['Tanker', 'Cargo', 'Tug/Special', 'Fishing'], 'MEDIUM')}
    report = {}
    for sname, (classes, conf) in settings.items():
        rows, y, g = setting_rows(meta, labels, classes, conf)
        print(f'\n######## {sname}: {len(rows)} chips, {len(set(g))} vessels, '
              + ', '.join(f'{c}={int((y == i).sum())}ch/{len(set(g[y == i]))}v' for i, c in enumerate(classes)))
        for name in ['openai', 'sarclip_gelu']:
            pooled = np.load(OUT / f'feat_{name}_pooled.npy')[rows]
            proj = np.load(OUT / f'feat_{name}_proj.npy')[rows]
            # zero-shot
            model, tok = load_model(name)
            W = encode_texts(model, tok, [sum((tmpl(n) for n in ZS_NAMES[c]), []) for c in classes])
            del model
            pz = (proj @ W.T).argmax(1)
            r = {'zero-shot OA': accuracy_score(y, pz), 'zero-shot macroF1': f1_score(y, pz, average='macro'),
                 'zero-shot pred dist': np.bincount(pz, minlength=len(classes)).tolist()}
            # linear probe, 5-fold: grouped by MMSI vs plain chip-level
            for split_name, splitter, groups in [
                    ('probe MMSI-grouped', StratifiedGroupKFold(5, shuffle=True, random_state=SEED), g),
                    ('probe chip-level (leaky)', StratifiedKFold(5, shuffle=True, random_state=SEED), None)]:
                pred = np.zeros_like(y)
                for tr, te in splitter.split(pooled, y, groups):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))
                    clf.fit(pooled[tr], y[tr]); pred[te] = clf.predict(pooled[te])
                r[f'{split_name} OA'] = accuracy_score(y, pred)
                r[f'{split_name} macroF1'] = f1_score(y, pred, average='macro')
                r[f'{split_name} perclassF1'] = f1_score(y, pred, average=None).round(3).tolist()
            report[f'{sname} | {name}'] = r
            print(f'  -- {name}')
            for k, v in r.items():
                print(f'     {k:32s} {v if isinstance(v, list) else round(float(v), 3)}')
    json.dump(report, open(OUT / 'eval.json', 'w'), indent=2, default=float)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'sanity'
    {'cache': stage_cache, 'sanity': stage_sanity, 'extract': stage_extract, 'eval': stage_eval}[stage]()
