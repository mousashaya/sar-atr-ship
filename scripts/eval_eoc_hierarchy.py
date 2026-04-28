"""
scripts/eval_eoc_hierarchy.py
==============================
Hierarchical inference on EOC_depression without angle routing.

Pipeline:
  chip → 4-class (Bus/Car/Truck/Special)
       → 21-class filtered to valid subtypes for predicted main class
       → 40-class filtered to valid vehicles for predicted subtype
       → final vehicle prediction

All three classifiers trained on 15° SOC data only — within EOC-Depression rules.
No angle routing, no translation — pure hierarchical filtering.

The hypothesis: narrowing the problem at each stage makes the final classifier
more robust to depression angle change than flat 40-class classification.

Usage:
    python3 scripts/eval_eoc_hierarchy.py
"""

import importlib
import json
import sys
from pathlib import Path
from collections import defaultdict

import psycopg2
import psycopg2.extras
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from db_config import DB_CONFIG

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved'
EOC_ANGLES = [30.0, 45.0, 60.0]

# Full hierarchy from DB query
HIERARCHY = {
    'Bus': {
        'Large_Bus':  ['Yutong_ZK6120HY1'],
        'Medium_Bus': ['Dongfeng_EQ6608LTV', 'MAXUS_V80', 'Yangzi_YZK6590XCA'],
        'Small_Bus':  ['Buick_GL8', "Chang'an_Starlight_4500",
                       'Dongfeng_Forthing_Lingzhi', 'Wuling_Rongguang_V'],
    },
    'Car': {
        'Compact_SUV': ['Jeep_Patriot', 'Lincoln_MKC', 'Mitsubishi_Outlander_2003'],
        'Large_Car':   ['Great_Wall_Voleex_C50', 'Hongqi_CA7180A3E', 'Hongqi_h5'],
        'Medium_Car':  ['Buick_Excelle_GT', 'Chery_Arrizo 5'],
        'Medium_SUV':  ["Chang'an_CS75_Plus", 'Changfeng_Cheetah_CFA6473C',
                        'Chevrolet_Blazer_1998'],
        'Mini_Car':    ['Chery_qq3', 'Hawtai_EV160B'],
    },
    'Speacial': {
        'Ambulance':   ['JINBEI_SY5033XJH'],
        'ECV':         ['Lveco_Proud_2009'],
        'Road_Roller': ['Changlin_8228-5'],
        'Shovel_Loader': ['SDLG_ZL40F'],
    },
    'Truck': {
        'Forklift':   ['Hyundai_HLF25_II'],
        'Heavy_DT':   ['Dongfeng_Tianjin_DFH2200B'],
        'Heavy_FT':   ['FAW_J6P', 'SHACMAN_DeLong_M3000'],
        'Heavy_ST':   ['Dongfeng_Tianjin_KR230', 'SHACMAN_DeLong_X3000'],
        'Light_DT':   ['FAW_Jiabao_T51', 'Foton_BJ1045V9JB5-54', 'WAW_Aochi_1800'],
        'Light_PV':   ['JAC_Junling'],
        'Medium_TT':  ['Dongfeng_Duolika', 'WAW_Aochi_Hongrui'],
        'Mixer_Truck': ['CNHTC_HOWO'],
        'Pickup':     ['Great_Wall_poer', 'Huanghai_N1'],
    },
}

# Reverse lookups
VEHICLE_TO_SUBCLASS  = {v: sub  for main, subs in HIERARCHY.items()
                                 for sub, vehs in subs.items()  for v in vehs}
VEHICLE_TO_MAINCLASS = {v: main for main, subs in HIERARCHY.items()
                                 for sub, vehs in subs.items()  for v in vehs}
SUBCLASS_TO_MAINCLASS = {sub: main for main, subs in HIERARCHY.items()
                                    for sub in subs}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class EOCDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['local_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (128, 128), color=(64, 64, 64))
        if self.transform:
            img = self.transform(img)
        return img, rec['true_class'], rec['true_angle']


def load_models(device):
    model_module = importlib.import_module('models.convnext')

    def load(exp_name):
        save_dir    = MODELS_DIR / exp_name
        cfg         = json.loads((save_dir / 'config.json').read_text())
        class_names = cfg['class_names']
        model       = model_module.build_model(num_classes=len(class_names)).to(device)
        model.load_state_dict(
            torch.load(save_dir / 'weights.pt', map_location=device))
        model.eval()
        return model, class_names

    print("Loading 15°-trained classifiers...")
    m4,  cls4  = load('exp_006_convnext_4class_angle15_v1')
    m21, cls21 = load('exp_010_convnext_21class_angle15_v1')
    m40, cls40 = load('exp_014_convnext_40class_angle15_v1')
    print(f"  4-class:  {cls4}")
    print(f"  21-class: {len(cls21)} subtypes")
    print(f"  40-class: {len(cls40)} vehicles")
    return (m4, cls4), (m21, cls21), (m40, cls40)


def load_eoc_depression():
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id AS image_id, local_path, depression_angle, target_type
            FROM defense.atrnet_images
            WHERE experiment_setting = 'EOC_depression'
              AND local_path IS NOT NULL
              AND depression_angle IN (30, 45, 60)
              AND target_type IS NOT NULL AND target_type != ''
        """)
        rows = cur.fetchall()
    conn.close()
    return [{'image_id':   r['image_id'],
             'local_path': r['local_path'],
             'true_angle': float(r['depression_angle']),
             'true_class': r['target_type']} for r in rows]


def run_hierarchy(records, models, transform, device, batch_size=64):
    (m4, cls4), (m21, cls21), (m40, cls40) = models

    cls4_to_idx  = {c: i for i, c in enumerate(cls4)}
    cls21_to_idx = {c: i for i, c in enumerate(cls21)}
    cls40_to_idx = {c: i for i, c in enumerate(cls40)}

    dataset = EOCDataset(records, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, persistent_workers=True)

    all_true, all_pred = [], []
    all_true_angles    = []
    correct_at_4  = 0
    correct_at_21 = 0

    print(f"  Running hierarchy on {len(records):,} images...")

    with torch.no_grad():
        for imgs, true_classes, true_angles in loader:
            imgs = imgs.to(device)

            # Stage 1: 4-class prediction
            logits4  = m4(imgs)
            pred4_idx = logits4.argmax(dim=1).cpu().tolist()
            pred4     = [cls4[i] for i in pred4_idx]

            # Stage 2: 21-class with 4-class filtering
            logits21  = m21(imgs)
            probs21   = torch.softmax(logits21, dim=1).cpu()
            pred21    = []
            for i, main_cls in enumerate(pred4):
                # Only consider subtypes that belong to predicted main class
                valid_subs = list(HIERARCHY[main_cls].keys())
                valid_idx  = [cls21_to_idx[s] for s in valid_subs
                              if s in cls21_to_idx]
                if valid_idx:
                    best_sub_idx = max(valid_idx, key=lambda j: probs21[i, j])
                    pred21.append(cls21[best_sub_idx])
                else:
                    pred21.append(cls21[probs21[i].argmax().item()])

            # Stage 3: 40-class with 21-class filtering
            logits40  = m40(imgs)
            probs40   = torch.softmax(logits40, dim=1).cpu()
            pred40    = []
            for i, sub_cls in enumerate(pred21):
                # Only consider vehicles in the predicted subclass
                valid_vehs = HIERARCHY.get(
                    SUBCLASS_TO_MAINCLASS.get(sub_cls, ''), {}
                ).get(sub_cls, [])
                valid_idx = [cls40_to_idx[v] for v in valid_vehs
                             if v in cls40_to_idx]
                if valid_idx:
                    best_veh_idx = max(valid_idx, key=lambda j: probs40[i, j])
                    pred40.append(cls40[best_veh_idx])
                else:
                    pred40.append(cls40[probs40[i].argmax().item()])

            # Track accuracy at each level
            for i, tc in enumerate(true_classes):
                true_main = VEHICLE_TO_MAINCLASS.get(tc, '')
                true_sub  = VEHICLE_TO_SUBCLASS.get(tc, '')
                if pred4[i] == true_main:
                    correct_at_4 += 1
                if pred21[i] == true_sub:
                    correct_at_21 += 1

            all_true.extend(true_classes)
            all_pred.extend(pred40)
            all_true_angles.extend(true_angles.tolist())

    n = len(all_true)
    acc4  = correct_at_4  / n
    acc21 = correct_at_21 / n
    acc40 = sum(p == t for p, t in zip(all_pred, all_true)) / n

    return all_true, all_pred, all_true_angles, acc4, acc21, acc40


def main():
    device = get_device()
    print(f"\nDevice: {device}")

    model_module = importlib.import_module('models.convnext')
    transform    = model_module.get_transform()

    models = load_models(device)

    print("\nLoading EOC_depression images (30°/45°/60°)...")
    records = load_eoc_depression()
    print(f"  Total: {len(records):,} images")

    all_true, all_pred, all_angles, acc4, acc21, acc40 = run_hierarchy(
        records, models, transform, device)

    print(f"\n{'='*60}")
    print(f"  EOC-DEPRESSION — Hierarchical Inference")
    print(f"  (all classifiers trained on 15° SOC only)")
    print(f"{'='*60}")
    print(f"  Stage 1 — 4-class accuracy:   {acc4*100:.1f}%")
    print(f"  Stage 2 — 21-class accuracy:  {acc21*100:.1f}%")
    print(f"  Stage 3 — 40-class accuracy:  {acc40*100:.1f}%")
    print(f"\n  ATRNet-STAR SOTA (flat):       43.1%")
    print(f"  Our flat baseline:             37.6%")
    if acc40 * 100 > 43.1:
        print(f"  *** BEATS SOTA by {acc40*100 - 43.1:.1f} pp ***")

    # Per angle breakdown
    print(f"\n  Per depression angle:")
    for angle in EOC_ANGLES:
        idx  = [i for i, a in enumerate(all_angles) if a == angle]
        if not idx:
            continue
        tc   = [all_true[i] for i in idx]
        pc   = [all_pred[i] for i in idx]
        acc  = sum(p == t for p, t in zip(pc, tc)) / len(tc)
        print(f"    {angle:>4.0f}°:  {acc*100:.1f}%  ({len(idx):,} images)")

    # Save result
    out_path = Path(__file__).parent.parent / 'logs' / 'eoc_depression_hierarchy.json'
    out_path.write_text(json.dumps({
        'approach':      'hierarchical_inference_15deg_only',
        'acc_4class':    round(acc4  * 100, 1),
        'acc_21class':   round(acc21 * 100, 1),
        'acc_40class':   round(acc40 * 100, 1),
        'sota':          43.1,
        'flat_baseline': 37.6,
        'delta_vs_sota': round(acc40 * 100 - 43.1, 1),
        'n_images':      len(records),
    }, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
