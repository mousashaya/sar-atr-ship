# Defense ML Research

SAR automatic target recognition and optical ship recognition research.
Investigating data leakage, hierarchical classification, and routing strategies
on two benchmarks: ATRNet-STAR (SAR vehicles) and ShipRSImageNet (optical ships).

---

## Projects

### 1. ATRNet-STAR SAR ATR

Investigation of the ATRNet-STAR benchmark — 194,324 SAR images across 40 vehicle
classes, 4 depression angles, 4 polarizations, and 2 radar bands.

**Key finding:** The official train/test split has cross-polarization data leakage.
Polarization channels (HH/HV/VH/VV) of the same physical acquisition are split
independently, allowing models to exploit cross-channel correlations rather than
learning generalizable vehicle features.

See [`sar_findings.md`](sar_findings.md) for the full write-up including mistakes made.

**Results summary:**

| Condition | Ours (clean split) | Paper | Notes |
|---|---|---|---|
| SOC accuracy | 95.8% | 96.4% | Leakage inflated paper by ~0.6% |
| EOC_depression | 40.0% | 37.2% | Genuine difficulty, leakage didn't help here |
| EOC_pol | 79.8% | 70.5% | Clean split beats paper by ~9% |
| EOC_scene | 26.2% | 18.0% | Inconclusive — needs investigation |

**Cascade finding:** Hierarchical routing (main class → vehicle type) fails under
EOC_depression because the routing signal itself breaks down across unseen angles.
64% L1 routing accuracy at test angles vs 97% at training angle.

**SOC cascade experiment** (in progress): Tests whether routing by physical property
(band, polarization, depression angle, azimuth) helps under fair conditions where
all angles are available for training.

---

### 2. ShipRSImageNet

Fine-grained ship recognition in optical satellite imagery. 3,435 images,
17,573 instances, 50 ship categories across a 4-level hierarchy
(Warship → Destroyer → Arleigh Burke DD).

See [`SHIPRS_README.md`](SHIPRS_README.md) for dataset details and audit findings.

**Status:** Planned. Waiting for SAR experiments to complete.

**Key questions:**
- Does the Warship → type → specific class hierarchy help vs flat classification?
- Does routing by ship category improve fine-grained warship recognition?
- How do environmental conditions (fog, glint, dark) affect accuracy?

---

## Repository Structure

```
scripts/
  train.py                  — Main training script (ConvNeXt, acquisition-key split)
  dataset.py                — Data loading with leakage-safe acquisition-key split
  eval_eoc_cascade.py       — EOC depression cascade evaluation
  eval_soc_cascade.py       — SOC cascade vs flat evaluation
  run_eoc_depression.sh     — EOC depression flat + cascade training
  run_eoc_pol_scene.sh      — EOC pol + scene flat training
  run_soc_cascade.sh        — SOC cascade experiment (all routing variants)

models/
  convnext.py               — ConvNeXt-Base
  convnext_small.py         — ConvNeXt-Small
  convnext_tiny.py          — ConvNeXt-Tiny
  saved/                    — Trained model weights and configs

app/
  routes.py                 — Flask viewer routes
  db.py                     — Database queries
  templates/                — Viewer HTML templates

data/
  atrnet.db                 — (local SQLite reference, main data in Postgres)

logs/                       — Training logs and JSON result summaries
```

## Key Design Decisions

**Acquisition-key split:** All images sharing (target_type, band, depression_angle,
azimuth_angle) stay on the same side of the train/test boundary. Prevents
cross-polarization leakage identified in Gao et al. (IEEE GRSL 2026).

**Model:** ConvNeXt-Small pretrained on ImageNet, fully fine-tuned. SAR images
loaded as 3-channel repeats. Input 224×224.

**Training:** AdamW lr=1e-4, batch 64, early stopping patience=3.

---

## Papers and References

See [`LITERATURE.md`](LITERATURE.md) for full reference list.

Key references:
- ATRNet-STAR dataset: Liu et al., submitted to IEEE TPAMI
- Data leakage (SOC): Gao et al., IEEE GRSL 2026
- SARATR-X foundation model: Li et al., IEEE TIP 2025
- ConvNeXt: Liu et al., CVPR 2022
- ShipRSImageNet: Zhang et al., IEEE JSTARS 2021

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision psycopg2-binary pillow scikit-learn flask

# Database: PostgreSQL with defense schema
# Copy .env.example to .env and fill in DB credentials
```

---

*Research by Mousa Shaya. AI assistance (Claude, Anthropic) used for coding and
manuscript drafting. All experimental design, analysis, and conclusions are the
author's own.*
