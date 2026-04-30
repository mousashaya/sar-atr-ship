# How to Use This Project

A practical guide for returning to this project after time away.
Written for future-you coming back cold after weeks or months.

---

## What This Project Is

Two ML research tracks in one codebase:

1. **ATRNet-STAR** — SAR (radar) vehicle recognition. 40 vehicle classes, 194K images.
   Found data leakage in the official benchmark split. Investigated flat vs cascade
   classifiers across multiple EOC (Extended Operating Condition) challenges.

2. **ShipRSImageNet** — Optical satellite ship recognition. 50 ship classes, 17K instances.
   Next research track after SAR work concluded.

Everything runs locally. No cloud. Models train on your Mac (MPS).

---

## Prerequisites

### External Drive
ShipRS images live on an external drive. Mount it before any ShipRS training:
```
/Volumes/Machine Learning/ShipRSImageNet/ShipRSImageNet_V1/VOC_Format/JPEGImages/
```
ATRNet-STAR images also live on external storage — check `local_path` in the DB
for the exact mount point.

### Environment
```bash
cd /Users/mousashaya/Projects/defense
source venv/bin/activate
```

### Database
PostgreSQL server must be running. The database is `fungidb` (shared with other
projects), with all defense data in the `defense` schema.

Tables:
- `defense.atrnet_images` — ATRNet-STAR image metadata
- `defense.experiments` — one row per training run (exp_097, exp_098, etc.)
- `defense.experiment_epochs` — per-epoch train loss and val accuracy
- `defense.experiment_metrics` — per-class precision/recall/F1 per split
- `defense.experiment_predictions` — per-image predictions (confusion matrix drill-down)
- `defense.shiprs_images` — ShipRS image metadata

### .env File
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fungidb
DB_USER=cluser
DB_PASSWORD=clPwd
GMAIL_USER=mshaya01.ml@gmail.com
GMAIL_APP_PASSWORD=<app password>
NOTIFY_EMAIL=mousa.shaya@gmail.com
```
Train.py emails you after every epoch and when training finishes.

---

## Order of Operations (Starting Fresh)

### ATRNet-STAR
```
1. Ingest data into DB:
   python3 scripts/ingest_atrnet.py

2. Train a model:
   python3 scripts/train.py --model convnext_small --config 40class \
       --setting SOC_40classes --epochs 15 --patience 3 \
       --batch-size 64 --lr 0.0001 --name my_experiment_v1

3. Run the viewer to inspect results:
   python3 run_viewer.py
   # Open http://localhost:5051
```

### ShipRS
```
1. Mount external drive (ShipRS images)

2. Ingest data into DB:
   python3 scripts/ingest_shiprs.py

3. Train (same train.py, different setting — TBD when ShipRS training is set up)

4. View results at http://localhost:5051/ships
```

---

## The Viewer

```bash
python3 run_viewer.py
# Open http://localhost:5051
```

Pages:
- `/` — ATRNet-STAR image browser (filter by class, angle, band, pol, scene)
- `/experiments` — All training runs with accuracy, confusion matrix, per-class metrics
- `/ships` — ShipRS image browser

The viewer reads live from the DB — no restart needed after new experiments.

---

## Training Scripts

### Main training script
```bash
python3 scripts/train.py \
    --model convnext_small \   # convnext, convnext_small, convnext_tiny, efficientnet_b2
    --config 40class \         # 4class (main_class), 21class (subclass), 40class (target_type)
    --setting SOC_40classes \  # SOC_40classes, EOC_depression, EOC_pol, EOC_scene, etc.
    --epochs 15 \
    --patience 3 \             # early stopping: stop if no improvement for N epochs
    --batch-size 64 \
    --lr 0.0001 \
    --name "my_experiment_v1"  # required, must be unique

# Filters (can combine):
    --angle 15                 # depression angle only (15, 30, 45, 60)
    --polarization HH          # one polarization (HH, HV, VH, VV)
    --band X                   # one radar band (X, KU)
    --azimuth-min 0 \
    --azimuth-max 90           # azimuth range
    --parent-class Truck       # images where main_class = Truck (cascade L2)
    --use-atrnet-splits        # use official ATRNet split instead of our clean split
                               # WARNING: official split has known leakage — avoid
    --augment                  # random flips and rotation
```

### Where models are saved
```
models/saved/exp_{ID}_{name}/
    weights.pt      — best model weights (by val accuracy)
    config.json     — hyperparameters, class names, best val accuracy
```

Experiment ID is auto-assigned by the DB (sequential: 97, 98, 99...).

### Training logs
```
logs/eoc_depression.log    — EOC depression experiment log
logs/soc_cascade.log       — SOC cascade experiment log
logs/eoc_depression_cascade.json  — cascade eval results
logs/soc_cascade.json             — SOC cascade eval results
```

---

## Overnight Run Scripts

Run these with `caffeinate` to prevent sleep:
```bash
# EOC depression (flat + cascade):
caffeinate -dim bash scripts/run_eoc_depression.sh 2>&1 | tee logs/eoc_depression.log

# EOC pol + scene flat baselines:
caffeinate -dim bash scripts/run_eoc_pol_scene.sh 2>&1 | tee -a logs/eoc_depression.log

# SOC cascade (all routing variants, ~18-24 hours):
caffeinate -dim bash scripts/run_soc_cascade.sh 2>&1 | tee logs/soc_cascade.log
```

Chain two scripts to run sequentially overnight:
```bash
# Wait for first script to finish then start second:
while pgrep -f "run_first.sh" > /dev/null; do sleep 60; done && \
    caffeinate -dim bash scripts/run_second.sh
```

---

## Evaluation Scripts

```bash
# Cascade evaluation for EOC_depression:
python3 scripts/eval_eoc_cascade.py

# SOC cascade vs flat comparison:
python3 scripts/eval_soc_cascade.py
```

---

## The Data Split — Why It Matters

The official ATRNet-STAR split has **data leakage**: polarization channels
(HH/HV/VH/VV) of the same physical vehicle measurement are split independently,
so the model can recognize a test image because it trained on the same vehicle
at a different polarization.

Our fix: **acquisition-key split** — all images sharing
(target_type, band, depression_angle, azimuth_angle) stay on the same side.
This is the default when you run `train.py` without `--use-atrnet-splits`.

Never use `--use-atrnet-splits` for reporting results. It exists only for
replicating paper baselines.

---

## ATRNet-STAR EOC Conditions

Each EOC condition restricts training data and tests on held-out conditions:

| Setting | Train on | Test on |
|---|---|---|
| EOC_depression | 15° only | 30°, 45°, 60° |
| EOC_pol | HH only | HV, VH, VV |
| EOC_band | X-band only | KU-band |
| EOC_scene | simple scenes | factory, urban, woodland |
| EOC_azimuth | 0°-60° azimuth | 60°-360° azimuth |

**Common mistake:** Training a SOC model (all conditions) and evaluating on
EOC test images. This gives ~98% accuracy but is meaningless — the model
trained on the test conditions. Always verify what's in your train set.

---

## ShipRS Setup

**Data location:** `/Volumes/Machine Learning/ShipRSImageNet/ShipRSImageNet_V1/`

**What we have:** V1 only (train + val splits). V1.1 test set unavailable
(hosted on Baidu Pan, inaccessible outside China).

**Our split strategy:** Image-level stratified split from V1 train data.
70% train / 15% val / 15% test, seed=42, stratified by level_3 class.
All instances in one image stay together — never split at instance level.

**Sparse classes:** 10 level_3 classes have fewer than 50 instances.
Merge into parent level_2 "Other" categories before training.

See `SHIPRS_README.md` for full class hierarchy and dataset details.

---

## Checking Experiment Results

Quick ways to check what ran and how it did:

```bash
# List all experiments with accuracy:
PGPASSWORD=clPwd psql -h localhost -U cluser -d fungidb -c "
SELECT id, name, setting, best_val_acc
FROM defense.experiments ORDER BY id DESC LIMIT 20;"

# Check a specific experiment's test accuracy from log:
grep "TEST accuracy" logs/eoc_depression.log

# Check cascade results:
cat logs/eoc_depression_cascade.json
cat logs/soc_cascade.json
```

---

## Key Files Reference

| File | Purpose |
|---|---|
| `README.md` | Project overview and results summary |
| `HOWTO.md` | This file — operational guide |
| `SHIPRS_README.md` | ShipRS dataset details and audit findings |
| `sar_findings.md` | Full SAR research write-up (findings + mistakes) |
| `paper_draft.md` | ATRNet-STAR leakage paper draft |
| `LITERATURE.md` | Key papers and references |
| `sbir_opportunities.md` | SBIR funding opportunities |
| `scripts/dataset.py` | Data loading, acquisition-key split logic |
| `scripts/train.py` | Main training script |
| `scripts/ingest_atrnet.py` | Load ATRNet-STAR into DB |
| `scripts/ingest_shiprs.py` | Load ShipRS into DB |
| `.env` | DB credentials and email config (not in git) |
| `run_viewer.py` | Start the web viewer |
