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

## Understanding the Code Structure (Start Here)

This section explains how the pieces fit together — written for someone
new to Python who wants to understand the project before modifying it.

---

### The Big Picture

```
Image files (external drive)
        ↓
   Database (PostgreSQL)  ← stores metadata, results, experiment records
        ↓
   dataset.py            ← loads images, applies train/val/test split
        ↓
   train.py              ← trains a model, saves weights + config to disk
        ↓
   models/saved/         ← where trained models live
        ↓
   eval_rotation.py      ← loads a trained model, tests it, prints results
```

Everything flows top to bottom. The database is the glue — it knows
where every image lives on disk and what class it belongs to.

---

### How a Model File Works

Every model lives in `models/` as its own Python file.
Example: `models/convnext_small.py`

Every model file has exactly two functions that the rest of the
pipeline expects:

```python
def build_model(num_classes):
    # Creates the neural network
    # num_classes = 40 for our vehicle classification
    # Returns the model object
    ...

def get_transform():
    # Describes how to prepare an image before feeding it to the model
    # Resize, normalize pixel values, convert to tensor
    # Returns a transform object
    ...
```

When you add ShayaSARNet, you create `models/shayasarnet.py` with
these same two functions. The rest of the pipeline works unchanged.

---

### How train.py Uses a Model

train.py is the main training script. When you run:

```bash
python3 scripts/train.py --model convnext_small --name my_experiment
```

train.py does this:
1. Reads `--model convnext_small`
2. Imports `models/convnext_small.py`
3. Calls `build_model(num_classes=40)` to create the network
4. Calls `get_transform()` to know how to prepare images
5. Loads images from DB using dataset.py
6. Trains for N epochs, saves best weights to `models/saved/exp_NNN_my_experiment/`
7. Writes `config.json` alongside the weights (stores accuracy, class names, settings)

You never need to modify train.py to add a new model — just create
the model file and pass `--model shayasarnet` on the command line.

---

### How dataset.py Works

dataset.py is responsible for one important job: making sure the
train/test split is fair (no data leakage).

It queries the database for all images matching your filters
(band, setting, etc), then groups them by acquisition key:
  (target_type, band, depression_angle, azimuth_angle)

All images in the same acquisition group go to the same split.
This is what prevents cross-polarization leakage.

You never call dataset.py directly — train.py calls it internally.

---

### What Lives in models/saved/

After training, each experiment gets its own folder:

```
models/saved/
  exp_108_rot_baseline_full_v1/
    weights.pt     ← the trained model weights (the "brain")
    config.json    ← settings, class names, best accuracy
```

config.json looks like:
```json
{
  "class_names": ["Buick_Excelle_GT", "Buick_GL8", ...],
  "best_val_acc": 97.2,
  "model_type": "convnext_small",
  "setting": "SOC_40classes"
}
```

The eval scripts read config.json to know how the model was built,
then load weights.pt to get the trained weights.

---

### How eval_rotation.py Works

eval_rotation.py demonstrates rotation sensitivity:

1. Loads a trained model from models/saved/
2. Loads test images from the DB
3. For each rotation angle (0°, 15°, 30° ... 180°):
   - Rotates each image on-the-fly in memory (no files saved to disk)
   - Feeds rotated images through the model
   - Records accuracy
4. Prints the table showing accuracy drop at each angle

The rotation happens inside the Dataset class in `__getitem__` —
this is called automatically for each image as it's loaded.

---

### What You Touch When Adding ShayaSARNet

Only ONE new file needed:
```
models/shayasarnet.py    ← create this
```

Everything else — train.py, dataset.py, eval_rotation.py, the DB,
the pipeline — stays exactly the same. That's the payoff of the
existing structure.

Then train it:
```bash
python3 scripts/train.py --model shayasarnet --band X \
    --config 40class --setting SOC_40classes \
    --epochs 30 --name shayasarnet_v1
```

Then eval it:
```bash
python3 scripts/eval_rotation.py --model shayasarnet_v1
```

Compare the curve to rot_baseline_full_v1. That's the experiment.

---

## Script Walkthroughs

How each key script is structured internally — plain English, no jargon.

---

### train.py — The Training Script

This is the most important script. When you run it, here is what
happens in order:

```
1. READ ARGUMENTS
   Reads everything you typed on the command line (--model, --epochs, etc.)

2. SET UP DEVICE
   Checks: is MPS (Mac GPU) available? If yes use it.
   If not, try CUDA (Nvidia GPU). If not, use CPU.

3. LOAD THE MODEL
   Reads --model flag (e.g. "convnext_small")
   Imports models/convnext_small.py
   Calls build_model(num_classes=40) → creates the neural network

4. CREATE EXPERIMENT IN DB
   Writes one row to defense.experiments table
   Gets back a unique ID (e.g. 108) — this is the experiment number

5. LOAD DATA (via dataset.py)
   Queries the DB for all matching images
   Applies acquisition-key split → train / val / test groups
   Creates DataLoader objects — these feed images to the model in batches

6. TRAINING LOOP (repeats for each epoch)
   For each batch of images:
     - Feed images through the model → get predictions (logits)
     - Compare predictions to true labels → calculate loss
     - Backpropagate — adjust weights to reduce loss
     - Repeat for all batches
   At end of epoch: evaluate on validation set
   If val accuracy improved: save weights to disk
   Send email notification
   If no improvement for N epochs (patience): stop early

7. FINAL EVALUATION
   Load best saved weights
   Run on test set → report TEST accuracy
   Save per-image predictions to DB
   Save per-class metrics (precision/recall/F1) to DB
   Write config.json alongside weights
```

Key concept — **loss**: a number measuring how wrong the model's
predictions are. Training tries to make this number smaller.
Lower loss = better predictions.

Key concept — **epoch**: one complete pass through all training images.
Each epoch the model gets slightly better (loss goes down).

Key concept — **backpropagation**: the math that figures out which
weights to adjust and by how much to reduce the loss. You never
touch this — PyTorch handles it with `loss.backward()`.

---

### dataset.py — The Data Loader

This script does one job: given a list of image records from the DB,
split them fairly into train/val/test.

```
1. QUERY DB
   Gets all images matching your filters (band, setting, etc.)
   Each image record has: file path, true class, acquisition metadata

2. GROUP BY ACQUISITION KEY
   Groups images that came from the same physical measurement:
   (target_type, band, depression_angle, azimuth_angle)
   These groups must never be split across train/test

3. SHUFFLE AND SPLIT
   Uses a fixed random seed (42) so results are reproducible
   85% of acquisition groups → training pool
   From that pool: 85% train, 15% val
   Remaining 15% of acquisition groups → test

4. CREATE PYTORCH DATASETS
   Wraps each split in a Dataset object
   Dataset.__getitem__ is called for each image:
     - Opens the image file
     - Applies the transform (resize, normalize)
     - Returns (image tensor, label, image_id)
```

Key concept — **seed**: a starting number for random operations.
Same seed = same split every time = reproducible experiments.

Key concept — **DataLoader**: a wrapper that feeds images to the
model in batches, in parallel, using multiple CPU workers.

---

### eval_rotation.py — The Rotation Sensitivity Script

```
1. LOAD MODEL
   Finds the experiment folder in models/saved/
   Reads config.json to get class names and model type
   Loads weights.pt into the model

2. LOAD TEST IMAGES FROM DB
   Same acquisition-key split logic as dataset.py
   Only loads the test portion (the held-out 15%)

3. FOR EACH ROTATION ANGLE (0°, 15°, 30° ... 180°)
   Creates a RotatedDataset — same as normal dataset but:
     __getitem__ rotates the image before the standard transform
     Rotation is done with PIL (Python image library) on the fly
     No rotated images saved to disk — generated in memory, used, discarded

4. RUN INFERENCE
   Feeds all (rotated) test images through the model
   Records predicted class for each image
   Compares to true class → calculates accuracy

5. PRINT TABLE
   Shows accuracy at each rotation angle
   Shows drop vs 0° baseline
   Saves results to logs/rotation_sensitivity.json
```

Key concept — **inference**: running images through a trained model
to get predictions. No learning happens — weights stay fixed.
`torch.no_grad()` tells PyTorch not to track gradients (faster, less memory).

Key concept — **PIL rotation**: `image.rotate(45)` rotates the image
45° counterclockwise, fills the corners with black. Simple, fast,
no disk I/O.

---

### models/convnext_small.py — A Model File (Example)

Every model file follows the same pattern:

```python
def build_model(num_classes):
    # 1. Load ConvNeXt-Small pretrained on ImageNet
    model = torchvision.models.convnext_small(pretrained=True)

    # 2. Replace the final layer (1000 ImageNet classes → our N classes)
    model.classifier[2] = nn.Linear(768, num_classes)

    return model

def get_transform():
    # Describes image preprocessing:
    # - Resize to 224×224 (ConvNeXt's expected input size)
    # - Convert to tensor (numpy array → PyTorch tensor)
    # - Normalize pixel values to ImageNet mean/std
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
```

When you build shayasarnet.py, it will have the same two functions.
The internals will use e2cnn layers instead of ConvNeXt layers —
but the interface is identical. That's the design.

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
