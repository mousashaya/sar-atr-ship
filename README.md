# sar-atr-study

Studies of what SAR automatic target recognition (ATR) models actually learn, for ground vehicles and ships.

The common thread is evaluation rigor:
- background-only and target-only controls
- held-out depression angles at a fixed data budget
- vessel- and region-disjoint splits
- majority-class and metadata-only baselines
- 3 seeds per result

The goal is to show which parts of high reported accuracies come from the target and which come from shortcuts.

*(Formerly `sar-atr-ship`. Old links redirect here.)*

---

## Vehicles (MSTAR, ATRNet-STAR)

*How Much of SAR Vehicle Recognition Is the Vehicle? Background Shortcuts, Angle Gaps and Partial Libraries on MSTAR and ATRNet-STAR.* M. Shaya, in preparation; the arXiv link will be added here.

| Experiment | Question | Script |
|---|---|---|
| Background controls | How accurate is a model that sees only the background, or only the target? Does the mask position alone leak the class? | `scripts/train_mask.py` (ConvNeXt-Small or SARATR-X), `scripts/build_atr_cache.py`; small-CNN version on MSTAR: `scripts/eval_mstar_background.py` |
| Depression-angle gap | At a fixed number of chips per class, how does accuracy at a held-out angle depend on the training angles? | `scripts/train_angle_all.py`, `scripts/train_angle_heldout.py` |
| Partial library | If the other classes have training chips at the test angle and one class does not, what happens to that class? | `scripts/train_angle_heldout_class.py` |
| Vision–language features | Zero-shot and linear-probe CLIP vs. SARCLIP on MSTAR | `scripts/eval_clip_mstar.py` (model loading in `scripts/eval_sarclip.py`) |

```bash
# Background controls (data: mstar | atr_soc | atr_scene; cond: full | bg | tgt | maskpos)
python scripts/train_mask.py --data atr_soc --cond bg --size 80 --seed 0
python scripts/train_mask.py --arch saratrx --bs 64 --data atr_soc --cond bg --size 80 --seed 0

# Depression-angle gap: train on a set of angles, test on every other angle
python scripts/train_angle_all.py --train-angles 15 60 --per-class 230 --seed 0

# Partial library: train at 15 and 45 degrees, test at 45; fold 0 of the 40 classes has no 45-degree chips
python scripts/train_angle_heldout_class.py --train-angle 15 --test-angle 45 --fold 0 --seed 0
```

**Data:**
- MSTAR public release, from AFRL SDMS.
- ATRNet-STAR (Liu et al., IEEE TPAMI 2026), from the dataset authors. We use the official `SOC_40classes` and `EOC_scene` settings and bounding boxes, with no re-splitting.

Neither dataset is redistributed here.

**SARATR-X** (Li et al., IEEE TIP 2025):
- Put `checkpoint-800.pth` from https://huggingface.co/waterdisappear/SARATR-X and the HiViT architecture files from https://github.com/waterdisappear/SARATR-X in `models/saratrx_arch/`.
- `models/saratrx.py` wraps the encoder with a linear classification head.
- The SARATR-X weights are CC BY-NC 4.0.

---

## Ships (Sentinel-1 + AIS), in progress

Ship type from free Sentinel-1 10 m GRD imagery, labelled by matching AIS tracks across several US regions. Key questions:
- Does a CNN learn ship appearance, or location and size?
- Does it transfer across regions?

| Step | Scripts |
|---|---|
| AIS and scene download | `scripts/build_ais_new_regions.py`, `scripts/download_new_regions.py` |
| AIS-matched chipping | `scripts/chip_ships_regions.py`, `scripts/chip_ships_gulf.py` |
| Labels (AIS + Global Fishing Watch) | `scripts/lookup_gfw_vessel_types.py`, `scripts/resolve_vessel_labels_v2.py` |
| Training and cross-region tests | `scripts/train_ship_classify.py`, `scripts/train_cross_region.py` |

High-resolution Umbra chipping (`scripts/chip_umbra_ships.py`) supports a later resolution study.

---

## Earlier work

`sar_findings.md` documents an earlier ATRNet-STAR investigation, including cross-polarization leakage in the official split and hierarchical-routing experiments, along with the mistakes made in it. `SHIPRS_README.md` describes a planned optical ship study on ShipRSImageNet.

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision numpy pillow psycopg2-binary scikit-learn matplotlib open_clip_torch transformers
cp .env.example .env    # PostgreSQL connection settings, read by scripts/db_config.py
psql -f scripts/setup_db.sql
python scripts/ingest_mstar.py && python scripts/ingest_atrnet.py
```

Data and cache paths default to an external volume (`/Volumes/Machine Learning/...`); edit them at the top of each script. Each training run writes a JSON file with its arguments, metrics and test predictions to `logs/`.

## License

Code: MIT (see `LICENSE`). Datasets and third-party model weights keep their own licenses.

---

*Research by Mousa Shaya. AI assistance (Claude, Anthropic) was used for coding and manuscript drafting. All experimental design, analysis and conclusions are the author's own.*
