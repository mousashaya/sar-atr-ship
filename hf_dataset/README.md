---
license: cc-by-4.0
task_categories:
- image-classification
- image-to-image
language:
- en
tags:
- SAR
- sentinel-1
- ship-classification
- ship-re-identification
- AIS
- maritime
- remote-sensing
size_categories:
- 10K<n<100K
---

# SAR Ship Gulf — Sentinel-1 GRD Ship Chips with AIS Labels

## Dataset Summary

18,342 dual-polarisation (VV + VH) Sentinel-1 IW GRD ship chips from the Gulf of Mexico,
automatically labeled using a four-signal AIS label resolver. No manual annotation was
used at any stage.

Each chip is a 512×512 float32 array centred on an AIS-matched vessel position,
paired with the vessel's MMSI (identity), resolved vessel type, and
chip-level metadata (length, speed, heading, acquisition time).

This dataset accompanies the paper:

> M. Shaya, "Zero-Curation SAR Ship Classification and Re-Identification
> via Multi-Signal AIS Label Resolution and SAR-Native Backbone,"
> *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, 2026 (under review).

---

## Dataset Statistics

| | |
|---|---|
| Total chips | 18,342 |
| Unique vessels (MMSI) | 2,088 |
| Vessels with ≥2 passes | 1,942 (93%) |
| Resolved labels (HIGH + MEDIUM) | 1,823 (87.3%) |
| Unresolved | 265 (12.7%) |

**Class distribution (HIGH-confidence only):**

| Class | Vessels | Chips |
|---|---|---|
| Tanker | 544 | ~1,368 |
| Cargo | 232 | ~500 |
| Fishing | 14 | ~71 |
| Tug/Special | 0 (MEDIUM only) | — |

---

## Data Sources

- **SAR imagery:** Sentinel-1 IW GRD (free, Copernicus programme), Gulf of Mexico, 2024
- **AIS data:** NOAA National AIS (free, public domain), matched ±10 min to SAR acquisition
- **GFW labels:** Global Fishing Watch Vessel Identity API (one of four resolver signals)

---

## Label Resolver

Raw AIS vessel type codes are unreliable. This dataset uses a four-signal voting resolver:

1. **AIS type code** — modal code across all observations mapped to class
2. **Ship length** — extreme lengths (≥200 m → Tanker, <50 m → Tug) cast a vote
3. **GFW classification** — Global Fishing Watch API vessel type
4. **Vessel name keywords** — curated lists (LNG, TANKER, BULK, CONTAINER, TUG, etc.)

Three or more agreeing signals → **HIGH** confidence.
Two agreeing with no dissenter → **HIGH**.
Two with a dissenter, or one uncontested → **MEDIUM**.
Fewer than two non-conflicting → **UNRESOLVED**.

87.3% of vessels are resolved. Only HIGH-confidence labels are used for classification experiments.

---

## File Format

Each chip is stored as a compressed NumPy `.npz` file:

```python
import numpy as np

chip = np.load('366123456_20240915T001234.npz', allow_pickle=True)

chip['chip_vv']        # float32 (512, 512) — VV polarisation amplitude
chip['chip_vh']        # float32 (512, 512) — VH polarisation amplitude
chip['mmsi']           # int — vessel identity (AIS MMSI)
chip['vessel_name']    # str
chip['vessel_type']    # int — raw AIS type code
chip['length_m']       # float — AIS-reported ship length (m)
chip['lat']            # float — AIS position latitude
chip['lon']            # float — AIS position longitude
chip['sog']            # float — speed over ground (knots)
chip['heading']        # float — AIS heading (degrees)
chip['scene_id']       # str — Sentinel-1 scene identifier
chip['acq_time']       # str — UTC acquisition time (ISO 8601)
chip['dt_sec']         # float — time delta between AIS fix and SAR acquisition (s)
chip['clean']          # bool — True if no other large vessel (≥150 m) in chip
chip['n_other_ships']  # int — count of other AIS vessels in chip
chip['nearest_large_px'] # float — pixel distance to nearest large vessel (-1 if none)
```

Files are organised into subdirectories by Sentinel-1 scene ID:
```
S1A_IW_GRDH_1SDV_20240915T001221.../
    366123456_20240915T001234.npz
    366789012_20240915T001234.npz
    ...
```

### Loading resolved labels

```python
import json

with open('resolved_labels.json') as f:
    labels = json.load(f)

# labels is a dict keyed by MMSI string
label = labels['366123456']
# {'label': 'Tanker', 'confidence': 'HIGH', 'sources': ['ais', 'length', 'name'], ...}
```

---

## Intended Uses

- SAR ship type classification benchmarking
- Ship re-identification across multiple SAR passes (MMSI as ground-truth identity)
- AIS label noise analysis and correction
- Backbone pretraining / fine-tuning for maritime SAR

---

## Limitations

- Gulf of Mexico only — open water, low land clutter; port and coastal scenes not included
- Fishing class is small (14 HIGH-confidence vessels, 71 chips)
- GRD resolution (10 m) limits fine-grained hull detail
- Chips are centred on AIS position, not SAR-detected centroid; small positional offset possible

---

## License

CC BY 4.0. Sentinel-1 imagery © ESA/Copernicus, used under free and open data policy.
NOAA AIS data is public domain. GFW data used under GFW API terms.

---

## Citation

```bibtex
@article{shaya2026sarship,
  title   = {Zero-Curation {SAR} Ship Classification and Re-Identification
             via Multi-Signal {AIS} Label Resolution and {SAR}-Native Backbone},
  author  = {Shaya, Mousa},
  journal = {{IEEE} J. Sel. Topics Appl. Earth Observ. Remote Sens.},
  year    = {2026},
  note    = {Under review}
}
```
