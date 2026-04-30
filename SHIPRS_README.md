# ShipRSImageNet — Dataset Reference

*Original README from [github.com/zzndream/ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet), enhanced with findings from our audit.*

---

## Dataset Overview

**ShipRSImageNet** is a large-scale fine-grained dataset for ship detection in high-resolution optical remote sensing images.

- **3,435 images** from various sensors, satellite platforms, locations, and seasons
- Each image is approximately **930×930 pixels**
- **17,573 ship instances** across **50 object categories**
- Hierarchical classification across **4 levels**

### Image Sources

| Source | Images | Notes |
|---|---|---|
| xView (WorldView-3, 0.3m) | 532 | Sliced from large scenes with 150px overlap |
| HRSC2016 | 1,057 | Corrected mislabels |
| FGSD | 1,846 | Corrected mislabels |
| Airbus Ship Detection | 21 | |
| Chinese satellites (GaoFen-2, JiLin-1) | 17 | |

**License:** Academic use only. Commercial use prohibited.  
**xView note:** xView maintainers issued a takedown request — 532 images were redistributed without authorization. Verify compliance before use in regulated/defense contexts.

---

## Class Hierarchy (4 Levels)

**Level 0:** Ship vs. non-ship (binary)  
**Level 1:** Warship / Merchant / Other Ship / Dock  
**Level 2:** Destroyer, Frigate, Aircraft Carrier, Landing, Auxiliary, Submarine, Patrol, Commander / Cargo, Container, Tanker, Ferry, etc.  
**Level 3:** 50 specific types

### Level 3 Class List
Other Ship, Other Warship, Submarine, Other Aircraft Carrier, Enterprise, Nimitz, Midway, Ticonderoga, Other Destroyer, Atago DD, Arleigh Burke DD, Hatsuyuki DD, Hyuga DD, Asagiri DD, Other Frigate, Perry FF, Patrol, Other Landing, YuTing LL, YuDeng LL, YuDao LL, YuZhao LL, Austin LL, Osumi LL, Wasp LL, LSD 41 LL, LHA LL, Commander, Other Auxiliary Ship, Medical Ship, Test Ship, Training Ship, AOE, Masyuu AS, Sanantonio AS, EPF, Other Merchant, Container Ship, RoRo, Cargo, Barge, Tugboat, Ferry, Yacht, Sailboat, Fishing Vessel, Oil Tanker, Hovercraft, Motorboat, Dock

**Abbreviations:** DD=Destroyer, FF=Frigate, LL=Landing, AS=Auxiliary Ship, LSD=Landing Ship Dock, LHA=Landing Helicopter Assault, AOE=Fast Combat Support Ship, EPF=Expeditionary Fast Transport, RoRo=Roll-on/Roll-off

---

## Official Splits

| Split | Images | Instances |
|---|---|---|
| Train | 2,198 | 10,860 |
| Val | 550 | 3,103 |
| Test (V1.1 only) | 687 | ~3,600 |

**We have V1 only** — train and val splits ingested, test set unavailable (hosted on Baidu Pan, inaccessible outside China).

---

## Our Audit Findings

### Split Strategy
The official split is **image-level** — all instances within an image stay on the same side. This is correct: one satellite image may contain multiple ships from the same scene, and splitting at the instance level would leak scene context.

Our `instance_id` column in the DB maps to annotation IDs (one row per instance), not to unique physical ships. The apparent 91.5% overlap of instance_ids across splits is an artifact of this — it is not data leakage.

### Planned Split (since V1.1 test set is unavailable)
Carve our own clean test set from the V1 train split:
- Split at **image level** (not instance level)
- Stratified by level_3 class
- 70% train / 15% val / 15% test (seed=42)
- All instances in an image stay together

### Sparse Level 3 Classes (< 50 instances)
These classes are too small for reliable standalone fine-grained classification:

| Class | Count |
|---|---|
| Midway | 18 |
| Wasp LL | 20 |
| Medical Ship | 27 |
| Osumi LL | 34 |
| Other Aircraft Carrier | 34 |
| Masyuu AS | 36 |
| YuZhao LL | 37 |
| Training Ship | 42 |
| YuDao LL | 45 |
| AOE | 48 |

Strategy: merge into parent level_2 "Other" categories, or treat as few-shot problem.

### Known Annotation Issues
- **OBB rotation mislabeling** (GitHub Issue #7): Some ships with non-zero rotation annotated with 0° rotation. Use horizontal bounding boxes (our approach) to avoid this.
- **Incomplete labeling** (GitHub Issue #4): Some ships in images appear unlabeled. Models may be penalized for detecting them.
- **Test annotations** (GitHub Issue #2): COCO format test JSON reported as empty in some versions.

### Environmental Conditions (per our DB)

| Condition | Count | % |
|---|---|---|
| Normal | 12,162 | 87.1% |
| Glint | 1,083 | 7.8% |
| Fog | 894 | 6.4% |
| Dark | 109 | 0.8% |

Dark is severely underrepresented. Worth tracking per-condition accuracy separately to assess operational robustness.

---

## Our Research Plan

**Goal:** Test whether physical-property routing (ship type hierarchy) helps fine-grained classification compared to a flat baseline — in a domain (optical satellite imagery) different from ATRNet-STAR (SAR).

**Key differences from ATRNet-STAR:**
- Optical imagery (not SAR) — angle generalization less of a concern
- True hierarchy with meaningful semantic levels
- Real class imbalance problem (sparse warship subtypes)
- No EOC-style generalization challenge — standard classification

**Cascade design:**
- L1: Warship / Merchant / Other Ship / Dock (4-way)
- L2: Ship type within each category (Destroyer, Frigate, etc.)
- L3: Specific vessel (Arleigh Burke, Perry FF, etc.) — only where data is sufficient

**Baseline:** Flat 50-class ConvNeXt-Small on clean image-level split.

---

## Citation

```
Z. Zhang, L. Zhang, Y. Wang, P. Feng and R. He,
"ShipRSImageNet: A Large-Scale Fine-Grained Dataset for Ship Detection
in High-Resolution Optical Remote Sensing Images,"
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing,
vol. 14, pp. 8458-8472, 2021, doi: 10.1109/JSTARS.2021.3104230.
```
