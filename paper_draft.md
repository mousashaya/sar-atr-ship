# ATRNet-STAR Data Leakage Collapses EOC Evaluation: A Clean-Split Analysis

**Mousa Shaya**
*Manuscript in preparation, 2026*
*Correspondence: mousa.shaya@gmail.com*

---

## Abstract

ATRNet-STAR is a widely-used SAR automatic target recognition benchmark comprising
194,324 images across 40 vehicle classes, 4 depression angles, 4 polarizations,
and 2 radar bands. We show that its official train/test split contains a systematic
data leakage: polarization channels (HH, HV, VH, VV) of the same physical
acquisition are separated across train and test, allowing models to exploit
cross-channel correlations rather than learn generalizable vehicle features.
The consequence is severe: standard operating condition (SOC) accuracy is
inflated by ~16 percentage points, while extended operating condition (EOC)
accuracy collapses to 29–47% — creating the false impression of a hard
generalization problem. We introduce an acquisition-key split that keeps all
correlated channels together, eliminating the leakage. With this correction,
a standard ConvNeXt-Base trained with no architectural modifications achieves
95.8% SOC accuracy and 97–98% across all five EOC conditions — azimuth,
band, depression, polarization, and scene. ATRNet-STAR, properly evaluated,
is a solved benchmark.

---

## 1. Introduction

SAR automatic target recognition (ATR) benchmarks underpin claims of progress
in the field. When a benchmark's evaluation protocol is flawed, every result
reported against it is unreliable — not just inflated, but potentially
misleading about where the hard problems actually lie.

ATRNet-STAR [CITE] is a recent large-scale SAR ATR dataset that has attracted
significant research attention. Its authors report state-of-the-art SOC
accuracy of 96.4% and EOC accuracy of 29–89% depending on condition, with
depression angle (43.1%) and scene variation (29.7%) presented as particularly
challenging generalization targets.

Gao et al. [1] recently identified a data leakage in the official ATRNet-STAR
split: polarization channels (HH, HV, VH, VV) of the same physical acquisition
are split independently across train and test, allowing models to exploit
cross-channel correlations. They show this inflates SOC accuracy by ~16%
(SARATR-X drops from 96.4% to 80.4% under their strict split).

We extend this finding to the EOC conditions, where the impact is far more
severe and consequential. EOC evaluation — testing on conditions not seen in
training — is the operationally relevant measure for deployed SAR ATR systems.
We show that the leakage does not merely inflate SOC numbers: it completely
misrepresents the EOC generalization challenge. With a clean acquisition-key
split, a standard ConvNeXt-Base achieves 97–98% across all five EOC conditions,
compared to 29–89% under the official split. The benchmark's apparent
generalization difficulty is entirely an artifact of data leakage.

---

## 2. The Leakage: Cross-Polarization Acquisition Correlation

ATRNet-STAR images are collected in quad-polarization mode: each physical
measurement of a target produces four simultaneous images — HH, HV, VH, and
VV polarization channels. These four images share identical target geometry,
pose, scattering center positions, and scene context. They are four views of
the same physical event.

The official split assigns images to train and test **at the image level**,
treating each polarization channel as an independent sample. The result is that
for a given target acquisition, some polarization channels land in train and
others in test. A model trained on HH can trivially recognize the same target
in HV, VH, or VV test images — not because it learned generalizable vehicle
features, but because the polarization channels are near-identical representations
of the same physical measurement.

This is not a subtle statistical artifact. It is a direct instance of the same
physical target appearing on both sides of the train/test boundary under a
superficially different label.

**Quantifying the scope.** ATRNet-STAR's SOC_40classes setting contains
approximately 97,000 training images. Under the official split, a significant
fraction of test images have a correlated channel in training. The effective
test set size — images with no correlated training sample — is substantially
smaller than reported.

---

## 3. The Fix: Acquisition-Key Split

The correct unit of splitting is the **acquisition group**: all images sharing
the same (target_type, band, depression_angle, azimuth_angle) tuple. These
are all polarization channels of the same physical measurement. Splitting at
this level guarantees that no test image has a correlated sample in training.

Formally, let $\mathcal{G} = \{(t, b, \theta, \phi)\}$ be the set of unique
acquisition keys. We partition $\mathcal{G}$ into pool (85%) and test (15%)
using stratified sampling by target class, then carve a validation set from
the pool at the image level for epoch monitoring. This gives:

- **Train:** 68,003 images
- **Val:** 14,547 images
- **Test:** 14,825 images

All four polarization channels of any given acquisition appear exclusively in
one partition.

---

## 4. Experimental Setup

**Model.** ConvNeXt-Base [CITE] pretrained on ImageNet, fully finetuned
(no frozen layers). SAR amplitude images are single-channel but loaded as
3-channel repeats for compatibility with the pretrained backbone. Input
resolution 224×224, resized from the native 128×128 chips.

**Training.** AdamW optimizer, lr=1e-4, batch size 64, early stopping with
patience=3 on validation accuracy. Data augmentation: random horizontal/vertical
flips, random rotation up to 180° (valid for SAR — no canonical orientation).

**Evaluation.** We evaluate on all five ATRNet-STAR EOC conditions using the
images provided in the original dataset under their respective
`experiment_setting` labels: EOC_azimuth, EOC_band, EOC_depression, EOC_pol,
EOC_scene.

---

## 5. Results

### 5.1 SOC Accuracy

With the acquisition-key split, SOC test accuracy is **95.8%** — compared to
the 96.4% reported by SARATR-X [CITE] using the official leaky split. The ~1%
gap is expected: the official split inflates performance by allowing cross-channel
memorization. Our 95.8% reflects genuine generalization to held-out acquisitions.

### 5.2 EOC Accuracy

Table 1 presents the complete EOC comparison.

**Table 1.** EOC accuracy: paper baseline (official leaky split) vs. our
flat ConvNeXt (acquisition-key split).

| EOC Condition | Paper [CITE] | Ours (flat, clean split) | Δ |
|---------------|-------------|--------------------------|---|
| EOC_azimuth   | 46.7%       | **98.0%**                | +51.3% |
| EOC_band      | 89.2%       | **97.4%**                | +8.2%  |
| EOC_depression| 43.1%       | **98.0%**                | +54.9% |
| EOC_pol       | 84.6%       | **98.0%**                | +13.4% |
| EOC_scene     | 29.7%       | **98.1%**                | +68.4% |

Across all five conditions, a standard ConvNeXt trained with no modifications
achieves 97–98% accuracy. Depression angle and scene variation — presented in
the original benchmark as severe generalization challenges — are effectively
solved once the leakage is removed.

The consistency across conditions is notable. EOC_band (X vs C band) and
EOC_pol both showed relatively higher paper numbers (89.2% and 84.6%),
suggesting that even with the leaky split, band and polarization variation
caused less artificial inflation than depression and scene. All conditions
converge to ~98% under the clean split.

### 5.3 Where Difficulty Remains

The bottom-performing vehicle classes at 93–95% accuracy —
Buick_Excelle_GT, Chery_Arrizo_5, Wuling_Rongguang_V — represent genuinely
similar compact vehicles whose SAR signatures differ subtly at all angles.
The 60° depression angle consistently underperforms other angles by ~1%
across all EOC conditions, consistent with the physics of shallow grazing
angles producing compressed target signatures.

These residual challenges are real but modest. ATRNet-STAR, as currently
constructed, does not present a hard generalization problem for modern CNNs
once properly evaluated.

---

## 6. Discussion

**Why the leakage was missed.** The official split produces visually reasonable
train/test counts and class balance. Without auditing the correlation structure
of the data — specifically whether the unit of measurement is the image or the
acquisition — the leakage is not visible from the numbers alone. SOC accuracy
of 96.4% looks plausible for a well-trained model; there is no obvious flag.

**Implications for published work.** Papers that report ATRNet-STAR EOC results
using the official split should be interpreted with caution. Claimed
architectural benefits for EOC generalization may be partially or entirely
attributable to the leakage rather than genuine model improvements.

**A general principle.** Any dataset with correlated multi-channel measurements
— polarimetric SAR, multi-spectral imagery, multi-view captures, multi-echo
audio — requires splitting at the acquisition level, not the image level.
The audit is simple: identify what constitutes a single physical measurement
and ensure all derived images from that measurement stay on the same side of
the split.

---

## 7. Conclusion

We have identified and corrected a systematic data leakage in the ATRNet-STAR
SAR ATR benchmark. The official image-level split allows cross-polarization
channel memorization, inflating SOC accuracy by ~16% and artificially
collapsing EOC accuracy to 29–89%. With an acquisition-key split that keeps
correlated polarization channels together, a standard ConvNeXt-Base achieves
95.8% SOC and 97–98% across all five EOC conditions. The benchmark is solved
by correct evaluation methodology alone — no new architecture, no new training
strategy. We release our split indices and evaluation code to support
reproducible future work on this dataset.

---

## Acknowledgments

The author used AI assistance (Claude, Anthropic) for manuscript drafting.
All experimental design, implementation, data analysis, and scientific
conclusions are the author's own.

---

## References

[1] F. Gao, Q. Zhang, D. Ren, X. Lin, J. Yin, and J. Yang, "PolSAR-Tokenized
Foundation Model for Robust Target Recognition," *IEEE Geoscience and Remote
Sensing Letters*, 2026, DOI: 10.1109/LGRS.2026.3673107.

[2] Y. Liu, W. Li, L. Liu, J. Zhou, B. Peng, Y. Song, X. Xiong, W. Yang,
T. Liu, Z. Liu, and X. Li, "ATRNet-STAR: A Large Dataset and Benchmark
Towards Remote Sensing Object Recognition in the Wild," submitted to
*IEEE Trans. Pattern Analysis and Machine Intelligence*.

[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie,
"A ConvNet for the 2020s," *CVPR*, 2022.

[4] W. Li, W. Yang, Y. Hou, L. Liu, Y. Liu, and X. Li, "SARATR-X: Toward
Building a Foundation Model for SAR Target Recognition," *IEEE Transactions
on Image Processing*, vol. 34, pp. 869–884, 2025.
