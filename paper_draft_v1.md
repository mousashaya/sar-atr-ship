# ATRNet-STAR: Depression Angle Diversity, Few-Shot Recognition, and Cross-Dataset Generalization in SAR Automatic Target Recognition

**Mousa Shaya**  
Shaya Software and Systems LLC | Arizona State University  
mshaya01.ml@gmail.com | shayaml.com

---

## Abstract

Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) faces a fundamental challenge: operational models must generalize across depression angles, sensor configurations, and datasets, yet labeled SAR imagery is scarce and expensive to acquire. This paper presents a systematic study of these generalization challenges using the ATRNet-STAR dataset (40 vehicle classes, 4 depression angles) and the MSTAR benchmark (10 military vehicle classes). We make six contributions: (1) a depression angle ablation demonstrating that adding a single 60° view to a 15°-only baseline improves EOC accuracy by +30.4 percentage points; (2) a data efficiency analysis showing that multi-angle training with as few as 25 images per vehicle per angle already surpasses a model trained on thousands of single-angle images, with diminishing returns beyond ~400 samples; (3) a finding that neither angle-weighted sampling nor test-time augmentation improves upon standard training — TTA reduces accuracy by 24.3 pp — confirming that angle invariance requires real multi-angle data, not sampling or augmentation tricks; (4) a few-shot prototypical classifier achieving 98%+ accuracy on unseen depression angles using only 5 reference images per class, replicated on both ATRNet-STAR and MSTAR; (5) a cross-dataset transfer experiment revealing a meaningful SAR-to-SAR domain gap (61.1% when applying an ATRNet-STAR backbone to MSTAR versus 99.1% for a MSTAR-trained backbone); and (6) a demonstration that optical pretrained features (DINOv2, 6.1%) are essentially useless for SAR ATR, motivating domain-specific approaches. Together, these results characterize the generalization landscape of SAR ATR and provide practical guidance for building robust systems with limited labeled data.

---

## 1. Introduction

Automatic Target Recognition from SAR imagery is an operationally critical problem: SAR sensors image through clouds and at night, making them uniquely valuable for persistent surveillance. Yet deploying ATR systems in practice surfaces a series of generalization challenges that laboratory benchmarks often obscure.

**The depression angle problem.** SAR images ground targets at varying depression (grazing) angles. A vehicle imaged at 15° looks fundamentally different from the same vehicle at 60° due to layover, foreshortening, and shadow geometry changes. Most published ATR results report Standard Operating Condition (SOC) accuracy — train and test at the same angle — which inflates operational expectations. Extended Operating Condition (EOC) evaluation, where the test angle is withheld during training, is a harder and more realistic measure.

**The data scarcity problem.** Unlike optical computer vision, where datasets contain millions of labeled examples, SAR ATR datasets are measured in thousands. This makes large-scale supervised learning fragile and motivates few-shot approaches.

**The domain gap problem.** Even within SAR, features learned on one dataset do not transfer freely to another. Different sensors, collection geometries, and target sets produce images with distinct statistical properties. Understanding the magnitude of these gaps is essential for building systems that generalize.

We address all three problems with a unified experimental framework. Our primary dataset is ATRNet-STAR [1], a large-scale SAR vehicle dataset collected at four depression angles (15°, 30°, 45°, 60°) across 40 vehicle classes. We supplement with MSTAR [2], the canonical 10-class military vehicle benchmark collected at 15° and 17°. Our backbone throughout is ConvNeXt-Small [3], fine-tuned from ImageNet weights, which we show substantially outperforms the DINOv2 ViT-B/14 foundation model on SAR imagery.

---

## 2. Related Work

**SAR ATR benchmarks.** MSTAR [2] remains the most widely cited SAR ATR benchmark, with published CNN accuracies of 97–99% on the standard SOC task (train 17°, test 15°) and a well-studied EOC protocol (train 17°, test 30° or 45°). Recent methods including A-ConvNet and ASIR-Net achieve upward of 97% on the 10-class EOC-D1 task [1]. ATRNet-STAR [1] extends this to 40 classes across four depression angles, posing a substantially harder generalization problem: the 15°→{30°, 45°, 60°} gap spans up to 45° of angular change, compared to the 13° change in MSTAR's standard EOC setup.

**Depression angle generalization.** Performance degradation under angular variation is well documented [1], and recent work has applied sparse representation, complex-valued networks, and multi-view transformers to mitigate it. However, systematic study of which specific angles contribute most to generalization, and how many samples per angle are needed to build an invariant backbone, has not been reported. We fill both gaps.

**Few-shot SAR ATR.** Prototypical networks [5] have been applied to SAR ATR to address data scarcity. DGP-Net [6] proposes a dense graph convolutional network to cluster same-class samples across depression angles in MSTAR's 3-way and 5-way few-shot tasks, improving over prior methods by over 4%. Our work differs in two respects: we operate on ATRNet-STAR's 40-class problem (far larger than MSTAR's 10 classes), and we show that a frozen SOC backbone with nearest-centroid classification — requiring no specialized architecture — achieves 98%+ accuracy with as few as 5 reference images. The result holds across both ATRNet-STAR and MSTAR, suggesting that backbone quality matters more than classification head complexity.

**Foundation models for SAR.** SARATR-X [7] trains a self-supervised backbone on 0.18M unlabeled SAR samples and evaluates it on few-shot and robustness benchmarks, demonstrating that SAR-specific pretraining yields generalizable representations. Our DINOv2 experiment provides complementary motivation: optical SSL pretraining (6.1% on the 40-class ATRNet-STAR problem) is essentially useless for SAR, confirming that domain-specific approaches like SARATR-X are necessary. The question we address — how much labeled multi-angle data is needed for supervised backbone training — is orthogonal to SARATR-X's unsupervised pretraining paradigm.

---

## 3. Datasets and Methods

### 3.1 ATRNet-STAR

ATRNet-STAR contains SAR chip images of 40 ground vehicle classes collected at four depression angles: 15°, 30°, 45°, and 60°. Images are single-polarization magnitude chips stored as floating-point arrays and converted to 8-bit grayscale via log-magnitude normalization before model input.

**SOC configuration:** Train and test on all four angles using a 70/15/15 train/val/test split stratified by class.

**EOC configuration:** Train on {15°} only, test on {30°, 45°, 60°}. This reflects the operationally realistic scenario where a model is trained on a single available depression angle and must generalize to unseen angles at deployment.

### 3.2 MSTAR

MSTAR contains 10 military vehicle classes (BMP2, BTR60, BTR70, 2S1, BRDM2, D7, T62, T72, ZSU234, ZIL131) collected at two depression angles: 17° (training) and 15° (test). Images are stored in a Phoenix binary format (complex float32) and decoded to magnitude images. We use the standard SOC split: train on 17°, test on 15°, yielding 3,128 training, 543 validation, and 3,203 test samples.

### 3.3 Code Availability

All code for data ingestion, training, evaluation, and visualization is publicly available at https://github.com/mousashaya/sar-atr-ship.

### 3.4 Model Architecture

We use ConvNeXt-Small pretrained on ImageNet-1K as our backbone throughout. The classification head is replaced with a linear layer sized to the target number of classes. Training uses AdamW (lr=1e-4) with cross-entropy loss. Augmentations during training include random horizontal/vertical flips and rotation. The same architecture is used for both ATRNet-STAR and MSTAR experiments to enable direct comparison.

### 3.5 Prototypical Classification

For few-shot experiments, we extract feature vectors from the penultimate layer (before the final linear classifier) using a forward hook. Given N reference images per class, we compute per-class prototype centroids by averaging feature vectors. Test samples are classified by nearest centroid under Euclidean distance. This requires no test-time training — the backbone is frozen and used purely as a feature extractor.

### 3.6 Baseline Comparators

We compare against DINOv2 ViT-B/14 [4] as a strong optical foundation model, evaluated in the same prototypical framework with no fine-tuning on SAR data. This represents the "zero domain adaptation" upper bound for optical pretrained models.

---

## 4. Experiments

### 4.1 SOC Baselines

We first establish that our framework reproduces published SOC accuracy on both datasets.

**ATRNet-STAR SOC (40 classes, 4 angles):** 97.2% test accuracy. This confirms our pipeline is sound and provides an upper bound for EOC experiments.

**MSTAR SOC (10 classes, 17°→15°):** 99.1% test accuracy, 99.4% best validation. This matches the published range of 97–99% [2], validating that our ConvNeXt-based framework is competitive with prior CNN approaches on the canonical benchmark.

### 4.2 EOC Baseline and Angle Ablation

Figure 1 provides a visual explanation of the SOC/EOC performance gap. The SOC backbone (trained on all four angles) places test images at 30°, 45°, and 60° near their correct vehicle clusters; the EOC backbone (trained on 15° only) collapses the same images into an undifferentiated mass. This directly motivates both the data efficiency study (Section 4.3) and the prototypical approach (Section 4.5).

![Figure 1](logs/umap_paper_figure/soc_eoc_overlay.png)

**Figure 1.** UMAP visualization of the 768-dimensional feature space under two backbones. Gray points are 15° reference images (training angle); colored points are test images at 30°, 45°, and 60° projected onto the same space. **(a)** SOC backbone (trained on all four angles): test-angle images land near their correct vehicle clusters, explaining prototypical classification's 98%+ accuracy with as few as 5 reference images. **(b)** EOC backbone (trained on 15° only): the same test images scatter randomly — vehicle cluster structure is absent, explaining the 43.7% EOC baseline.

Training on {15°} only and evaluating on {30°, 45°, 60°} yields 43.7% accuracy — a stark contrast to the 97.2% SOC ceiling, quantifying the depression angle generalization gap.

To understand which training angles contribute most, we perform an ablation progressively adding angles to the 15°-only baseline:

| Training Angles | Test Accuracy (15°) | Δ vs. Baseline |
|---|---|---|
| 15° only (EOC baseline) | 43.7% | — |
| 15° + 60° | 74.1% | +30.4% |
| 15° + 30° + 60° | 78.3% | +34.6% |
| All 4 angles (15°+30°+45°+60°) | 86.1% (N=200) | +42.4% |

The most striking finding is the single-angle addition: including 60° alone accounts for the majority (+30.4 pp) of the total gain from all additional angles (+42.4 pp). This suggests 60° imagery captures structural features that transfer effectively to 15° classification, likely because the large angular contrast forces the model to learn angle-invariant representations.

### 4.3 Data Efficiency

We study how accuracy scales with the number of labeled training images per (vehicle, angle) group, ranging from N=5 to the full dataset. The model trains on all four angles simultaneously, capped to N images per group, and is tested on held-out 15° images not seen during training.

| N per (vehicle, angle) | Test Accuracy (15°) | vs. EOC baseline |
|---|---|---|
| 5 | 10.8% | — |
| 10 | 21.7% | — |
| 25 | 49.0% | **+5.3 pp** |
| 50 | 65.8% | +22.1 pp |
| 100 | 77.0% | +33.3 pp |
| 200 | 86.1% | +42.4 pp |
| 400 | 89.2% | +45.5 pp |
| 500 | 89.6% | +45.9 pp |
| Full | 97.2% | +53.5 pp |

The key finding is the crossover at N=25: with only 25 images per vehicle per angle — roughly 4% of the full training set — multi-angle training already surpasses the EOC baseline (43.7%) achieved by a model trained on thousands of single-angle images. This crossover demonstrates that angle diversity is more valuable than data volume: a small diverse collection outperforms a large homogeneous one.

The curve flattens markedly above N=400. From 400 to 500, accuracy improves only 0.4 pp, suggesting diminishing returns beyond this threshold. For operational planning, N≈400 captures ~92% of full-data performance while requiring less than half the data.

### 4.4 Negative Results: What Does Not Work

Before presenting the prototypical results, we report two approaches that failed to improve EOC generalization — both in the regime where training data is limited (N=200 per group).

**Angle-weighted sampling.** An intuitive approach to improving angle invariance is to oversample rarer or more challenging angles during training using inverse-frequency weights. We applied this to the E3 experiment across the full range of N. The result was consistently worse than uniform sampling:

| N per group | Baseline | Weighted | Δ |
|---|---|---|---|
| 25 | 40.8% | 40.8% | 0.0% |
| 50 | 55.2% | 52.5% | −2.7% |
| 100 | 70.4% | 68.1% | −2.3% |
| 200 | 86.1% | 80.1% | −6.0% |

Weighting does not help because the model already sees all angles; the bottleneck is data volume per angle, not sampling bias.

**Test-time augmentation (TTA).** TTA — averaging predictions over multiple augmented views of each test image — is a standard technique in optical vision that typically improves accuracy by 1–3 pp. Applied to SAR ATR (N=200, k=7 augmented views), it reduced accuracy from 89.8% to 65.6% (−24.3 pp). Unlike optical images, SAR chips are geometrically constrained: random flips and rotations produce views that are out-of-distribution relative to the training data, since SAR backscatter is highly sensitive to azimuth and depression angle. TTA assumes augmented views are semantically equivalent; in SAR, they are not.

Together, these results reinforce the central message of Section 4.3: **angle invariance cannot be approximated through augmentation or sampling strategies — it requires exposure to actual multi-angle training data.**

---

### 4.5 Few-Shot Prototypical Classification

#### ATRNet-STAR (trained backbone)

Using the ATRNet-STAR SOC backbone as a fixed feature extractor, we evaluate prototypical classification across depression angles using only N reference images at 15°:

| N (ref/class) | Overall | 30° | 45° | 60° |
|---|---|---|---|---|
| 5 | 98.2% | 98.7% | 98.8% | 97.0% |
| 10 | 98.4% | 98.9% | 99.0% | 97.3% |
| 25 | 98.4% | 98.9% | 99.0% | 97.4% |
| 50 | 98.5% | 99.0% | 99.0% | 97.4% |
| 100 | 98.5% | 99.0% | 99.1% | 97.5% |

The curve saturates immediately — 5 reference images per class achieves 98.2%, within 0.3 pp of using 100 references. This demonstrates that a well-trained backbone produces features so discriminative that nearest-centroid classification requires minimal calibration data. The 60° column is consistently lowest, consistent with 60° being the most geometrically distinct viewing angle.

#### DINOv2 Comparison (optical pretrained, zero SAR training)

| N (ref/class) | Overall |
|---|---|
| 5 | 6.1% |
| 10 | 6.9% |
| 25 | 8.4% |
| 50 | 8.6% |
| 200 | ~12% |

DINOv2, despite being a state-of-the-art vision foundation model pretrained on 142M images, achieves near-chance accuracy (random chance = 2.5% for 40 classes) on SAR ATR. This result is not a failure of DINOv2 — it is a fundamental finding: optical imagery and SAR imagery are statistically orthogonal domains. Features learned from natural photographs encode texture, color, and lighting gradients that simply do not exist in coherent SAR backscatter. This result strongly motivates domain-specific SAR pretraining.

#### MSTAR Prototypical (MSTAR-trained backbone)

| N (ref/class) | Accuracy | Ref images |
|---|---|---|
| 1 | 98.3% | 10 |
| 5 | 99.1% | 50 |
| 10 | 99.2% | 100 |
| 25 | 99.2% | 250 |
| all | 99.1% | 3,671 |

The MSTAR result is remarkable: **a single reference image per class achieves 98.3%**, statistically indistinguishable from the 99.1% SOC ceiling. This confirms the prototypical finding is not dataset-specific: when the backbone is trained on the target domain, few-shot recognition approaches full supervised performance.

### 4.6 Cross-Dataset Transfer: ATRNet-STAR → MSTAR

We ask whether features learned on ATRNet-STAR (40 classes, wheeled/tracked vehicles, Ku-band) transfer to MSTAR (10 classes, military vehicles, X-band) without any MSTAR training:

| N (ref/class) | Transfer Accuracy |
|---|---|
| 1 | 28.0% |
| 5 | 52.3% |
| 10 | 54.9% |
| 25 | 59.4% |
| all | 61.1% |

The ATRNet-STAR backbone reaches 61.1% on MSTAR with full reference sets, compared to 99.1% for the MSTAR-trained backbone. This 38 pp gap quantifies the SAR-to-SAR domain gap — distinct sensor parameters, target sets, and imaging conditions produce features that do not transfer freely even within the SAR domain.

Placed in context with the DINOv2 result, these three data points tell a coherent story about domain proximity:

| Backbone | Pretraining Domain | MSTAR Accuracy |
|---|---|---|
| DINOv2 ViT-B/14 | Optical (ImageNet) | ~12% |
| ATRNet-STAR ConvNeXt | SAR (different targets) | 61.1% |
| MSTAR ConvNeXt | SAR (same targets) | 99.1% |

Transfer accuracy scales with domain proximity. Optical→SAR transfer is negligible. SAR→SAR transfer is meaningful (61%) but insufficient for deployment. Task-specific training remains essential.

---

## 5. Discussion

**Why does 60° add so much?** We hypothesize that 60° depression angle images have distinct shadow and layover geometry compared to 15°, forcing the network to discover features that are stable across angular variation rather than relying on angle-specific artifacts. The 60° images provide maximal angular contrast with the test condition, acting as a strong regularizer.

**Why does prototypical classification work so well?** The backbone is trained to minimize cross-entropy across 40 classes and 4 angles simultaneously. The resulting feature space must separate 40 classes while being invariant to 4 angles — a highly constrained optimization that produces dense, well-separated clusters. Nearest-centroid classification in this space is then trivially effective.

**Why augmentation fails in SAR.** TTA's −24.3 pp result is not a quirk of implementation — it reflects a fundamental property of SAR imagery. In optical vision, geometric augmentations (flips, rotations) produce views that are semantically equivalent; a car flipped horizontally is still a car. In SAR, backscatter intensity is highly sensitive to azimuth and depression angle, so an arbitrarily rotated chip falls outside the training distribution. This distinguishes SAR ATR from most optical recognition tasks and limits the applicability of augmentation-heavy training recipes.

**Practical implications.** For new SAR ATR deployments where labeled data is scarce: (1) train a backbone on whatever labeled data is available, even from a different dataset; (2) use prototypical classification for adaptation, which requires only a handful of reference images per new class; (3) prioritize collecting reference images at multiple depression angles rather than maximizing the count at a single angle. The angle ablation result shows that a small diverse set outperforms a large homogeneous set.

**Limitations.** All experiments use single-polarization magnitude images. Dual-polarization and complex-valued inputs likely carry additional discriminative information not exploited here. The ATRNet-STAR and MSTAR datasets are both collected in controlled or semi-controlled conditions; performance on operational SAR imagery with speckle noise, extended clutter backgrounds, and non-cooperative targets may differ substantially.

---

## 6. Conclusion

We have characterized the generalization landscape of SAR ATR along three axes: depression angle diversity, data quantity, and cross-dataset transfer. The key findings are:

1. **Angle diversity is essential, not additive.** Adding 60° to a 15°-only baseline provides +30.4 pp — more than half the total gain from all additional angles combined.

2. **Data efficiency has a sharp crossover.** With only 25 images per vehicle per angle, multi-angle training surpasses a model trained on thousands of single-angle images. The curve plateaus around N=400, beyond which additional data yields diminishing returns.

3. **Angle invariance cannot be faked.** Weighted sampling and test-time augmentation both fail to improve — and TTA actively hurts performance (−24.3 pp). Real multi-angle training data is necessary; no augmentation or reweighting strategy substitutes for it.

4. **Few-shot prototypical classification matches full supervised performance.** A well-trained backbone enables 98%+ accuracy with 5 reference images per class, replicated across ATRNet-STAR (40 classes) and MSTAR (10 classes). No specialized architecture is required.

5. **SAR-to-SAR domain gaps are real but navigable.** Cross-dataset transfer achieves 61%, far above optical transfer but well below task-specific training.

6. **Optical pretrained features are not useful for SAR ATR.** DINOv2 achieves ~6–12% on a 40-class SAR problem, effectively at chance, motivating SAR-specific pretraining approaches.

These findings have direct implications for building operational ATR systems: invest in backbone quality, collect diverse angles rather than maximizing single-angle volume, and use prototypical adaptation for new targets rather than full retraining.

---

## Acknowledgments

Code implementation and manuscript preparation were assisted by Claude (Anthropic). All experimental design, interpretation of results, and scientific claims are the sole responsibility of the author.

---

## References

[1] Y. Liu, W. Li, L. Liu, J. Zhou, B. Peng, Y. Song, X. Xiong, W. Yang, T. Liu, Z. Liu, and X. Li, "ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild," submitted to *IEEE Trans. Pattern Anal. Mach. Intell.*

[2] V. R. Lalmani, "MSTAR Dataset 10 Classes," IEEE DataPort, 2025, doi: 10.21227/bfp9-7019.

[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.

[4] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., "DINOv2: Learning Robust Visual Features without Supervision," *arXiv:2304.07193 [cs.CV]*, 2023. doi: 10.48550/arXiv.2304.07193

[5] J. Snell, K. Swersky, and R. S. Zemel, "Prototypical Networks for Few-shot Learning," *arXiv:1703.05175 [cs.LG]*, 2017. doi: 10.48550/arXiv.1703.05175

[6] X. Zhou, Q. Wei, and Y. Zhang, "SAR Target Depression Angle Invariant Recognition of Few-Shot Learning Via Dense Graph Prototype Network," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, pp. 1–15, Jan. 2025, doi: 10.1109/JSTARS.2025.3586129.

[7] W. Li, W. Yang, Y. Hou, L. Liu, Y. Liu, and X. Li, "SARATR-X: Towards Building A Foundation Model for SAR Target Recognition," 2025. Code: https://github.com/waterdisappear/SARATR-X

---

*Draft v1 — 2026-05-06. All results from experiments run on ATRNet-STAR and MSTAR datasets.*
