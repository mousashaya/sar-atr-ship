# SAR ATR on ATRNet-STAR: Findings, Mistakes, and Lessons

**Mousa Shaya**  
*April 2026*

---

## Background

I spent several weeks investigating ATRNet-STAR, a large-scale SAR automatic target recognition (ATR) benchmark comprising 194,324 images across 40 vehicle classes, 4 depression angles, 4 polarizations, and 2 radar bands. My goal was to understand whether the reported results held up under scrutiny, and whether a hierarchical cascade classifier could improve on the flat baseline.

This write-up documents what I found — including two significant mistakes I made along the way. I think honest accounts of failed or redirected research efforts are underrepresented in the literature, and I hope this saves someone else time.

---

## The Leakage Problem

### What ATRNet-STAR measures

ATRNet-STAR collects images in quad-polarization mode: each physical measurement of a vehicle produces four simultaneous images — HH, HV, VH, and VV polarization channels. These four images share identical target geometry, pose, and scene context. They are four views of the same physical event.

The official train/test split assigns images **at the image level**, treating each polarization channel as an independent sample. The result: for a given vehicle acquisition, some polarization channels land in train and others in test. A model trained on HH can recognize the same target in HV at test time — not because it learned generalizable vehicle features, but because the polarization channels are near-identical representations of the same physical measurement.

This is data leakage. Gao et al. (IEEE GRSL 2026) independently identified this problem and showed it inflates SOC accuracy by approximately 16 percentage points.

### The fix

The correct unit of splitting is the **acquisition group**: all images sharing the same (target_type, band, depression_angle, azimuth_angle) tuple. Splitting at this level guarantees no test image has a correlated sample in training. I implemented this as an acquisition-key split and used it for all experiments reported here.

---

## Experimental Setup

**Model:** ConvNeXt-Small pretrained on ImageNet, fully fine-tuned. SAR amplitude images loaded as 3-channel repeats. Input resolution 224×224, resized from native 128×128.

**Training:** AdamW, lr=1e-4, batch size 64, early stopping with patience=3 on validation accuracy. No augmentation.

**Evaluation:** ATRNet-STAR provides explicit train/test splits for each EOC condition. I used these splits directly, with a 15% validation set carved from the training partition for early stopping.

---

## Mistake #1: Evaluating the Wrong Thing

My first EOC results showed 97–98% accuracy across all five EOC conditions. I initially interpreted this as confirming that the leakage — once corrected — made the benchmark trivially solvable.

This was wrong.

I had trained a model on **all conditions** (all angles, all scenes, all polarizations) and then evaluated it on EOC test images. Of course it got 98% on EOC_depression test images at 30° — it had trained on 30° images. That is not an EOC evaluation. It is a SOC model evaluated on a subset of its own training distribution.

The correct EOC evaluation: **train only on the restricted data** specified by the benchmark (e.g., 15° only for EOC_depression), then test on the held-out conditions (30°/45°/60°). I caught this error before publishing anything, but it cost significant time and nearly produced a misleading paper.

**Lesson:** Before reporting any accuracy number, verify exactly what is in the train set and what is in the test set. When results look surprisingly good, that is a signal to check harder, not to celebrate.

---

## EOC Results: What the Numbers Actually Show

With the correct evaluation protocol, using ConvNeXt-Small:

| EOC Condition | Our result | Paper (ConvNeXt) | Difference |
|---|---|---|---|
| EOC_depression (15° → 30/45/60°) | 40.0% | 37.2% | +2.8% |
| EOC_pol (HH → HV/VH/VV) | 79.8% | 70.5% | +9.3% |
| EOC_scene (simple → complex) | 26.2% | 18.0% | −7.8% |

**EOC_depression:** Our 40% essentially matches the paper's 37.2%. This matters: the depression condition uses a split defined purely by angle (all 15° in train, all 30/45/60° in test), so there is no cross-polarization leakage possible within this structure. The paper's depression number was real. The 15°→30/45/60° generalization challenge is genuine and not an artifact.

**EOC_pol:** We beat the paper by ~9 points. Polarization change does not affect the geometric projection of the target, so angle-invariant features trained at all depression angles transfer well. The leakage correction here likely removes some artificial inflation the paper benefited from.

**EOC_scene:** We are worse than the paper by ~8 points. The cause is not fully understood. It may be model size (we used Small, paper may have used Base), or differences in how the scene split interacts with the training data. This result should be investigated before drawing strong conclusions.

**Overall:** The leakage story is more nuanced than a uniform inflation across all conditions. It depends on the structure of each EOC split and whether cross-polarization correlation can actually help in that setting.

---

## Mistake #2: Overconfident Paper Draft

Before catching Mistake #1, I drafted a full paper claiming 97–98% EOC accuracy and framing ATRNet-STAR as a "solved benchmark." The abstract and conclusions were strongly worded. I emailed a draft to myself for review.

Had that draft gone to co-authors or reviewers, it would have been embarrassing at best and damaging at worst.

**Lesson:** Strong results demand stronger verification, not faster publication. Sleep on surprising findings before writing them up.

---

## The Cascade Experiment

With EOC_depression results in hand, I investigated whether a hierarchical cascade classifier could improve over the flat baseline. The hypothesis: routing by main class (Bus/Car/Truck/Special) first, then classifying vehicle type within each class, might improve generalization by specializing each sub-classifier.

**Results for EOC_depression:**

| Method | Test accuracy |
|---|---|
| Flat ConvNeXt-Small | 40.0% |
| Cascade (predicted routing) | 39.4% |
| Cascade (oracle routing) | 53.8% |
| L1 routing accuracy | 64.1% |

The cascade did not help. The L1 routing model — trained on 15° images — achieved 97% validation accuracy on 15° images but only 64.1% on the 30/45/60° test set. Depression angle changes SAR signatures enough that even coarse 4-way main-class classification fails badly at unseen angles.

With 36% of images misrouted before L2 even runs, the cascade accumulates errors rather than reducing them. The oracle result (53.8%) shows the cascade *could* help if routing were reliable — but reliable routing requires angle-invariant features that single-angle training cannot provide.

**Per-angle breakdown (cascade predicted routing):**

| Depression angle | Accuracy |
|---|---|
| 30° | 58.3% |
| 45° | 40.6% |
| 60° | 19.3% |

The model falls apart at 60° — the most extreme departure from the 15° training geometry. This is consistent with SAR physics: steeper depression angles produce dramatically different scattering patterns.

**Lesson:** Hierarchy helps when the routing signal is stable. When the routing signal itself fails under domain shift, cascade amplifies the error.

---

## What Would Actually Help EOC_depression

I explored this question honestly. The fundamental problem — learning angle-invariant features from a single depression angle — is not easily solved by standard ML tricks. Some directions exist:

- **Domain adaptation** (DANN and variants) requires unlabeled target-domain images at training time, which may violate benchmark rules
- **Physics-based feature extraction** (angle-invariant scattering features) requires specialized SAR signal processing outside standard deep learning pipelines
- **Generative translation** (train a model to synthesize 30/45/60° images from 15°) is promising but adds significant complexity and assumptions

No straightforward architectural change solves it. The EOC_depression challenge is genuinely hard.

---

## Key Takeaways

1. **Always split at the acquisition level**, not the image level, for multi-channel datasets. The unit of splitting is the physical measurement, not the derived image.

2. **Verify evaluation correctness before trusting results.** When numbers look surprisingly good, that is a reason to check harder.

3. **Cascade classification requires stable routing.** If the routing signal fails under domain shift, the cascade makes things worse, not better.

4. **EOC_depression is a real unsolved problem.** The benchmark's reported difficulty for this condition is not a leakage artifact — it reflects genuine challenges in SAR angle generalization.

5. **The leakage effect is condition-specific.** Not all EOC conditions are equally affected. Depression is unaffected; polarization shows clear improvement with clean split; scene is inconclusive.

---

## What's Next

I'm moving to ShipRS, an optical satellite imagery dataset for warship recognition. The challenges there are different: severe class imbalance (some warship classes have fewer than 30 examples), fine-grained recognition within destroyer and frigate classes, and robustness to environmental conditions (fog, glint, dark). The acquisition-key lesson applies there too — same ship instance must not appear on both sides of the split.

---

## Code and Data

All training scripts, evaluation code, and split logic are available at [github link]. The acquisition-key split implementation is in `scripts/dataset.py`.

---

*I used AI assistance (Claude, Anthropic) for coding and manuscript drafting. All experimental design, analysis, and conclusions are my own.*
