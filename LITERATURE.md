# Literature Review Notes

---

## 1. Combination of Global and Local Filters for Robust SAR ATR Under Various EOCs
**Authors:** Ding, Baiyuan; Wen, Gongjian  
**Source:** Elsevier — Information Sciences, 2019, Vol.476, p.48-63  
**Summary:** Proposes an ATR method combining global and local filters to improve robustness under various EOCs including depression angle and configuration variance.  
**Relevance:** Foundational EOC framing, but MSTAR-only — useful as a baseline reference.

---

## 2. Data Augmentation Based on Attributed Scattering Centers to Train Robust CNN for SAR ATR
**Authors:** Lv, Junya; Liu, Yue  
**Source:** IEEE Access, 2019, Vol.7, p.25459-25473  
**DOI:** https://doi.org/10.1109/access.2019.2900522  
**Summary:** Uses attributed scattering centers (ASC) to reconstruct and augment SAR training images, then trains a CNN robust to EOCs including depression angle variance, noise, and occlusion.  
**Relevance:** ASC-based augmentation is directly related to our scattering-aware approach; MSTAR results show 99.48% SOC accuracy.

---

## 3. A Lightweight Fully Convolutional Neural Network for SAR ATR
**Authors:** Yu, Jimin; Zhou, Guangyu; Zhou, Shangbo; Yin, Jiajun  
**Source:** MDPI Remote Sensing, 2021-08, Vol.13(15), Article 3029  
**Summary:** Proposes ASIR-Net — a lightweight FCN using channel-attention, channel-shuffle, and inverted-residual blocks — achieving strong SOC and EOC performance on MSTAR.  
**Relevance:** Lightweight architecture reference; EOC benchmarking methodology transferable to our setting.

---

## 4. Selecting Pseudo Supervision for Unsupervised Domain Adaptive SAR Target Classification
**Authors:** Zhao, Lingjun; He, Qishan; Ding, Ding; Zhang, Siqian; Kuang, Gangyao; Liu, Li  
**Source:** EURASIP Journal on Advances in Signal Processing, 2022, Vol.2022(1), Article 84  
**Summary:** Unsupervised domain adaptation via iterative pseudo-labelling with breaking-ties selection, targeting depression angle distribution shift in SAR classification.  
**Relevance:** Directly addresses depression angle domain shift — core to our EOC evaluation.

---

## 5. Scattering Attribute Embedded Network for Few-Shot SAR ATR
**Authors:** Qin, Jiang; Zou, Bin; Chen, Yifan; Li, Haolin; Zhang, Lamei  
**Source:** IEEE Transactions on Aerospace and Electronic Systems, 2024-08, Vol.60(4), p.4182-4197  
**Summary:** SANet embeds physical scattering attributes into a two-stream architecture for few-shot SAR ATR, showing robustness across depression angles and occlusions.  
**Relevance:** Few-shot + scattering physics + depression robustness — intersects several of our contributions.

---

## 6. Learning Hierarchical Visual Transformation for Domain Generalizable Visual Matching and Recognition
**Authors:** Yang, Xun; Chang, Tianyu; Zhang, Tianzhu; Wang, Shanshan; Hong, Richang; Wang, Meng  
**Source:** International Journal of Computer Vision, 2024-11, Vol.132(11), p.4823-4849  
**Summary:** HVT/EHVT transforms training samples hierarchically (global, local, pixel) to maximize domain diversity and learn domain-invariant features.  
**Relevance:** General domain generalization framework applicable to our cross-condition SAR setting.

---

## 7. No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems
**Authors:** Sohoni, Nimit S; Dunnmon, Jared A; Angus, Geoffrey; Gu, Albert; Ré, Christopher  
**Source:** NeurIPS 2020  
**Summary:** GEORGE measures and mitigates hidden stratification (variable performance across unlabeled subclasses) using clustering-based pseudo-labels in a distributionally robust objective.  
**Relevance:** Directly relevant to our hierarchical classification — coarse vehicle class labels masking fine-grained subclass performance gaps.

---

## 8. Zero-Shot Domain Adaptation for SAR Target Recognition Based on Cooperative Learning of Domain and Task Alignment
**Authors:** Chen, Guo; Zhang, Siqian; Zhou, Zheng; Zhao, Lingjun; Kuang, Gangyao  
**Source:** IEEE Transactions on Radar Systems, 2025, Vol.3, p.890-904  
**Summary:** ZSDA trains on simulated+real seen-class data using domain adversarial and task alignment losses, then generalizes to unseen real classes (91.68% top result).  
**Relevance:** Zero-shot sim-to-real transfer — relevant if we address novel class generalization.

---

## 9. VSFA: Visual and Scattering Topological Feature Fusion and Alignment for Unsupervised Domain Adaptation in SAR ATR
**Authors:** Zhang, Chen; Wang, Yinghua; Liu, Hongwei; Sun, Yuanshuang; Wang, Siyuan  
**Source:** IEEE Transactions on Geoscience and Remote Sensing, 2023, Vol.61, Article 5216920  
**Summary:** Combines visual reconstruction and graph-based scattering topology features with a two-stage class-wise domain alignment loss; evaluated on SAMPLE dataset (99.15% accuracy).  
**Relevance:** Scattering topology + domain alignment — directly related to our cross-condition approach.

---

## 10. HDSS-Net: Hierarchically Designed Network With Spherical Space Classifier for Ship Recognition in SAR
**Authors:** Shang, Yuanzhe et al.  
**Source:** IEEE Transactions on Geoscience and Remote Sensing, 2023, Vol.61, Article 5222420  
**Summary:** Hierarchical feature pyramid + spherical space classifier to handle large size variation and intraclass diversity in SAR ship recognition.  
**Relevance:** Hierarchical classification design reference; different target domain (ships) but architecture ideas transferable.

---

## 20. Multiview SAR Target Recognition Based on Adaptive Semantic-View Feature Embedding Fusion Network
**Authors:** Wang, Haochuan; Sun, Bing; Yang, Wei; Zeng, Hongcheng; Li, Chunsheng; Chen, Jie  
**Source:** IEEE Transactions on Geoscience and Remote Sensing, 2025, Vol.63, Article 5221022  
**Summary:** SeVi dual-encoder fuses view-aware features for multiview SAR ATR with a prototype-based missing-view completion strategy; evaluated on ATRNet-STAR, MSTAR, and SARAircraft.  
**Relevance:** Direct ATRNet-STAR benchmark — strong baseline for multiview/azimuth-aware recognition.

---

## 21. PolSAR-Tokenized Foundation Model for Robust Target Recognition
**Authors:** Gao, Fan; Zhang, Qingjun; Ren, Dawei; Lin, Xunhao; Yin, Junjun; Yang, Jian  
**Source:** IEEE Geoscience and Remote Sensing Letters, 2026, DOI: 10.1109/LGRS.2026.3673107  
**Summary:** Pre-trains DINOv2 on SARDet-186K and adapts to ATRNet-STAR as ChannelViT with sensor-mode-aware masking for single/dual/full polarization robustness (92.23% strict SOC-40).  
**Leakage finding:** The standard ATRNet-STAR SOC-40 benchmark splits the 4 polarization channels (HH, HV, VH, VV) of the same scene independently across train/test. Models memorize scene-level speckle patterns rather than learning target features — SARATR-X drops from 96.4% to 80.4% under the strict split. Fix: all channels of a scene must go to the same split.  
**Relevance:** Must-cite for benchmarking methodology; our `split_images()` now implements the scene-based split this paper requires.

---

## 22. SAR-GTR: Attributed Scattering Information Guided SAR Graph Transformer Recognition Algorithm
**Authors:** Xiong, Xuying; Zhang, Xinyu; Jiang, Weidong; Liu, Li; Liu, Yongxiang; Liu, Tianpeng  
**Source:** arXiv, 2025-05  
**Summary:** Graph Transformer combining GNN and Transformer with edge information enhancement for node/edge feature learning; validated on ATRNet-STAR vehicle dataset.  
**Relevance:** Direct ATRNet-STAR competitor using scattering-graph features — key comparison point.

---

## 23. Rollable Latent Space for Azimuth Invariant SAR Target Recognition
**Authors:** Sagi, Kazutoshi; Toizumi, Takahiro; Senda, Yuzo  
**Source:** IEEE — IGARSS 2018, p.21-24, Article 8519366  
**Summary:** Proposes a rollable latent space (RLS) where rolling latent features corresponds to 3D object rotation, enabling azimuth-invariant recognition and data augmentation from limited viewing angles; improves accuracy by 30% on unseen viewing angles.  
**Relevance:** Directly addresses azimuth invariance — core to our azimuth detector work.

---

## 25. Fast and Effective: Progressive Hierarchical Fusion Classification for Remote Sensing Images
**Authors:** Geng, Xueli; Jiao, Licheng; Li, Lingling; Liu, Xu; Liu, Fang; Yang, Shuyuan  
**Source:** IEEE Transactions on Multimedia, 2024, Vol.26, p.9776-9789  
**Summary:** PHFNet — lightweight FCN with pyramid dual-path encoder and ConvGRU decoder for multisource pixel-level remote sensing classification.  
**Relevance:** Weak — pixel-level fusion classification, not ATR. Keep only if we add a scene-level component.

---

## 26. ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild
**Authors:** Yongxiang Liu, Weijie Li, Li Liu, Jie Zhou, Bowen Peng, Yafei Song, Xuying Xiong, Wei Yang, Tianpeng Liu, Zhen Liu, Xiang Li  
**Source:** IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, DOI: 10.1109/TPAMI.2026.3656649  
**Summary:** Introduces the ATRNet-STAR dataset — 194,324 SAR images of 40 vehicle types across 5 scenes, 4 depression angles (15/30/45/60°), 4 polarizations (HH/HV/VH/VV), and 2 bands (X, Ku). Benchmarks 15 methods across 7 experimental settings (SOC-40, SOC-50, EOC-Scene, EOC-Depression, EOC-Azimuth, EOC-Band, EOC-Polarization). Best published SOC-40 accuracy: 96.4% (SARATR-X). Best EOC results: EOC-Band 89.2%, EOC-Pol 84.6%, EOC-Depression 43.1%, EOC-Azimuth 46.7%, EOC-Scene 29.7%.  
**Relevance:** Primary dataset paper — must cite. Provides the SOTA baseline numbers our work compares against. The official 7:3 train/test split used in these benchmarks has the cross-polarization leakage we identified and corrected.

---

## To Review

*(none)*

---

## Reviewed — Relevant

- 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 26

---

## Reviewed — Not Relevant / Weak

- 1, 2, 3 (MSTAR-only baselines, useful for background only)
- 10 (ship domain)
- 25 (pixel-level classification, not ATR)
