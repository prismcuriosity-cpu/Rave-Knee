# SACN + RGSSPD: Publication Plan

**Target venues:** *Medical Image Analysis* (MedIA) or *IEEE Transactions on
Medical Imaging* (TMI). Both are top-tier (A*/Q1) journals for the method +
clinical-validation profile of this work.

**Working title:** *Severity-Aware Cartilage Segmentation with Intrinsic
Feature Modulation and Thickness-Consistent Multi-Task Learning for Knee
Osteoarthritis.*

---

## 1. Motivation & clinical impact

Knee osteoarthritis (OA) affects hundreds of millions of people and is a
leading cause of disability. Quantitative cartilage morphometry — compartmental
volume and **thickness** — from MRI is the accepted imaging biomarker for OA
progression and for endpoint measurement in disease-modifying drug trials.
Automatic cartilage segmentation is the bottleneck: cartilage is a thin
(1–4 mm), low-contrast sheet whose appearance changes drastically with disease
severity (Kellgren–Lawrence, KL, grade 0–4). A single model trained across all
severities compromises between the smooth cartilage of mild knees and the
eroded, fibrillated, partially-denuded cartilage of severe knees.

**Contribution to humanity:** more accurate, severity-robust, *uncertainty-aware*
cartilage segmentation that reports the clinically-used thickness biomarker
directly — improving OA trial efficiency and enabling trustworthy deployment.

---

## 2. Methodological contributions (novelty)

1. **Intrinsic severity conditioning (SACN).** OA severity, proxied by a
   retrieval-based soft KL vote, modulates *every decoder stage* through FiLM.
   Unlike post-hoc adapter blending (our RGSSPD baseline), severity shapes
   feature computation itself. Ablation isolates FiLM vs. no-conditioning vs.
   input-concatenation vs. hard routing.

2. **Thickness-consistent multi-task learning.** Joint prediction of
   segmentation, a signed-distance boundary field, and a per-voxel
   **cartilage-thickness** map. Supervising the clinical endpoint (thickness)
   regularises the thin-structure boundary where Dice/HD95 are decided.

3. **Retrieval-gated severity specialists (RGSSPD).** The existing frozen-backbone
   SwinUNETR + three severity-specialist cross-attention heads gated by a
   temperature-scaled soft KL vote — retained as a strong, interpretable
   baseline and as a source of the severity gate for SACN.

4. **Evidential per-voxel uncertainty.** Optional Dirichlet head yields
   calibrated voxel uncertainty in one forward pass; evaluated for
   failure-detection utility (uncertainty vs. error correlation, ECE).

---

## 3. Data

- **Primary:** OAI-ZIB (507 manually-segmented knee MRIs; 6 labels —
  background, femur, femoral cartilage, tibia, medial + lateral tibial
  cartilage), with KL grades from the OAI clinical database.
- **Splits:** patient-level, stratified by KL grade. 5-fold cross-validation;
  report mean ± std across folds. No subject appears in more than one fold.
- **Severity strata:** mild (KL 0–1), moderate (KL 2), severe (KL 3–4).
- **External generalisation (if available):** a held-out site / second dataset
  for cross-dataset Dice to demonstrate robustness (reviewers expect this at
  MedIA/TMI).

---

## 4. Baselines

| # | Method | Purpose |
|---|--------|---------|
| B1 | 3D U-Net | classic reference |
| B2 | nnU-Net (3d_fullres) | SOTA auto-configured baseline (must-have) |
| B3 | SwinUNETR (frozen) | foundation-model baseline |
| B4 | SwinUNETR + RGSSPD | our retrieval-gated adapter |
| B5 | SACN (no severity) | ablates conditioning |
| **B6** | **SACN (full)** | **proposed** |

Same preprocessing, augmentation, patch size, and 5-fold splits for all.

---

## 5. Ablation matrix (SACN)

- Severity conditioning: none / input-concat / **FiLM** / hard-routing.
- Severity source: oracle KL / RGSSPD soft vote / learned-from-image.
- Aux heads: seg only / +boundary / +thickness / +both.
- Deep supervision: on/off.
- Evidential head: on/off (+ calibration metrics).
- Backbone width & depth sweep (base_channels, stage_depths).

RGSSPD's own A1–A10 conditions (retrieval, gating, temperature, k) carry over.

---

## 6. Metrics

- **Overlap:** Dice per class, mean cartilage Dice.
- **Surface:** HD95, Average Symmetric Surface Distance (ASSD).
- **Clinical:** compartmental cartilage **thickness** error (mean abs. error in
  mm) and volume error; correlation of predicted vs. reference thickness.
- **Severity-stratified** reporting for every metric (mild/moderate/severe) —
  the central claim is severity-robustness.
- **Uncertainty:** ECE, error–uncertainty correlation, risk–coverage curves.

---

## 7. Statistical protocol

- Paired **Wilcoxon signed-rank** tests between B6 and each baseline on
  per-subject Dice/HD95; Holm–Bonferroni correction across classes.
- **Bootstrap 95% CIs** (10k resamples) for all reported means.
- Report effect sizes; pre-register the primary endpoint (mean cartilage Dice,
  severe stratum) to avoid multiplicity concerns.

---

## 8. Reproducibility

- Fixed seeds; environment pinned (PyTorch, MONAI, CUDA versions).
- Release code, configs, trained weights, and per-subject metric CSVs.
- Deterministic evaluation pipeline; document any nondeterministic ops.

---

## 9. Figures

1. Architecture schematic (encoder / FiLM / multi-task heads).
2. Qualitative severe-KL cases: baseline vs. SACN vs. GT, with uncertainty overlay.
3. Severity-stratified Dice/HD95 bars with significance brackets.
4. Predicted vs. reference thickness scatter + Bland–Altman.
5. Blend-weight vs. true-KL (RGSSPD interpretability, existing figure).
6. Risk–coverage / calibration curves.

---

## 10. Status & what is / isn't validated here

- ✅ Implemented & unit/smoke-tested (CPU): `sacn_model.py` (architecture,
  losses, target-field helpers), refined `rgsspd_module.py`.
- ⏳ Requires GPU training on OAI-ZIB (RTX 5090): all Dice/HD95/thickness
  numbers, ablations, and statistical tests. The code here is written to be
  dropped into the existing training/eval pipeline for those runs.

---

## 11. Suggested paper timeline

1. Train B1–B6 with 5-fold CV; log per-subject metrics.
2. Run SACN ablations; select final config on validation folds only.
3. Statistical tests + figures on the held-out test fold.
4. External-dataset generalisation experiment.
5. Draft (Methods → Experiments → Results → Clinical discussion).
