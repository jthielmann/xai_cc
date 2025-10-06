# Morphology → Gene Expression: Pipeline, Options, and Decisions

> **TL;DR default**  
> **Frozen encoder → PCA (k=256–512)** *(whiten if adding SAE)* → **(optional) Sparse Autoencoder (SAE)** → **tile→slide pooling (mean)** → **Ridge (multi-target)**.  
> Add **Elastic Net** for sparsity, **RF/XGBoost** only as non‑linear comparators.  
> For XAI: **virtual spatialization (linear maps)** + **CRP** + **PCX** on SAE concepts.

---

## 1) End‑to‑end overview

**Inputs:** WSIs/tiles or ST spots  
**Goal:** Predict bulk/spot gene expression vector from morphology

**Graph (per tile):**  
`tile → encoder (frozen) → embedding z ∈ ℝ^D → PCA (k) [±whiten] → (optional) SAE → a_tile → pool → ā_slide/spot → head → ŷ (genes)`

- **Encoder:** UNI, DINO, etc. **frozen**
- **PCA:** linear rotation/truncation; fit on **train only**
- **SAE (optional):** sparse, interpretable “concept” latents
- **Pooling:** mean to start (attention optional later)
- **Head:** **Ridge** (default), **Elastic Net** (sparser), MLP (only if lots of data), RF/XGBoost (non‑linear baseline)
- **XAI:** CRP & PCX downstream of SAE/linear head + tile‑level linear maps

---

## 2) Data & preprocessing

- **Image preprocessing:** deterministic resize/crop; encoder’s required mean/std.  
- **Augmentations:**  
  - **Simple path:** *no* random augs at embedding extraction; **precompute** embeddings and PCA.  
  - **If using augs:** fit PCA on the **train** distribution you’ll use (e.g., K augmented views per tile for train). At inference, optional **test‑time augmentation (TTA)** with the **same** PCA applied to each view.

---

## 3) Dimensionality reduction (PCA)

- **What it does:** center → rotate to top‑variance directions → truncate to **k=256–512**  
- **Fitting:** **train only** (no leakage); save `(μ, V_k, λ_k)`  
- **Whitening:** divide PCs by `√(λ_k + ε)`  
  - **Use when:** feeding an **SAE** next (helps cleaner sparse codes)  
  - **Optional when:** going straight to **Ridge/EN**
- **Placement:** **per‑tile before pooling** (keeps tile‑level explainability and saves memory)
- **Keep PCA in the graph for XAI:** implement as a fixed linear layer  
  - weights `W = V_k / √(λ_k+ε)` (or `V_k` if not whitening)  
  - bias `b = −μW`

---

## 4) Optional Sparse Autoencoder (SAE)

- **Purpose:** interpretable “concept” units (monosemantic-ish), better CRP/PCX
- **Input:** **whitened PCs** (recommended)
- **Width:** **1.5–4×** PCA dims (overcomplete)  
- **Sparsity:** **1–5%** average activation (Top‑K or L1), ReLU codes  
- **Training:** early stop on reconstruction; fit on **train** embeddings; freeze for head  
- **Output:** per‑tile codes `a_tile`; **pool** (mean) → `ā_slide/spot` to feed the head

---

## 5) Pooling (tile → slide/spot)

- **Default:** **mean pooling** of tile features (`a_tile` or PCs)  
- **Optional:** gated/attention pooling if mean plateaus and you have time

---

## 6) Prediction heads (choose by constraints)

**Recommended default:**
- **Ridge (multi‑target):** robust with limited slides; coefficients are interpretable  
  - α grid (logspace): `1e−6 … 10`, nested CV on train folds

**Also consider:**
- **Elastic Net:** adds sparsity; tune `(α, l1_ratio ∈ [0.1, 0.9])`; good for feature selection & XAI
- **Random Forest / XGBoost:** non‑linear comparators; use SHAP for attribution
- **MLP (Linear→ReLU→Linear):** only if **lots** of data; prefer **shared bottleneck** + per‑gene linear readout

---

## 7) Evaluation & validation

- **Splits:** patient‑stratified; **fit PCA/SAE only on train** then transform val/test
- **Metrics:**  
  - **Gene‑wise Pearson r** (pred vs. measured)  
  - **# of significant genes** (e.g., BH‑FDR)  
  - (Optional) mean r over **top‑50 HVGs** for comparability with common benchmarks
- **Stability:** bootstrap folds/genes to check coefficient & concept stability

---

## 8) Explainability (XAI)

- **Virtual spatialization (linear maps):**  
  - If head on PCs: `score_tile,g = w_g^T · z'_tile`  
  - If head on SAE: `score_tile,g = w_g^T · a_tile`  
  → heatmaps per gene showing **where** morphology drives expression
- **CRP (Concept Relevance Propagation):**  
  - Run end‑to‑end: output gene node → head → (SAE/PCs) → tiles → pixels  
  - Get **concept attributions** + **spatial heatmaps** per gene
- **PCX (Prototypical Concept‑based eXplanations):**  
  - Cluster **concept‑level explanation vectors** over many slides  
  - For regression: apply to **high‑prediction** subsets per gene (e.g., top 20%) to get representative prototypes
- **Coefficient‑based narratives:** for Ridge/EN, report top positive/negative concepts/PCs per gene

---

## 9) Decisions (fast path vs. bigger path)

- **Time short, XAI required →**  
  **Frozen encoder → PCA(256–512)** *(whiten if SAE)* → **(SAE optional)** → **mean pool** → **Ridge**  
  - Add **Elastic Net** if you want sparse, easy narratives  
  - Skip random augs; consider **TTA** only if needed
- **More data / time →**  
  - Add **attention pooling** or small **MLP** (shared bottleneck)  
  - Compare with **RF/XGBoost** for non‑linear lift  
  - Scale SAE width; curate concept library for CRP/PCX

---

## 10) Minimal hyperparameter cheatsheet

- **PCA k:** 256 (baseline) → 384/512 if budget allows  
- **Whitening:** **yes** if SAE; **no** (or try both) if direct Ridge/EN  
- **SAE:** width `2×k`, Top‑K `3–8` (≈1–4% density), LR `1e−3`, 5–15 epochs, early stop  
- **Ridge α:** `1e−6 … 10` (log grid)  
- **Elastic Net:** α same grid; `l1_ratio ∈ {0.2, 0.5, 0.8}`  
- **RF:** ~70–200 trees (quick), depth moderate; use as comparator  
- **Pooling:** mean first; attention only if needed

---

## 11) Deliverables to save/checkpoint

- `encoder_name/` frozen weights (hash/version)  
- `pca.joblib` (μ, V_k, λ_k, whiten flag)  
- `sae.pt` (if used) + sparsity stats  
- `head_ridge.pkl` (and/or EN / RF)  
- Scripts/notebooks for: metrics, **per‑gene heatmaps**, **CRP** runs, **PCX** prototypes  
- Repro notes: RNG seeds, tile sampling policy, slide inclusion criteria

---

## 12) Quick checklist (copy/paste)

- [ ] Freeze encoder; set deterministic preprocessing (no random augs)  
- [ ] Extract **train** embeddings; fit **PCA(k)** on train; decide whitening  
- [ ] (Optional) Train **SAE** on train PCs; freeze  
- [ ] Transform **train/val/test** with fitted PCA (and SAE if used)  
- [ ] **Pool** tile features to slide/spot  
- [ ] Train **Ridge** (and EN comparator) with nested CV (train only)  
- [ ] Evaluate: gene‑wise Pearson r, # significant genes (BH‑FDR), HVG summary  
- [ ] XAI: per‑gene tile heatmaps (linear maps), **CRP** concept + pixel attributions  
- [ ] PCX: build prototypes from concept explanations (per‑gene high‑prediction subset)  
- [ ] Package artifacts (PCA/SAE/heads) + reproducibility notes
