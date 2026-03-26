# CNN Odd-One-Out Image Classifier

**McGill COMP551 Mini-Project 3** — Group 110
Ryu Quan · Yantian Yin · Junu Seo

---

## Task

Given groups of **5 grayscale 32×32 images**, identify the index (0–4) of the **odd-one-out** — the one image that does not share the hidden property of the other four. This is a **5-class classification** problem where random chance is 20%.

- **Training set:** 3,000 groups
- **Test set:** 2,000 groups (1,000 public / 1,000 private leaderboard)
- **Parameter budget:** ≤ 25,000 trainable parameters

---

## Baseline

A logistic regression on flattened inputs achieved **~19.4%** validation accuracy — essentially random.

---

## Model Architecture

A compact CNN with **24,992 trainable parameters** built around two key ideas:

### 1. Depthwise Separable Convolutions
Standard convolutions are replaced with depthwise + pointwise convolutions to stay within the parameter budget while maintaining expressiveness.

> Example: a 32→32 conv with standard conv costs `32×32×3×3 = 9,216` params; depthwise+pointwise costs `32×3×3 + 32×32 = 1,312` params — a **7× reduction**.

### 2. Leave-One-Out (LOO) Comparison Head
After encoding each of the 5 images independently, a **leave-one-out mean** is computed per image: the embedding of each image is subtracted from the mean of the other four. This difference vector captures how much each image deviates from the group, which is exactly what the task requires.

```
e      — (B, 5, d)   per-image embeddings
loo    — (B, 5, d)   leave-one-out mean of the other 4
diff   — e - loo     deviation from the group
score  — Linear([e, diff, |diff|]) → 1 scalar per image → 5-class logits
```

### Encoder Blocks

| Block | Operation | Output shape |
|-------|-----------|--------------|
| Stem  | Conv 1→16, stride 2 | 16×16×16 |
| 1     | DW+PW 16→32, stride 2 | 32×8×8 |
| 2     | DW+PW 32→32, stride 1 | 32×8×8 |
| 3     | DW+PW 32→64, stride 2 | 64×4×4 |
| 4     | DW+PW 64→64, stride 1 | 64×4×4 |
| 5     | DW+PW 64→32, stride 2 | 32×2×2 |
| Pool  | AdaptiveAvgPool → Linear(32, 95) | 95-dim embedding |

Blocks alternate between **stride-2 downsampling** and **stride-1 refinement** to preserve spatial information longer before compression.

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| LR schedule | Linear warmup (20 epochs) → CosineAnnealing (280 epochs) |
| Label smoothing | 0.1 |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Epochs | ~255 (from CV) |

### Data Augmentation
Applied per group during training:

1. **Random permutation** of the 5 images (label updated accordingly)
2. **Random horizontal/vertical flip**
3. **Gaussian noise** (σ = 0.01)
4. **Random crop** (pad 2px with reflection, crop back to 32×32)
5. **Random rotation** ±8°

---

## Results

| Model | Validation Accuracy |
|-------|-------------------|
| Logistic Regression (baseline) | 19.4% |
| CNN (5-fold CV mean ± std) | **89.9% ± 1.0%** |
| CNN (public test / Kaggle leaderboard) | **76.0%** |

5-fold cross-validation was used to select hyperparameters and determine the optimal number of training epochs before fitting on the full training set.

---

## Parameter Count Verification

```
Total trainable parameters: 24,992  ✓  (≤ 25,000)
```

---

## Files

| File | Description |
|------|-------------|
| `code (1).ipynb` | Full training notebook with outputs |
| `predicted_labels.csv` | Kaggle submission (2,000 predictions) |
| `best_model.pt` | Saved model weights |
