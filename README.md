# INDUCTION-MOTOR-FAILURE-PREDICTION
# Induction Motor Failure Prediction via Vibration Signal Classification

A deep learning and machine learning comparison study for detecting mechanical imbalance faults in induction motors using raw vibration signals. The project pits a custom 1D CNN (with residual blocks and Squeeze-and-Excitation attention) against traditional baselines — all on **identical raw signal inputs** — for a fair, end-to-end evaluation.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)

---

## Problem Statement

Induction motors are susceptible to rotor imbalance faults, which if undetected lead to mechanical failure and costly downtime. This project frames fault detection as a **binary classification** task:

- **Class 0 — Normal:** Healthy motor vibration signal
- **Class 1 — Imbalance:** Faulty motor vibration signal (rotor mass imbalance at varying severity levels)

The central research question is: *Can a deep CNN learn discriminative fault features directly from raw time-series signals, outperforming traditional ML models that also receive raw signals (no hand-crafted features)?*

---

## Dataset

The dataset is loaded from an `archive/` directory with the following structure:

```
archive/
├── normal/          # 49 healthy motor recordings
└── imbalance/
    ├── 6g/          # 49 samples — mild imbalance
    ├── 10g/         # 48 samples
    ├── 15g/         # 48 samples
    ├── 20g/         # 49 samples
    ├── 30g/         # 47 samples
    └── 35g/         # 45 samples — severe imbalance
```

| Property | Value |
|---|---|
| Total Samples | 335 |
| Signal Length | 15,000 samples per recording |
| Normal Samples | 49 (14.6%) |
| Imbalance Samples | 286 (85.4%) |
| Total Dataset Memory | ~19.17 MB |

Each sample is a univariate time-series CSV file (one column of float32 values). Files with lengths other than 15,000 are resampled using `scipy.signal.resample`.

**Class Imbalance:** The dataset is heavily skewed (~5.8:1 imbalance-to-normal ratio). This is addressed with SMOTE (with random oversampling as a fallback).

---

## Project Structure

```
.
├── INDUNCTION_MOTOR_FAILURE_PREDICTION.ipynb   # Main notebook
├── archive/                                     # Dataset directory
│   ├── normal/
│   └── imbalance/
│       ├── 6g/, 10g/, 15g/, 20g/, 30g/, 35g/
└── fair_comparison_all_raw_signals.png          # Output visualization
```

---

## Methodology

### 1. Data Loading & Preprocessing

- Each CSV is read with `pandas` (chunked for large files) and cast to `float32` for memory efficiency.
- All signals are resampled to a fixed length of **15,000 samples** using `scipy.signal.resample`.
- Signals are **z-score normalized** per sample: `(x - mean) / (std + 1e-8)`.
- CNN input is reshaped to `(N, 15000, 1)`.

### 2. Class Imbalance Handling

- **SMOTE** (`imblearn`) is attempted on the flattened CNN input. Due to the very high dimensionality (15,000 features), SMOTE typically fails and falls back to **random oversampling** of the minority class.
- After balancing: 229 normal + 229 imbalance = **458 training samples**.

### 3. Train/Test Split

- 80/20 stratified split: **268 training**, **67 test** samples.
- The same split indices are reused across all models for strict comparability.

### 4. Data Augmentation (CNN only)

Real-time augmentation is applied via a custom `DataGenerator` (50% probability per sample per epoch):
- **Gaussian noise** injection: `σ = 0.05 × signal_std`
- **Amplitude scaling:** uniform random in `[0.95, 1.05]`
- **Temporal shift:** random circular roll up to ±2% of signal length

### 5. Feature Engineering (RF baseline reference only)

A 55-dimensional feature vector is extracted for reference — **not used in the final fair comparison**. Features include:
- **Time-domain (15):** mean, std, variance, max, min, median, peak-to-peak, RMS, skewness, kurtosis, crest factor, MAE, shape factor, 25th/75th percentiles
- **Frequency-domain (25):** top-20 FFT magnitudes, FFT mean, std, max, dominant frequency index, total spectral energy
- **Wavelet-domain (15):** Daubechies-4 (db4) wavelet decomposition at 4 levels — mean, std, max absolute coefficient per sub-band

---

## Model Architectures

### Pure CNN (Primary Model)

A custom **ResNet-style 1D CNN** with Squeeze-and-Excitation attention, trained end-to-end on raw signals.

**Architecture overview:**

```
Input: (15000, 1)
│
├── Conv1D(64, k=15, stride=2) → BN → ReLU → MaxPool(3, stride=2)
│
├── Residual Block [64 filters, k=11]  → SE Attention → Dropout(0.20)
├── Residual Block [128 filters, k=9, stride=2] → SE Attention → Dropout(0.25)
├── Residual Block [256 filters, k=7, stride=2] → SE Attention → Dropout(0.30)
├── Residual Block [512 filters, k=5, stride=2] → SE Attention → Dropout(0.35)
│
├── GlobalAveragePooling1D
│
├── Dense(512, relu) + BN + Dropout(0.50)  [L1=1e-5, L2=1e-4]
├── Dense(256, relu) + BN + Dropout(0.40)  [L1=1e-5, L2=1e-4]
├── Dense(128, relu) + Dropout(0.30)       [L1=1e-5, L2=1e-4]
│
└── Dense(1, sigmoid)
```

**Residual Block:** Each block has two Conv1D layers with BN, followed by an identity or projection shortcut (1×1 Conv1D if dimensions change), with residual addition and ReLU.

**Squeeze-and-Excitation Block:** Channel-wise attention computed as GlobalAvgPool → Dense(C/16, relu) → Dense(C, sigmoid) → channel-wise multiply.

| Parameter | Value |
|---|---|
| Total Parameters | 3,627,517 (~13.84 MB) |
| Trainable Parameters | 3,620,221 |
| Non-trainable Parameters | 7,296 (BatchNorm) |

**Training configuration:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Accuracy, AUC, Recall
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Batch size: 32
- Real-time augmentation via custom `DataGenerator`

---

### Logistic Regression (Baseline)

- Input: Raw flattened signal (15,000 features)
- Trained on the SMOTE-balanced training set

### Random Forest (Baseline)

- Input: Raw flattened signal (15,000 features) — same as CNN/LR
- Default scikit-learn parameters

> **Note on fairness:** Although feature engineering was explored for RF (55-feature vectors), the final comparison uses **raw signals for all three models** to ensure a like-for-like evaluation.

---

## Results

All models trained and evaluated on **identical raw vibration signals** (no feature engineering).

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Specificity |
|---|---|---|---|---|---|---|
| **Pure CNN** | 0.6866 | **0.9737** | 0.6491 | 0.7789 | **0.9263** | **0.9000** |
| Logistic Regression | 0.4925 | 0.8286 | 0.5088 | 0.6304 | 0.5070 | 0.4000 |
| Random Forest | **0.8507** | 0.8507 | **1.0000** | **0.9194** | 0.6719 | 0.0000 |

### Rankings (by ROC-AUC)

1. **Pure CNN** — ROC-AUC: 0.9263 *(37.86% better than RF, 82.70% better than LR)*
2. **Random Forest** — ROC-AUC: 0.6719 *(32.53% better than LR)*
3. **Logistic Regression** — ROC-AUC: 0.5070

### Key Observations

- The **CNN achieves the best ROC-AUC (0.9263)** and highest precision (0.9737) and specificity (0.9000), meaning it is the best at distinguishing healthy vs. faulty motors and generates very few false alarms.
- **Random Forest** achieves perfect recall (1.0000) — it catches every fault — but at the cost of a very high false-positive rate (specificity 0.0000), suggesting it defaults to predicting the majority class almost always.
- **Logistic Regression** struggles significantly on raw 15,000-dimensional signals (ROC-AUC ≈ 0.507, essentially random), confirming that linear models cannot learn meaningful representations from high-dimensional raw waveforms without preprocessing.
- The CNN's lower raw accuracy (0.6866) relative to RF (0.8507) is a consequence of the class imbalance: RF classifies most samples as faulty and gets a high accuracy score, but fails to identify any normal samples (specificity 0.0000).
- ROC-AUC is the most reliable metric here given class imbalance — by this metric, **CNN is the clear winner**.

### Output Artifacts

- `fair_comparison_all_raw_signals.png` — Multi-panel visualization including grouped bar chart (all metrics), ROC curves for all three models, and confusion matrices.

---

## Dependencies

```txt
Python 3.8+
tensorflow==2.13.0
scikit-learn
imbalanced-learn
scipy
PyWavelets (pywt)
pandas
numpy
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install tensorflow==2.13.0 scikit-learn imbalanced-learn scipy PyWavelets pandas numpy matplotlib seaborn
```

> **GPU Note:** The notebook was run on CPU (`GPU Available: False`). GPU support is automatic via TensorFlow if CUDA is configured.

> **Thread configuration:** OpenMP, MKL, OpenBLAS, and NumExpr thread counts are all pinned to 1 at the top of the notebook to prevent thread contention on CPU.

---

## Usage

1. **Clone / download** this repository and place your dataset in the `archive/` directory following the structure described above.

2. **Install dependencies** (see above).

3. **Run the notebook:**

```bash
jupyter notebook INDUNCTION_MOTOR_FAILURE_PREDICTION.ipynb
```

4. Execute cells sequentially. The notebook will:
   - Load and preprocess all vibration signals
   - Apply SMOTE / random oversampling to balance classes
   - Build and train the Pure CNN with real-time augmentation
   - Train Logistic Regression and Random Forest on the same raw signals
   - Evaluate and compare all models
   - Generate and save `fair_comparison_all_raw_signals.png`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Fixed signal length (15,000) | Enables batch processing; signals resampled via `scipy.signal.resample` |
| Per-sample z-score normalization | Removes sensor offset and gain differences across recordings |
| SMOTE → fallback to random oversampling | SMOTE fails at 15,000 dimensions; random oversampling maintains minority class distribution |
| Squeeze-and-Excitation attention | Lets the CNN selectively weight the most informative frequency/amplitude channels in each layer |
| Residual connections | Mitigates vanishing gradients in the deep network; allows learning identity mappings |
| L1+L2 regularization in Dense layers | Reduces overfitting in the classification head given the small dataset (335 samples) |
| Progressive dropout (0.20 → 0.50) | Higher dropout deeper in the network where overfitting is more likely |
| Fair comparison protocol | All three models receive raw signals (no engineered features), isolating the effect of architecture |
| ROC-AUC as primary ranking metric | More robust than accuracy under class imbalance; measures ranking quality across all thresholds |
