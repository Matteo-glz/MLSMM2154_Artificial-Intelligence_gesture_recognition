# Gesture Recognition Pipeline
**MLSMM2154 – Artificial Intelligence** | UCLouvain Mons  
Professor: Marco Saerens | Assistants: Alexis Airson, Diego Eloi & Nicolas Szelagowski

> **Status:** All five baselines implemented and evaluated — Edit Distance, DTW, $1 Dollar, 3-Cent, BiLSTM, and Transformer.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation](#installation)
5. [Data](#data)
6. [How to Run](#how-to-run)
7. [Hyperparameter Grid](#hyperparameter-grid)
8. [Output Files](#output-files)
9. [Validation Strategy](#validation-strategy)
10. [What's Next](#whats-next)

---

## Project Overview

This project implements a **3D hand gesture recognition system** using statistical machine learning techniques. A hand trajectory is recorded as a sequence of `(x, y, z)` coordinates over time. The system classifies each trajectory into one of 10 gesture categories.

Two datasets are used:
- **Domain 1** — digits 0 to 9 drawn in 3D
- **Domain 4** — 3D geometric figures (Cuboid, Cylinder, Sphere, etc.)

Both datasets share the same structure: **10 users × 10 gesture types × 10 repetitions = 1000 sequences** each.

---

## Repository Structure

```
project/
│
├── data_loading.py              # Load Domain 1 and Domain 4 from raw .txt/.csv files
├── data_splitting.py            # Cross-validation strategies (user-dependent & independent)
├── data_preparation.py          # Normalisation and PCA (fit on train, apply to both)
│
├── baseline_edit_distance.py    # K-Means alphabet + Levenshtein edit-distance k-NN
├── baseline_dollar_one.py       # $1 Dollar template-matching recogniser (3D)
├── baseline_three_cent.py       # 3-Cent template-matching recogniser (3D, no rotation)
├── baseline_bilstm.py           # Bidirectional LSTM gesture classifier
├── baseline_transformer.py      # Transformer encoder gesture classifier
│
├── utils_algorithms.py          # Edit distance & DTW implemented from scratch (numba)
├── utils_assessment.py          # Majority vote
├── utils_saving.py              # Write .txt reports and _raw.csv files
├── utils_misc.py                # Experimental / scratch utilities
│
├── main.py                      # Standard experiment runner (all baselines)
├── main_optimized.py            # Parallelised runner (~90 % CPU utilisation)
├── precompute_results.py        # Parallel grid precomputation (joblib)
├── results_explorer.py          # Streamlit interactive results explorer
│
├── viz_pipeline.py              # Interactive Plotly pipeline visualisation dashboard
├── viz_mds.py                   # MDS 2-D embedding of gesture trajectories
│
├── results/                     # Auto-generated output folder
│   ├── domain1_edit-distance_dependent.txt
│   ├── domain1_edit-distance_dependent_raw.csv
│   └── ...
│
└── README.md
```

---

## Pipeline Architecture

The pipeline is modular — every preprocessing and evaluation step is reusable across methods and datasets. All preprocessing is **fit exclusively on the training set** of each fold and applied to both train and test (no data leakage).

### Edit Distance

```
Raw trajectories → Normalisation → [Optional PCA]
  → K-Means clustering → symbolic sequences ("AAABBBCCA…")
  → [Optional compression] (remove consecutive duplicates → "ABCA")
  → Levenshtein edit distance + k-NN → prediction
```

### DTW

```
Raw trajectories → Normalisation → [Optional PCA]
  → Dynamic Time Warping distance + k-NN → prediction
```

> DTW works directly on 3D coordinates — no clustering or symbolic conversion needed.

### $1 Dollar (3D)

```
Raw trajectories → Normalisation → [Optional PCA]
  → Resample to N points → rotate to indicative angle
  → scale by bounding box → translate to centroid
  → Golden Section Search over ±45° → nearest template → prediction
```

### 3-Cent (3D)

```
Raw trajectories → Normalisation → [Optional PCA]
  → Resample to N points → scale by arc length (uniform)
  → translate to centroid (no rotation — direction is discriminative)
  → path distance to nearest template → prediction
```

> 3-Cent keeps gesture orientation intact, which is important for 3D mid-air gestures where direction is meaningful (swipe-left ≠ swipe-right).

### BiLSTM

```
Raw trajectories → Normalisation
  → Resample to fixed length → Bidirectional LSTM
  → BatchNorm → Dropout → Dense → softmax → prediction
```

### Transformer

```
Raw trajectories → Normalisation
  → Resample to fixed length → Dense projection to d_model
  → + sinusoidal positional encoding
  → Multi-Head Self-Attention + LayerNorm
  → Feed-Forward Network + LayerNorm
  → Global Average Pooling → Dropout → softmax → prediction
```

---

## Installation

```bash
pip install numpy pandas scikit-learn scipy numba
pip install tensorflow          # for BiLSTM and Transformer
pip install tqdm                # for main_optimized.py
pip install plotly              # for viz_pipeline.py
pip install streamlit           # for results_explorer.py
pip install joblib              # for precompute_results.py
```

> All core algorithms (edit distance, DTW) are implemented from scratch in `utils_algorithms.py` using numba JIT compilation, as required by the project guidelines.

---

## Data

Place the data folders as follows, or update the paths in `main.py`:

```
GestureData_Mons/
├── GestureDataDomain1_Mons/
│   └── Domain1_csv/        # .csv files, one per gesture recording
└── GestureDataDomain4_Mons/
    └── *.txt               # .txt files, one per gesture recording
```

Each file contains header metadata (subject ID, gesture type) followed by rows of `x, y, z` coordinates sampled over time. The timestamp column is ignored — constant time steps are assumed.

---

## How to Run

### Standard runner

Update the two data paths at the top of the `__main__` block in `main.py`, then:

```bash
python main.py
```

Loops over all combinations of dataset × method × CV mode and saves one result file per combination in `./results/`.

### Parallelised runner (recommended for Edit Distance and 3-Cent)

```bash
python main_optimized.py                       # domain 1+4, edit-distance + 3-Cent
python main_optimized.py --domain 1            # domain 1 only
python main_optimized.py --method ed tc dtw    # include DTW
python main_optimized.py --cv dependent        # one CV mode only
python main_optimized.py --jobs 8              # cap worker count
```

Uses ~90 % of available CPU cores via fine-grained task parallelism. Wall-clock time ≈ time for a single fold instead of 20×.

### BiLSTM and Transformer

Each deep learning baseline has its own convenience wrapper:

```python
from baseline_bilstm import run_bilstm_for_dataset
from baseline_transformer import run_transformer_for_dataset

run_bilstm_for_dataset("domain1", gestures, cv_mode="dependent")
run_transformer_for_dataset("domain1", gestures, cv_mode="independent")
```

Or run the file directly after updating the data paths in the `__main__` block:

```bash
python baseline_bilstm.py
python baseline_transformer.py
```

### Interactive visualisation dashboard

```bash
python viz_pipeline.py                   # Domain 1, all sections
python viz_pipeline.py --domain 4 --open # Domain 4, auto-open browser
python viz_pipeline.py --skip-baselines  # skip slow CV sections
```

### Results explorer

```bash
streamlit run results_explorer.py
```

---

## Hyperparameter Grid

| Parameter | Edit Distance | DTW | $1 / 3-Cent | BiLSTM | Transformer |
|---|---|---|---|---|---|
| `k` (nearest neighbours) | 1, 3, 5, 7 | 1, 3, 5, 7 | — | — | — |
| `n_clusters` (K-Means) | 5 → 21 (step 2) | — | — | — | — |
| `compression` | True / False | — | — | — | — |
| `n_components` (PCA) | no\_pca, 1, 2, 3 | no\_pca, 1, 2, 3 | no\_pca, 1, 2, 3 | — | — |
| `n_points` (resample) | — | — | 16, 32, 64, 128, 256 | 32, 64, 128 | 32, 64, 128 |
| `n_units` (BiLSTM size) | — | — | — | 32, 64, 128 | — |
| `d_model` (Transformer dim) | — | — | — | — | 32, 64, 128 |

All combinations are evaluated across **user-dependent** and **user-independent** CV on both **Domain 1** and **Domain 4**.

---

## Output Files

For each experiment combination, two files are written to `./results/`:

**`{domain}_{method}_{cv_mode}.txt`** — human-readable report:
```
============================================================
RESULTS — domain1_edit-distance_dependent
Generated: 2026-04-07 01:14:02
============================================================

FULL SUMMARY (mean accuracy ± std per config)
----------------------------------------
n_components  n_clusters  k
no_pca        7           3     mean: 0.8234   std: 0.0512
              7           5     mean: 0.8101   ...

BEST CONFIG: ('no_pca', 7, 3)
Mean accuracy : 0.8234
Std           : 0.0512

CONFUSION MATRIX (best config)
----------------------------------------
[[45,  2,  0, ...],
 ...]
```

**`{domain}_{method}_{cv_mode}_raw.csv`** — one row per fold, for further analysis or statistical testing.

---

## Validation Strategy

### Cross-validation — what is a fold?

Instead of splitting data once (unreliable), we repeat the train/test split multiple times in different ways and average the results. Each split is called a **fold**.

### User-independent (Leave-One-User-Out)

The test user is completely unseen during training. There are **10 folds**, one per user:

```
Fold 0:  Train = users [1..9]     Test = user [0]   → 900 train / 100 test
Fold 1:  Train = users [0,2..9]   Test = user [1]   → 900 train / 100 test
...
Fold 9:  Train = users [0..8]     Test = user [9]   → 900 train / 100 test
```

This evaluates whether the system generalises to a **new user** it has never seen.

### User-dependent (Leave-One-Repetition-Out)

The model has seen all users but not this specific repetition. There are **10 folds**, one per repetition index:

```
Fold 0:  Train = repetitions [1..9]   Test = repetition [0]  → 900 train / 100 test
Fold 1:  Train = repetitions [0,2..9] Test = repetition [1]  → 900 train / 100 test
...
```

This evaluates performance when the system has been **calibrated on the specific user**.

> As expected, user-dependent accuracy is higher than user-independent — the model benefits from knowing the user's personal gesture style.

### No data leakage

All preprocessing (normalisation, PCA, K-Means) is **fit exclusively on the training set** of each fold and then applied to both train and test. Deep learning models are rebuilt and retrained from scratch at every fold.

---

## What's Next

- [ ] Statistical hypothesis testing on user-independent results (Wilcoxon signed-rank test)
- [ ] Write final report (PDF, max 10 pages, deadline May 17 2026)

---

*Last updated: April 2026*
