# Gesture Recognition Pipeline
**MLSMM2154 – Artificial Intelligence** | UCLouvain Mons  
Professor: Marco Saerens | Assistants: Alexis Airson, Diego Eloi & Nicolas Szelagowski

> **Status:** Complete — four classifiers implemented, evaluated, and reported (DTW, Edit-distance, Three-Cents, BiLSTM).

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
10. [Key Results](#key-results)

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
├── main.py                      # Main experiment runner (all methods)
├── statistical_assessment.py    # Friedman + Nemenyi + Wilcoxon–Holm tests
├── utils_ablation.py            # Ablation study across preprocessing configurations
│
├── data/
│   ├── data_loading.py          # Load Domain 1 and Domain 4 from raw files
│   ├── data_splitting.py        # Cross-validation strategies (user-dep. & indep.)
│   └── data_preparation.py      # Normalisation and PCA (fit on train, apply to both)
│
├── pipelines/
│   ├── pipeline_dtw.py          # DTW distance + k-NN
│   ├── pipeline_edit_distance.py  # K-Means alphabet + Levenshtein edit-distance k-NN
│   ├── pipeline_three_cent.py   # 3-Cent template-matching recogniser (3D, no rotation)
│   ├── pipeline_bilstm.py       # Bidirectional LSTM gesture classifier
│   └── pipeline_transformer.py  # Transformer encoder classifier (experimental)
│
├── utils/
│   ├── utils_algorithms.py      # Edit distance & DTW implemented from scratch (numba)
│   ├── utils_assessment.py      # Majority vote & evaluation helpers
│   ├── utils_saving.py          # Write .txt reports and _raw.csv files
│   └── utils_misc.py            # Miscellaneous utilities
│
├── viz/
│   ├── viz_pipeline.py          # Interactive Plotly pipeline visualisation dashboard
│   ├── viz_mds.py               # MDS 2-D embedding of gesture trajectories
│   ├── results_explorer.py      # Streamlit interactive results explorer
│   └── dashboard.html           # Pre-rendered dashboard
│
├── results/                     # Auto-generated output folder
│   ├── domain1_dtw_dependent.txt
│   ├── domain1_dtw_dependent_raw.csv
│   └── ...
│
├── ablation_results/            # Output of utils_ablation.py
├── rapport/                     # LaTeX source of the final report
└── README.md
```

---

## Pipeline Architecture

The pipeline is modular — every preprocessing and evaluation step is reusable across methods and datasets. All preprocessing is **fit exclusively on the training set** of each fold and applied to both train and test (no data leakage).

### DTW

```
Raw trajectories → Normalisation → [Optional PCA]
  → Dynamic Time Warping distance (optional Sakoe-Chiba window) + k-NN → prediction
```

### Edit Distance

```
Raw trajectories → Normalisation → [Optional PCA]
  → K-Means clustering → symbolic sequences ("AAABBBCCA…")
  → [Optional compression] (remove consecutive duplicates → "ABCA")
  → Levenshtein edit distance (normalised by max sequence length) + k-NN → prediction
```

> Edit distance is divided by `max(len(s1), len(s2))` so the metric falls in `[0, 1]` and is comparable across gesture pairs of different symbolic length.

### 3-Cent (3D)

```
Raw trajectories → Normalisation → [Optional PCA]
  → Resample to N points → scale by arc length (uniform)
  → translate to centroid (no rotation — direction is discriminative)
  → path distance to nearest template → prediction
```

> 3-Cent keeps gesture orientation intact, which is important for 3D mid-air gestures where direction is meaningful.

### BiLSTM

```
Raw trajectories → Normalisation
  → Resample to fixed length → Bidirectional LSTM (input dropout)
  → BatchNorm → Dropout → Dense → softmax → prediction
```

> Dropout rate is now a tunable hyperparameter (Srivastava et al. 2014) and is swept alongside sequence length and hidden units during HP selection.

---

## Installation

```bash
pip install numpy pandas scikit-learn scipy numba
pip install tensorflow          # for BiLSTM
pip install tqdm
pip install plotly              # for viz/viz_pipeline.py
pip install streamlit           # for viz/results_explorer.py
```

> All core algorithms (edit distance, DTW) are implemented from scratch in `utils/utils_algorithms.py` using numba JIT compilation, as required by the project guidelines.

---

## Data

Place the data folders as follows, or update the paths at the top of `main.py`:

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

### Main experiment runner

Update the two data paths at the top of `main.py`, then:

```bash
python main.py
```

Loops over all combinations of dataset × method × CV mode and saves one result file per combination in `./results/`.

### Statistical tests

```bash
python statistical_assessment.py
```

Runs the Friedman test followed by Nemenyi and Wilcoxon–Holm post-hoc comparisons on the user-independent results stored in `./results/`.

### Ablation study

```bash
python utils_ablation.py
```

Evaluates all combinations of normalisation and PCA settings for each classifier and saves results to `./ablation_results/`.

### Interactive visualisation

```bash
python viz/viz_pipeline.py                   # pipeline visualisation (Plotly)
streamlit run viz/results_explorer.py        # interactive results explorer
python viz/viz_mds.py                        # MDS embedding of trajectories
```

---

## Hyperparameter Grid

| Parameter | Edit-distance | DTW | 3-Cent | BiLSTM |
|---|---|---|---|---|
| `k` (nearest neighbours) | 1, 3, 5, 7 | 1, 3, 5, 7 | — | — |
| `n_clusters` (K-Means) | 15, 20, 25, 30, 35, 40 | — | — | — |
| `compression` | True / False | — | — | — |
| `n_components` (PCA) | no\_pca, 2, 3 | no\_pca, 2, 3 | no\_pca, 2, 3 | — |
| `w` (Sakoe-Chiba window) | — | None, 10, 20 | — | — |
| `n_points` (resample) | — | — | 16, 32, 64 | 16, 32, 64 |
| `n_units` (hidden size) | — | — | — | 16, 32, 64 |
| `dropout_rate` | — | — | — | 0.1, 0.2, 0.3, 0.5 |

All combinations are evaluated under **user-dependent** and **user-independent** CV on both **Domain 1** and **Domain 4**.

---

## Output Files

For each experiment combination, two files are written to `./results/`:

**`{domain}_{method}_{cv_mode}.txt`** — human-readable report:
```
============================================================
RESULTS — domain1_edit-distance_dependent
Generated: 2026-04-07 01:14:02
============================================================

BEST CONFIG: ('no_pca', 35, 1)
Mean accuracy : 0.9910
Std           : 0.0071

CONFUSION MATRIX (best config)
----------------------------------------
[[100,  0,  0, ...],
 ...]
```

**`{domain}_{method}_{cv_mode}_raw.csv`** — one row per fold, for statistical testing.

---

## Validation Strategy

### User-independent (Leave-One-User-Out)

The test user is completely unseen during training. There are **10 folds**, one per user:

```
Fold 0:  Train = users [1..9]     Test = user [0]   → 900 train / 100 test
Fold 1:  Train = users [0,2..9]   Test = user [1]   → 900 train / 100 test
...
```

### User-dependent (Leave-One-Repetition-Out)

The model has seen all users but not this specific repetition. There are **10 folds**, one per repetition index:

```
Fold 0:  Train = repetitions [1..9]   Test = repetition [0]  → 900 train / 100 test
Fold 1:  Train = repetitions [0,2..9] Test = repetition [1]  → 900 train / 100 test
...
```

All preprocessing (normalisation, PCA, K-Means) is **fit exclusively on the training set** of each fold. Deep learning models are rebuilt and retrained from scratch at every fold.

---

## Key Results

Best test accuracy (mean ± std across 10 folds) using the best hyperparameter configuration per method:

| Method | Domain 1 indep. | Domain 1 dep. | Domain 4 indep. | Domain 4 dep. |
|---|---|---|---|---|
| DTW | 82.7 ± 14.6 % | 99.5 ± 0.5 % | 72.6 ± 13.4 % | **99.1 ± 1.3 %** |
| Edit-distance | 74.6 ± 20.9 % | 98.6 ± 1.0 % | 66.2 ± 13.5 % | 98.3 ± 1.5 % |
| **Three-Cents** | **96.3 ± 4.6 %** | **99.8 ± 0.4 %** | **95.1 ± 5.3 %** | 98.6 ± 1.3 % |
| BiLSTM | 85.7 ± 11.7 % | 96.0 ± 3.4 % | 73.7 ± 9.8 % | 87.9 ± 6.1 % |

Three-Cents is the only method statistically superior to all others in the user-independent setting (Friedman test, $p < 0.001$; Nemenyi and Wilcoxon–Holm post-hoc, $\alpha = 0.05$). Edit-distance results improved after normalising the Levenshtein distance by sequence length.

---

*Last updated: May 2026*
