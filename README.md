# Gesture Recognition Pipeline
**MLSM2154 – Artificial Intelligence** | UCLouvain Mons  
Professor: Marco Saerens | Assistants: Alexis Airson, Diego Eloi & Nicolas Szelagowski

> **Status:** Work in progress — baseline methods implemented (Edit Distance, DTW). State-of-the-art methods and statistical tests pending.

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
10. [Current Results](#current-results)
11. [What's Next](#whats-next)

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
├── data_loading.py         # Load Domain 1 and Domain 4 from raw .txt/.csv files
├── data_splitting.py       # Cross-validation strategies (dependent & independent)
├── data_preparation.py     # Normalization and PCA (fit on train, apply to both)
├── clustering.py           # K-Means, symbolic transformation, sequence compression, k-NN for edit distance
├── tool_from_scratch.py    # Edit distance (Levenshtein) and DTW — implemented from scratch
├── assessment.py           # Accuracy, confusion matrix aggregation
│
├── main.py                 # Master script — runs all experiments, saves results
│
├── results/                # Auto-generated output folder
│   ├── domain1_edit-distance_dependent.txt
│   ├── domain1_edit-distance_dependent_raw.csv
│   └── ...                 # 8 .txt + 8 _raw.csv files total
│
└── README.md
```

---

## Pipeline Architecture

The pipeline is modular — every function is reusable across methods and datasets.

### Edit Distance pipeline

```
Raw trajectories
      │
      ▼
Normalisation (z-score, fit on train)
      │
      ▼
[Optional] PCA (fit on train)
      │
      ▼
K-Means clustering → symbolic sequences ("AAABBBCCA...")
      │
      ▼
Sequence compression (remove consecutive duplicates → "ABCA")
      │
      ▼
Levenshtein edit distance + k-NN → prediction
```

### DTW pipeline

```
Raw trajectories
      │
      ▼
Normalisation (z-score, fit on train)
      │
      ▼
[Optional] PCA (fit on train)
      │
      ▼
Dynamic Time Warping distance + k-NN → prediction
```

> DTW works directly on 3D coordinates — no clustering or symbolic conversion needed.

---

## Installation

```bash
pip install numpy pandas scikit-learn tslearn scipy
```

All core algorithms (edit distance, DTW) are implemented from scratch in `tool_from_scratch.py` as required by the project guidelines. `tslearn` is used **only during hyperparameter search** to speed up computation — the final reported results use the custom implementation.

---

## Data

Place the data folders as follows, or update the paths in `main.py`:

```
GestureData/
├── GestureDataDomain1_Mons/
│   └── Domain1_csv/        # .csv files, one per gesture recording
└── GestureDataDomain4_Mons/
    └── *.txt               # .txt files, one per gesture recording
```

Each file contains the header metadata (subject ID, gesture type) followed by rows of `x, y, z` coordinates sampled over time. The `t` column (timestamp) is ignored — we assume constant time steps.

---

## How to Run

Open `main.py` and update the two data paths at the top of `__main__`:

```python
path_domain_1 = "/your/path/to/Domain1_csv"
path_domain_4 = "/your/path/to/GestureDataDomain4_Mons"
```

Then run:

```bash
python main.py
```

This will automatically loop over all combinations of dataset × method × CV mode and save one result file per combination in `./results/`.

---

## Hyperparameter Grid

The grid currently tested is:

| Parameter | Edit Distance | DTW |
|---|---|---|
| `k` (nearest neighbours) | 1, 3, 5, 7 | 1, 3, 5, 7 |
| `n_clusters` (K-Means alphabet size) | 5, 7, 9, 11, 13, 15 | — |
| `n_components` (PCA) | no\_pca, 2, 3 | no\_pca, 2, 3 |

For each configuration, both **user-dependent** and **user-independent** cross-validation are run on both **Domain 1** and **Domain 4**, giving **8 result files** in total.

---

## Output Files

For each of the 8 combinations, two files are written to `./results/`:

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
...

BEST CONFIG: ('no_pca', 7, 3)
Mean accuracy : 0.8234
Std           : 0.0512

CONFUSION MATRIX (best config)
----------------------------------------
[[45,  2,  0, ...],
 ...]]
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

> As expected by the guidelines, user-dependent accuracy is higher than user-independent — the model benefits from knowing the user's personal gesture style.

### Key implementation detail — no data leakage

All preprocessing (normalisation, PCA, K-Means) is **fit exclusively on the training set** of each fold and then applied to both train and test. This prevents any information from the test set influencing the model.

---

## Current Results

> To be completed after the overnight run.

---

## What's Next

- [ ] Run full hyperparameter grid overnight and fill in results table above
- [ ] Implement second baseline (DTW final evaluation with custom implementation)
- [ ] Implement at least 2 state-of-the-art methods (e.g. feature extraction + SVM, $1 recognizer or LSTM)
- [ ] Statistical hypothesis testing on user-independent results (Wilcoxon signed-rank test)
- [ ] Run everything on Domain 4
- [ ] Write final report (PDF, max 10 pages, deadline May 17 2026)

---

*Last updated: April 2026*