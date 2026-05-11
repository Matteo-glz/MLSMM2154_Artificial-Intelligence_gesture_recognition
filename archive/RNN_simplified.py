"""
RNN_simplified.py
─────────────────────────────────────────────────────────────────────────────
Bidirectional LSTM gesture recognizer — padding + masking approach.

Why padding + masking instead of resampling
-------------------------------------------
Resampling maps every trajectory onto a fixed time grid, destroying information
about the gesture's speed profile (how quickly each sub-movement is executed).
Padding preserves the original timing structure: short trajectories receive
zero-frames at the tail, and the Masking layer instructs the LSTM to skip those
frames entirely — the network only processes authentic trajectory points.
On a dataset of 1 000 sequences, retaining every real data point matters.

Why no attention / multi-head layers
-------------------------------------
With ~900 training samples per fold, attention adds O(T²) parameters (quadratic
in sequence length) with no proven benefit over the LSTM's built-in gated memory
for tasks of this scale.  Adding complexity on a small dataset primarily hurts
generalisation; the BiLSTM's hidden state already integrates long-range context.

Why fixed hyperparameters (no grid search)
------------------------------------------
Cross-validation already uses all samples, leaving no held-out set for fair
hyperparameter selection.  Running a grid search here would implicitly tune on
the test fold, inflating reported accuracy.  Fixing the architecture avoids this
and keeps the comparison fair with the other baseline methods.

Architecture
------------
  Masking  →  BiLSTM(64)  →  BatchNorm  →  Dropout(0.3)  →  Dense(32)  →  Dense(10)

  • Masking:     flags zero-padded frames; LSTM never processes them.
  • BiLSTM(64):  forward + backward temporal context; many gestures have
                 meaningful patterns in both reading directions.
  • BatchNorm:   normalises activations — critical for stable training on
                 small datasets where gradient magnitudes can vary wildly.
  • Dropout(0.3):prevents co-adaptation of units; especially important with
                 only ~900 samples per fold.
  • Dense(32):   compact discriminative representation before classification.
  • Dense(10):   softmax over 10 gesture classes.

Public API
----------
  build_bilstm_model(input_shape, n_classes)     → keras.Sequential
  run_bilstm_pipeline(gestures, cv_mode,
                      domain_name, seed=42)       → (pd.DataFrame, dict)
  run_and_save(domain_name, gestures, cv_mode,
               output_dir, seed)                  → pd.DataFrame
"""

import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (BatchNormalization, Bidirectional,
                                     Dense, Dropout, Input, LSTM, Masking)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Local pipeline modules — same imports as the other baseline files
from data_loading import load_data_domain_1, load_data_domain_4
from data_preparation import apply_normalizer, fit_normalizer
from data_splitting import user_dependent_cv, user_independent_cv
from utils_saving import save_results


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    """Pin all random seeds so results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Padding helper
# ─────────────────────────────────────────────────────────────────────────────

def _pad_fold(train_gestures: list, test_gestures: list):
    """
    Pack variable-length trajectories into padded arrays for one fold.

    The padding length is derived from the TRAINING set only (max trajectory
    length in train).  Test sequences longer than that limit are right-truncated
    (rare in practice; trajectories across users rarely differ dramatically).
    Shorter sequences receive post-padding zeros that the Masking layer ignores.

    Returns
    -------
    X_train : float32 array, shape (n_train, max_len, 3)
    y_train : int32   array, shape (n_train,)
    X_test  : float32 array, shape (n_test,  max_len, 3)
    y_test  : int32   array, shape (n_test,)
    max_len : int — padding width (= longest training sequence this fold)
    """
    # Compute max length from training fold only — prevents test-set leakage
    max_len = max(g["trajectory"].shape[0] for g in train_gestures)

    train_seqs = [g["trajectory"] for g in train_gestures]
    test_seqs  = [g["trajectory"] for g in test_gestures]

    # pad_sequences converts the ragged list to a uniform 3-D array
    X_train = pad_sequences(train_seqs, maxlen=max_len,
                            padding="post", truncating="post",
                            dtype="float32", value=0.0)
    X_test  = pad_sequences(test_seqs,  maxlen=max_len,
                            padding="post", truncating="post",
                            dtype="float32", value=0.0)

    y_train = np.array([g["gesture_type"] for g in train_gestures], dtype=np.int32)
    y_test  = np.array([g["gesture_type"] for g in test_gestures],  dtype=np.int32)

    return X_train, y_train, X_test, y_test, max_len


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm_model(input_shape: tuple, n_classes: int) -> Sequential:
    """
    Build and compile the fixed BiLSTM architecture.

    Parameters
    ----------
    input_shape : (max_len, n_dims) — e.g. (T, 3) for 3-D trajectories
    n_classes   : number of gesture categories (10)

    Returns
    -------
    Compiled keras.Sequential model, ready for model.fit()
    """
    model = Sequential([
        # Explicit Input layer avoids the TF 2.16 deprecation warning about
        # passing input_shape directly to a non-trainable layer (Masking).
        Input(shape=input_shape),

        # Masking: zero-padded frames are invisible to the LSTM
        Masking(mask_value=0.0),

        # BiLSTM: fuses forward and backward hidden states so the model reads
        # both the start-to-end and end-to-start temporal patterns
        Bidirectional(LSTM(64, return_sequences=False)),

        # BatchNorm + Dropout: stabilise training and prevent overfitting
        BatchNormalization(),
        Dropout(0.3),

        # Two-layer classifier head
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ], name="BiLSTM_masked")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_bilstm_pipeline(
        gestures:    list,
        cv_mode:     str,
        domain_name: str,
        seed:        int = 42,
):
    """
    Cross-validated BiLSTM experiment using padding + masking.

    Each fold:
      1. Normalise — fit on training split, apply to both (no leakage).
         This is identical to the convention used in data_preparation.py
         and the other baseline methods.
      2. Pad sequences to the longest training trajectory of that fold.
      3. Build the fixed BiLSTM from scratch (prevents cross-fold leakage).
      4. Train with EarlyStopping monitoring val_loss (10% validation split).
      5. Evaluate on the held-out test fold and record per-fold accuracy.

    Parameters
    ----------
    gestures    : list of gesture dicts (standard pipeline format from
                  data_loading.load_data_domain_*)
    cv_mode     : "dependent" or "independent"
    domain_name : str label used in filenames — e.g. "domain1"
    seed        : random seed applied to Python, NumPy, and TensorFlow

    Returns
    -------
    df               : pd.DataFrame — one row per fold, columns:
                         fold_id, method, domain, cv_mode,
                         n_components, n_clusters, compression, k, n_points,
                         accuracy
    global_preds     : dict — {"bilstm_masked": {"y_true": [...], "y_pred": [...]}}
                       mirrors the global_predictions dict returned by
                       run_pipeline() in main.py so the same save_results()
                       call works without modification
    """
    _set_seeds(seed)

    all_types = sorted(set(g["gesture_type"] for g in gestures))
    n_classes = len(all_types)

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    all_results  = []
    global_preds = {"bilstm_masked": {"y_true": [], "y_pred": []}}

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)

        # Per-fold normalisation — identical to the convention in main.py
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        # Pad trajectories (max length derived from training split only)
        X_train, y_train, X_test, y_test, max_len = _pad_fold(
            train_norm, test_norm)

        # One-hot encode labels for categorical_crossentropy
        Y_train = to_categorical(y_train, num_classes=n_classes).astype("float32")

        # Rebuild model from scratch each fold — no cross-fold information leakage
        tf.keras.backend.clear_session()
        _set_seeds(seed)  # re-seed after session reset for reproducibility

        model = build_bilstm_model(
            input_shape=(max_len, X_train.shape[2]),
            n_classes=n_classes,
        )

        # EarlyStopping watches val_loss (not train_loss) so patience reflects
        # generalisation rather than memorisation
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.10,   # 10% of train used for early-stopping monitor
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred   = np.argmax(model.predict(X_test, verbose=0), axis=1)
        accuracy = float(np.mean(y_pred == y_test))

        global_preds["bilstm_masked"]["y_true"].extend(y_test.tolist())
        global_preds["bilstm_masked"]["y_pred"].extend(y_pred.tolist())

        print(f"    accuracy = {accuracy:.4f}")

        # Dummy columns keep the DataFrame compatible with save_results() and
        # the groupby logic used in main.py (columns that don't apply to BiLSTM
        # are set to "N/A" to match the existing CSV schema)
        all_results.append({
            "fold_id":      fold_id,
            "method":       "BiLSTM",
            "domain":       domain_name,
            "cv_mode":      cv_mode,
            "n_components": "N/A",   # BiLSTM does not use PCA
            "n_clusters":   "N/A",   # BiLSTM does not cluster
            "compression":  "N/A",
            "k":            "N/A",
            "n_points":     "N/A",
            "accuracy":     accuracy,
        })

    df  = pd.DataFrame(all_results)
    acc = df["accuracy"].values
    print(f"\n  [BiLSTM] {domain_name} | {cv_mode} — "
          f"mean={acc.mean():.4f}  std={acc.std():.4f}")

    return df, global_preds


# ─────────────────────────────────────────────────────────────────────────────
# Save helper — mirrors the save pattern in main.py exactly
# ─────────────────────────────────────────────────────────────────────────────

def run_and_save(
        domain_name: str,
        gestures:    list,
        cv_mode:     str,
        output_dir:  str = "results",
        seed:        int = 42,
) -> pd.DataFrame:
    """
    Run the full BiLSTM pipeline for one (domain, cv_mode) and persist results.

    Saved files follow the existing pipeline naming convention:
        results/{domain_name}_bilstm_masked_{cv_mode}.txt
        results/{domain_name}_bilstm_masked_{cv_mode}_raw.csv

    The txt file contains the summary accuracy and confusion matrix.
    The raw CSV contains one row per fold with all metadata columns.

    Parameters
    ----------
    domain_name : str  — e.g. "domain1" or "domain4"
    gestures    : list — output of load_data_domain_1 / load_data_domain_4
    cv_mode     : str  — "dependent" or "independent"
    output_dir  : str  — directory for saved results (default: "results")
    seed        : int  — random seed

    Returns
    -------
    df : pd.DataFrame — per-fold accuracy results (same object saved to CSV)
    """
    os.makedirs(output_dir, exist_ok=True)
    config_label = f"{domain_name}_bilstm_masked_{cv_mode}"
    print(f"\nRunning: {config_label}")

    df, global_preds = run_bilstm_pipeline(gestures, cv_mode, domain_name, seed)

    # Build a trivial summary (single config — no hyperparameter sweep)
    # save_results expects a DataFrame indexed by config key with mean/std columns
    df["_config"] = "bilstm_masked"
    summary = df.groupby("_config")["accuracy"].agg(["mean", "std"])
    best_config = "bilstm_masked"

    # Confusion matrix aggregated across all folds
    labels = sorted(set(g["gesture_type"] for g in gestures))
    y_true = global_preds["bilstm_masked"]["y_true"]
    y_pred = global_preds["bilstm_masked"]["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    save_results(summary, best_config, cm, df, config_label, output_dir)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.abspath(__file__))

    datasets = {
        "domain1": load_data_domain_1(
            os.path.join(_BASE, "GestureData",
                         "GestureDataDomain1_Mons", "Domain1_csv")),
        "domain4": load_data_domain_4(
            os.path.join(_BASE, "GestureData",
                         "GestureDataDomain4_Mons")),
    }

    created_files = []

    for domain_name, gestures in datasets.items():
        for cv_mode in ["dependent", "independent"]:
            run_and_save(domain_name, gestures, cv_mode,
                         output_dir="results", seed=42)
            label = f"{domain_name}_bilstm_masked_{cv_mode}"
            created_files.append(f"./results/{label}.txt")
            created_files.append(f"./results/{label}_raw.csv")

    print("\n" + "=" * 60)
    print("Files created:")
    for f in created_files:
        print(f"  {f}")
