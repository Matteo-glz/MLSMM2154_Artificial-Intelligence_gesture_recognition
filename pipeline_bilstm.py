"""
baseline_bilstm.py
─────────────────────────────────────────────────────────────────────────────
Bidirectional LSTM gesture recognizer — integrated into the existing pipeline.

Design decisions
----------------
• Trajectories are resampled to a fixed number of points (target_length) so
  that all sequences have the same shape before being fed to the network.
  This replaces the variable-length masking approach, which is unnecessary
  here and adds no value when all inputs are resampled.

• The model is rebuilt and retrained from scratch at every fold to prevent
  any information leakage across folds (same contract as the other methods).

• Normalisation is fitted on the training set of each fold and applied to
  both train and test — consistent with data_preparation.py.

• Hyperparameter selection uses a two-phase protocol to prevent data leakage:
    Phase 1 — HP search: a 20 % inner-validation split is carved out of the
    training portion of each outer fold. All (target_length, n_units)
    combinations are trained on the remaining 80 % and evaluated on the inner
    val. The test user's data is NEVER seen during this phase.
    Phase 2 — Final model: the best HP is then used to retrain a fresh model
    on the FULL outer training set. This final model is evaluated on the held-
    out test user.

• Hyperparameters swept: target_length (resample resolution) and n_units
  (BiLSTM hidden size).

Public API
----------
    resample_trajectory(traj, target_length)  → np.ndarray
    build_bilstm_model(input_shape, n_classes, n_units, dropout_rate)
        → keras.Sequential
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense, Dropout,
                                     BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data_preparation import fit_normalizer, apply_normalizer
from utils_saving import save_results

GROUP_COLS = ["target_length", "n_units"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: trajectory resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a trajectory to a fixed number of time-steps using linear
    interpolation along each spatial dimension independently.

    Parameters
    ----------
    traj          : np.ndarray, shape (n_samples, n_dims) — raw trajectory
    target_length : int — desired number of time-steps

    Returns
    -------
    np.ndarray, shape (target_length, n_dims)
    """
    n, n_dims = traj.shape
    if n == target_length:
        return traj.copy()

    old_indices = np.arange(n)
    new_indices = np.linspace(0, n - 1, target_length)

    return np.stack(
        [np.interp(new_indices, old_indices, traj[:, dim])
         for dim in range(n_dims)],
        axis=1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_bilstm_model(input_shape: tuple, n_classes: int,
                       n_units: int = 64,
                       dropout_rate: float = 0.3) -> Sequential:
    """
    Build and compile a Bidirectional LSTM classifier.

    Architecture
    ------------
    BiLSTM(n_units) → BatchNorm → Dropout → Dense(32, relu) → Dense(n_classes, softmax)

    BatchNormalization after the recurrent layer stabilises training on small
    datasets such as ours (1 000 sequences).

    Parameters
    ----------
    input_shape  : tuple — (target_length, n_dims), e.g. (64, 3)
    n_classes    : int   — number of gesture categories (10)
    n_units      : int   — number of LSTM units per direction
    dropout_rate : float — dropout probability

    Returns
    -------
    Compiled keras.Sequential model
    """
    model = Sequential([
        Bidirectional(LSTM(n_units, return_sequences=False),
                      input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ], name=f"BiLSTM_{n_units}u")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_tensors(gestures: list, target_length: int,
                   n_classes: int, label_offset: int):
    """Convert gesture dicts to (X, y_int, Y_onehot) tensors."""
    X = np.array([resample_trajectory(g["trajectory"], target_length)
                   for g in gestures], dtype=np.float32)
    y = np.array([g["gesture_type"] - label_offset for g in gestures],
                 dtype=np.int32)
    Y = to_categorical(y, num_classes=n_classes).astype(np.float32)
    return X, y, Y


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, target_length_options, n_units_options,
                 cv_mode="dependent", epochs=50, batch_size=16,
                 dropout_rate=0.3, val_fraction=0.20):
    """
    Two-phase cross-validated BiLSTM experiment.

    Phase 1 — HP selection GLOBALE (inner val sur tous les folds)
        Best HP = (target_length, n_units) avec la meilleure moyenne sur tous les folds.

    Phase 2 — Final evaluation avec le best_hp FIXE sur tous les folds.
    """
    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── PHASE 1 : HP selection sur TOUS les folds ────────────────────
    hp_scores = defaultdict(list)

    for train, test, fold_id in cv_fn(gestures):
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_norm, val_fraction)

        for target_length in target_length_options:
            X_it, _, Y_it    = _build_tensors(inner_train, target_length, n_classes, label_offset)
            X_iv, y_iv, Y_iv = _build_tensors(inner_val,   target_length, n_classes, label_offset)

            for n_units in n_units_options:
                tf.keras.backend.clear_session()
                model = build_bilstm_model(
                    input_shape  = (target_length, X_it.shape[2]),
                    n_classes    = n_classes,
                    n_units      = n_units,
                    dropout_rate = dropout_rate,
                )
                model.fit(
                    X_it, Y_it,
                    epochs          = epochs,
                    batch_size      = batch_size,
                    validation_data = (X_iv, Y_iv),
                    callbacks       = [EarlyStopping(monitor="val_loss", patience=5,
                                                     restore_best_weights=True, verbose=0)],
                    verbose         = 0,
                )
                val_acc = float(np.mean(
                    np.argmax(model.predict(X_iv, verbose=0), axis=1) == y_iv))
                hp_scores[(target_length, n_units)].append(val_acc)

    # ← EN DEHORS de la boucle
    best_hp = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val_acc_global = float(np.mean(hp_scores[best_hp]))
    best_target_length, best_n_units = best_hp
    print(f"  Global best HP: target_length={best_target_length}, n_units={best_n_units}")

    # ── PHASE 2 : Evaluation finale avec best_hp FIXE ────────────────
    all_results        = []
    global_predictions = {"y_true": [], "y_pred": []}

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        x_train, _, y_train  = _build_tensors(train_norm, best_target_length, n_classes, label_offset)
        x_test,  y_test,  _  = _build_tensors(test_norm,  best_target_length, n_classes, label_offset)

        tf.keras.backend.clear_session()
        model = build_bilstm_model(
            input_shape  = (best_target_length, x_train.shape[2]),
            n_classes    = n_classes,
            n_units      = best_n_units,
            dropout_rate = dropout_rate,
        )
        model.fit(
            x_train, y_train,
            epochs           = epochs,
            batch_size       = batch_size,
            validation_split = 0.10,
            callbacks        = [EarlyStopping(monitor="val_loss", patience=5,
                                              restore_best_weights=True, verbose=0)],
            verbose          = 0,
        )

        y_pred_original = np.argmax(model.predict(x_test, verbose=0), axis=1) + label_offset
        y_test_original = y_test + label_offset
        
        y_pred_train = np.argmax(model.predict(x_train, verbose=0), axis=1) + label_offset  # ← nouveau
        y_train_orig = y_train + label_offset                                                # ← nouveau
        train_acc    = float(np.mean(y_pred_train == y_train_orig))

        accuracy = float(np.mean(y_pred_original == y_test_original))
        global_predictions["y_true"].extend(y_test_original.tolist())
        global_predictions["y_pred"].extend(y_pred_original.tolist())
        all_results.append({
            "fold_id":       fold_id,
            "target_length": best_target_length,
            "n_units":       best_n_units,
            "train_accuracy": train_acc,
            "val_accuracy":   best_val_acc_global,
            "accuracy":      accuracy,
        })
        print(f"    Test accuracy = {accuracy:.4f}")

    return pd.DataFrame(all_results), global_predictions, best_hp


if __name__ == "__main__":
    PATH_DOMAIN_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    PATH_DOMAIN_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }
    target_length_options = [32, 64, 128]
    n_units_options       = [32, 64, 128]
    cv_modes              = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_bilstm_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds, best_config = run_pipeline(
                gestures, target_length_options, n_units_options, cv_mode
            )

            mean_acc = df["accuracy"].mean()
            std_acc  = df["accuracy"].std()
            print(f"  Best config: {best_config}")
            print(f"  Mean accuracy : {mean_acc:.4f}")
            print(f"  Std           : {std_acc:.4f}")

            y_true = preds["y_true"]
            y_pred = preds["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            summary = df.groupby(["target_length", "n_units"])["accuracy"].agg(["mean", "std"])
            print(f"  Train accuracy : {df['train_accuracy'].mean():.4f}")
            print(f"  Val accuracy   : {df['val_accuracy'].mean():.4f}")
            print(f"  Test accuracy  : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")