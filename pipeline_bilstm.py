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

• The function signature and return format of `run_pipeline` mirror
  `run_pipeline` exactly so that the same `save_results` / `summary` code
  works without modification.

• A validation split (10 %) is taken from the training set inside each fold
  to allow EarlyStopping to monitor generalisation rather than training loss.

• Hyperparameters swept: target_length (resample resolution) and n_units
  (BiLSTM hidden size). This mirrors the cluster / k sweep of the baseline.

Public API
----------
    resample_trajectory(traj, target_length)  → np.ndarray
    build_bilstm_model(input_shape, n_classes, n_units, dropout_rate)
        → keras.Sequential
"""

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
from data_splitting import user_dependent_cv, user_independent_cv
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
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, target_length_options, n_units_options,
                 cv_mode="dependent", epochs=50, batch_size=16,
                 dropout_rate=0.3, validation_split=0.10):
    all_results        = []
    global_predictions = {}

    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)   # 0-indexed vs 1-indexed labels

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for target_length in target_length_options:
            # Resample once per (fold, target_length) — reused across n_units
            X_train = np.array([resample_trajectory(g["trajectory"], target_length)
                                 for g in train_norm], dtype=np.float32)
            y_train = np.array([g["gesture_type"] - label_offset
                                 for g in train_norm], dtype=np.int32)
            Y_train = to_categorical(y_train, num_classes=n_classes).astype(np.float32)

            X_test  = np.array([resample_trajectory(g["trajectory"], target_length)
                                 for g in test_norm], dtype=np.float32)
            y_test  = np.array([g["gesture_type"] - label_offset
                                 for g in test_norm], dtype=np.int32)

            for n_units in n_units_options:
                config_key = (target_length, n_units)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                # Rebuild from scratch each fold — no information leakage
                tf.keras.backend.clear_session()
                model = build_bilstm_model(
                    input_shape  = (target_length, X_train.shape[2]),
                    n_classes    = n_classes,
                    n_units      = n_units,
                    dropout_rate = dropout_rate,
                )

                model.fit(
                    X_train, Y_train,
                    epochs           = epochs,
                    batch_size       = batch_size,
                    validation_split = validation_split,
                    callbacks        = [EarlyStopping(monitor="val_loss", patience=5,
                                                      restore_best_weights=True, verbose=0)],
                    verbose          = 0,
                )

                y_pred          = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_pred_original = y_pred  + label_offset
                y_test_original = y_test  + label_offset

                accuracy = float(np.mean(y_pred_original == y_test_original))
                global_predictions[config_key]["y_true"].extend(y_test_original.tolist())
                global_predictions[config_key]["y_pred"].extend(y_pred_original.tolist())
                all_results.append({
                    "fold_id":       fold_id,
                    "target_length": target_length,
                    "n_units":       n_units,
                    "accuracy":      accuracy,
                })
                print(f"    target_length={target_length}, n_units={n_units}"
                      f"  →  accuracy={accuracy:.4f}")

    return pd.DataFrame(all_results), global_predictions


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

            df, preds   = run_pipeline(gestures, target_length_options,
                                       n_units_options, cv_mode)
            summary     = df.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std"])
            best_config = summary["mean"].idxmax()
            print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")

            y_true = preds[best_config]["y_true"]
            y_pred = preds[best_config]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
