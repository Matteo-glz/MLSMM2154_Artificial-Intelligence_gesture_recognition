


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
 
• The function signature and return format of `run_bilstm_pipeline` mirror
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
    run_bilstm_pipeline(gestures, target_length_options, n_units_options,
                        cv_mode, epochs, batch_size)
        → (DataFrame, global_predictions)
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
 
# Local pipeline modules
from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import fit_normalizer, apply_normalizer
 
 
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
 
    resampled = np.stack(
        [np.interp(new_indices, old_indices, traj[:, dim])
         for dim in range(n_dims)],
        axis=1
    )
    return resampled                              # shape: (target_length, n_dims)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Utility: dataset preparation
# ─────────────────────────────────────────────────────────────────────────────
 
def _prepare_data(gestures: list, target_length: int,
                  n_classes: int, label_offset: int = 0):
    """
    Convert a list of gesture dicts into numpy tensors ready for Keras.
 
    Parameters
    ----------
    gestures      : list of gesture dicts (standard pipeline format)
    target_length : int — number of time-steps after resampling
    n_classes     : int — total number of gesture classes
    label_offset  : int — subtracted from gesture_type before one-hot encoding
                    (use 1 if gesture_type starts at 1, 0 if it starts at 0)
 
    Returns
    -------
    X : np.ndarray, shape (n_gestures, target_length, n_dims)  — float32
    y : np.ndarray, shape (n_gestures,)                        — int (raw labels)
    Y : np.ndarray, shape (n_gestures, n_classes)              — one-hot float32
    """
    X, y = [], []
    for g in gestures:
        traj = resample_trajectory(g["trajectory"], target_length)
        X.append(traj)
        y.append(g["gesture_type"] - label_offset)
 
    X = np.array(X, dtype=np.float32)    # (N, T, D)
    y = np.array(y, dtype=np.int32)      # (N,)
    Y = to_categorical(y, num_classes=n_classes).astype(np.float32)
    return X, y, Y
 
 
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
        # Bidirectional LSTM: reads the sequence forwards AND backwards,
        # capturing temporal patterns in both directions.
        Bidirectional(LSTM(n_units, return_sequences=False),
                      input_shape=input_shape),
 
        # BatchNorm: normalises activations across the batch at each step,
        # which accelerates convergence and acts as a mild regulariser.
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
 
def run_bilstm_pipeline(
        gestures: list,
        target_length_options: list = None,
        n_units_options: list = None,
        cv_mode: str = "dependent",
        epochs: int = 50,
        batch_size: int = 16,
        dropout_rate: float = 0.3,
        validation_split: float = 0.10,
):
    """
    Run the full cross-validated BiLSTM experiment.
 
    Mirrors `run_pipeline` from main.py so that the same `save_results` /
    `summary` code works without modification.
 
    Hyperparameters swept
    ---------------------
    target_length : resampling resolution (replaces n_clusters)
    n_units       : BiLSTM hidden size    (replaces k)
 
    Parameters
    ----------
    gestures              : list of gesture dicts (standard pipeline format)
    target_length_options : list of int — e.g. [32, 64, 128]
                            Defaults to [64] if None.
    n_units_options       : list of int — e.g. [32, 64, 128]
                            Defaults to [64] if None.
    cv_mode               : "dependent" or "independent"
    epochs                : maximum number of training epochs
    batch_size            : mini-batch size
    dropout_rate          : dropout probability in the model
    validation_split      : fraction of the training set used for validation
                            (used by EarlyStopping to monitor val_loss)
 
    Returns
    -------
    df                 : pd.DataFrame  — one row per (fold, target_length, n_units)
    global_predictions : dict          — same structure as in run_pipeline
                         key   = (target_length, n_units)
                         value = {"y_true": [...], "y_pred": [...]}
    """
    if target_length_options is None:
        target_length_options = [64]
    if n_units_options is None:
        n_units_options = [64]
 
    # ── Detect gesture_type range to handle 0-indexed vs 1-indexed labels ──
    all_types   = sorted(set(g["gesture_type"] for g in gestures))
    n_classes   = len(all_types)
    label_offset = min(all_types)          # 0 → no shift; 1 → shift by 1
    # After shifting, gesture_type becomes a clean 0-to-(n_classes-1) index
    # that to_categorical can consume directly.
 
    all_results        = []
    global_predictions = {}
 
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv
 
    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
 
        # ── Normalisation — fitted on training fold only ──────────────────
        mean, std   = fit_normalizer(train)
        train_norm  = apply_normalizer(train, mean, std)
        test_norm   = apply_normalizer(test,  mean, std)
 
        # ── Hyperparameter sweep ──────────────────────────────────────────
        for target_length in target_length_options:
 
            # Prepare tensors once per (fold, target_length) — reused across
            # n_units values to avoid redundant resampling.
            X_train, y_train, Y_train = _prepare_data(
                train_norm, target_length, n_classes, label_offset)
            X_test,  y_test,  _       = _prepare_data(
                test_norm,  target_length, n_classes, label_offset)
 
            for n_units in n_units_options:
 
                config_key = (target_length, n_units)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}
 
                # ── Model — rebuilt from scratch each fold ────────────────
                # Guarantees that no information leaks between folds.
                tf.keras.backend.clear_session()
                model = build_bilstm_model(
                    input_shape  = (target_length, X_train.shape[2]),
                    n_classes    = n_classes,
                    n_units      = n_units,
                    dropout_rate = dropout_rate,
                )
 
                # EarlyStopping monitors the *validation* loss, not training
                # loss, so the patience criterion reflects generalisation.
                early_stop = EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 5,
                    restore_best_weights = True,
                    verbose              = 0,
                )
 
                # ── Training ──────────────────────────────────────────────
                model.fit(
                    X_train, Y_train,
                    epochs           = epochs,
                    batch_size       = batch_size,
                    validation_split = validation_split,
                    callbacks        = [early_stop],
                    verbose          = 0,
                )
 
                # ── Inference ─────────────────────────────────────────────
                y_pred_prob = model.predict(X_test, verbose=0)
                y_pred      = np.argmax(y_pred_prob, axis=1)
                # Shift back to original label space for consistency with
                # y_true stored in the gesture dicts.
                y_pred_original = y_pred + label_offset
                y_test_original = y_test + label_offset
 
                # ── Metrics ───────────────────────────────────────────────
                accuracy = float(np.mean(y_pred_original == y_test_original))
 
                global_predictions[config_key]["y_true"].extend(
                    y_test_original.tolist())
                global_predictions[config_key]["y_pred"].extend(
                    y_pred_original.tolist())
 
                all_results.append({
                    "fold_id":       fold_id,
                    # Columns kept for save_results / groupby compatibility
                    "n_components":  "N/A",       # BiLSTM does not use PCA
                    "n_clusters":    "N/A",        # BiLSTM does not cluster
                    "compression":   "N/A",
                    "target_length": target_length,
                    "n_units":       n_units,
                    "k":             "N/A",
                    "accuracy":      accuracy,
                })
 
                print(f"    target_length={target_length}, n_units={n_units}"
                      f"  →  accuracy={accuracy:.4f}")
 
    return pd.DataFrame(all_results), global_predictions
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Integration block — paste into (or call from) your main __main__ section
# ─────────────────────────────────────────────────────────────────────────────
 
def run_bilstm_for_dataset(domain_name: str, gestures: list,
                           cv_mode: str, output_dir: str = "results"):
    """
    Convenience wrapper that runs the full BiLSTM sweep for one
    (dataset, cv_mode) combination and saves the results.
 
    Call this from your __main__ block in the same loop that handles
    edit-distance, DTW, and $1.
 
    Parameters
    ----------
    domain_name : str  — e.g. "domain1" or "domain4"
    gestures    : list — output of load_data_domain_1 / load_data_domain_4
    cv_mode     : str  — "dependent" or "independent"
    output_dir  : str  — folder where results are written (default: "results")
    """
    from utils_saving import save_results
 
    config_label = f"{domain_name}_bilstm_{cv_mode}"
    print(f"\nRunning: {config_label}")
 
    df, preds = run_bilstm_pipeline(
        gestures              = gestures,
        target_length_options = [32, 64, 128],
        n_units_options       = [32, 64, 128],
        cv_mode               = cv_mode,
        epochs                = 50,
        batch_size            = 16,
    )
 
    # ── Summary ──────────────────────────────────────────────────────────
    group_cols  = ["target_length", "n_units"]
    summary     = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
    best_config = summary["mean"].idxmax()
 
    print(f"  Best config : {best_config}  "
          f"mean={summary.loc[best_config, 'mean']:.4f}  "
          f"std={summary.loc[best_config, 'std']:.4f}")
 
    # ── Confusion matrix for best configuration ───────────────────────────
    labels = sorted(set(g["gesture_type"] for g in gestures))
    key    = best_config   # (target_length, n_units)
 
    y_true = preds[key]["y_true"]
    y_pred = preds[key]["y_pred"]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
 
    save_results(summary, best_config, cm, df,
                 config_label, output_dir=output_dir)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution example
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    from data_loading import load_data_domain_1, load_data_domain_4
 
    path_domain_1 = r"c:\Users\basti\OneDrive - UCL\Master 1\Artificial Inteligence\DATASET\GestureDataDomain1_Mons\Domain1_csv"
    path_domain_4 = r"c:\Users\basti\OneDrive - UCL\Master 1\Artificial Inteligence\DATASET\GestureDataDomain4_Mons"
 
    datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }
 
    for domain_name, gestures in datasets.items():
        for cv_mode in ["dependent", "independent"]:
            run_bilstm_for_dataset(domain_name, gestures,
                                   cv_mode, output_dir="results")
 
    print("\nAll done — BiLSTM results saved in ./results/")




'''# Remplacer ceci :
early_stop = EarlyStopping(
    monitor              = "val_loss",
    patience             = 5,
    restore_best_weights = True,
    verbose              = 0,
)
model.fit(
    X_train, Y_train,
    epochs           = epochs,
    batch_size       = batch_size,
    validation_split = validation_split,
    callbacks        = [early_stop],
    verbose          = 0,
)

# Par ceci :
model.fit(
    X_train, Y_train,
    epochs     = epochs,
    batch_size = batch_size,
    verbose    = 0,
)'''