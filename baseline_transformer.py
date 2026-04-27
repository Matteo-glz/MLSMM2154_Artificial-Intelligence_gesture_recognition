"""
baseline_transformer.py
─────────────────────────────────────────────────────────────────────────────
Transformer encoder gesture recognizer — integrated into the existing pipeline.

Design decisions
----------------
• Same resampling and data-preparation helpers as baseline_bilstm.py (resample_trajectory,
  _prepare_data) — kept local to avoid cross-file coupling.

• Architecture:
    Input → Dense(d_model) → + sinusoidal positional encoding
          → [MultiHeadAttention + residual + LayerNorm]
          → [FFN(d_model*2 → d_model) + residual + LayerNorm]
          → GlobalAveragePooling1D → Dropout → Dense(n_classes, softmax)

• One encoder block is chosen deliberately: gesture sequences are short
  (32–128 points), so stacking blocks gives diminishing returns and
  over-fits faster on a small dataset. A single block already gives every
  time-step access to the full sequence context via self-attention.

• Positional encoding is sinusoidal (fixed, Vaswani et al. 2017), not
  learned. On a small dataset learned position embeddings tend to over-fit
  the position distribution of training examples.

• n_heads is derived automatically from d_model: the largest power-of-2
  ≤ 8 that divides d_model evenly, so every head has at least 4 dimensions.

• Hyperparameters swept: target_length (resample resolution, same as BiLSTM)
  and d_model (embedding dimension, analogous to n_units in BiLSTM).

• All training details (EarlyStopping, fold-level model rebuild, per-fold
  normalisation) are identical to baseline_bilstm.py.

Relation to baseline_bilstm.py
-------------------------------
This file is a drop-in companion to baseline_bilstm.py. The public API mirrors
run_bilstm_pipeline / run_bilstm_for_dataset exactly so that the same
save_results / summary code works without modification.

Public API
----------
    resample_trajectory(traj, target_length)               → np.ndarray
    positional_encoding(seq_len, d_model)                  → np.ndarray
    build_transformer_model(input_shape, n_classes,
                            d_model, dropout_rate)         → keras.Model
    run_transformer_pipeline(gestures, target_length_options,
                             d_model_options, cv_mode,
                             epochs, batch_size)           → (DataFrame, global_predictions)
    run_transformer_for_dataset(domain_name, gestures,
                                cv_mode, output_dir)       → None
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import fit_normalizer, apply_normalizer


# ─────────────────────────────────────────────────────────────────────────────
# Utility: trajectory resampling (identical to baseline_bilstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def resample_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
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
# Utility: dataset preparation (identical to baseline_bilstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_data(gestures: list, target_length: int,
                  n_classes: int, label_offset: int = 0):
    X, y = [], []
    for g in gestures:
        traj = resample_trajectory(g["trajectory"], target_length)
        X.append(traj)
        y.append(g["gesture_type"] - label_offset)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    Y = to_categorical(y, num_classes=n_classes).astype(np.float32)
    return X, y, Y


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Returns shape (1, seq_len, d_model) — broadcast-ready for Keras addition.
    """
    positions = np.arange(seq_len)[:, np.newaxis]       # (T, 1)
    dims      = np.arange(d_model)[np.newaxis, :]       # (1, d_model)

    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)

    angles[:, 0::2] = np.sin(angles[:, 0::2])  # even indices → sin
    angles[:, 1::2] = np.cos(angles[:, 1::2])  # odd  indices → cos

    return angles[np.newaxis, :, :].astype(np.float32)  # (1, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def _n_heads_for(d_model: int) -> int:
    """Largest power-of-2 head count that divides d_model evenly and is ≤ 8."""
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


def build_transformer_model(input_shape: tuple, n_classes: int,
                            d_model: int = 64,
                            dropout_rate: float = 0.3) -> Model:
    """
    Build and compile a Transformer encoder classifier.

    Architecture
    ------------
    Input(T, D) → Dense(d_model) → +PosEnc(T, d_model)
               → MHA(n_heads) + residual + LayerNorm
               → FFN(d_model*2 → d_model) + residual + LayerNorm
               → GlobalAvgPool → Dropout → Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape  : (target_length, n_dims) e.g. (64, 3)
    n_classes    : number of gesture classes
    d_model      : embedding / attention dimension
    dropout_rate : dropout probability

    Returns
    -------
    Compiled keras.Model
    """
    n_heads = _n_heads_for(d_model)
    key_dim = d_model // n_heads
    seq_len = input_shape[0]

    pe = positional_encoding(seq_len, d_model)      # (1, T, d_model)
    pe_constant = tf.constant(pe, dtype=tf.float32)

    # ── Input & projection ───────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="trajectory")
    x = layers.Dense(d_model, name="input_projection")(inputs)  # (B, T, d_model)

    # ── Positional encoding ──────────────────────────────────────────────────
    x = x + pe_constant

    # ── Transformer encoder block ────────────────────────────────────────────
    # Self-attention sub-layer
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=key_dim, dropout=dropout_rate,
        name="mha",
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x + attn_out)

    # Feed-forward sub-layer (expand then project back)
    ffn = layers.Dense(d_model * 2, activation="relu", name="ffn_1")(x)
    ffn = layers.Dense(d_model,                         name="ffn_2")(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x + ffn)

    # ── Pooling & classification ─────────────────────────────────────────────
    x = layers.GlobalAveragePooling1D(name="pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(x)

    model = Model(inputs, outputs,
                  name=f"Transformer_d{d_model}_h{n_heads}")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_pipeline(
        gestures: list,
        target_length_options: list = None,
        d_model_options: list = None,
        cv_mode: str = "dependent",
        epochs: int = 50,
        batch_size: int = 16,
        dropout_rate: float = 0.3,
        validation_split: float = 0.10,
):
    """
    Run the full cross-validated Transformer experiment.

    Mirrors run_bilstm_pipeline from baseline_bilstm.py exactly — same return format,
    same cross-validation contract, same normalisation approach.

    Hyperparameters swept
    ---------------------
    target_length : resampling resolution (same role as in BiLSTM)
    d_model       : Transformer embedding dimension (analogous to n_units)

    Parameters
    ----------
    gestures              : list of gesture dicts (standard pipeline format)
    target_length_options : list of int — e.g. [32, 64, 128]. Defaults to [64].
    d_model_options       : list of int — e.g. [32, 64, 128]. Defaults to [64].
    cv_mode               : "dependent" or "independent"
    epochs                : maximum training epochs
    batch_size            : mini-batch size
    dropout_rate          : dropout probability
    validation_split      : fraction of training set used for EarlyStopping

    Returns
    -------
    df                 : pd.DataFrame — one row per (fold, target_length, d_model)
    global_predictions : dict — key   = (target_length, d_model)
                                value = {"y_true": [...], "y_pred": [...]}
    """
    if target_length_options is None:
        target_length_options = [64]
    if d_model_options is None:
        d_model_options = [64]

    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)       # 0 → no shift; 1 → shift labels by 1

    all_results        = []
    global_predictions = {}

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)

        # ── Normalisation — fitted on training fold only ──────────────────
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        # ── Hyperparameter sweep ──────────────────────────────────────────
        for target_length in target_length_options:

            # Prepare tensors once per (fold, target_length) — reused across
            # d_model values to avoid redundant resampling.
            X_train, y_train, Y_train = _prepare_data(
                train_norm, target_length, n_classes, label_offset)
            X_test,  y_test,  _       = _prepare_data(
                test_norm,  target_length, n_classes, label_offset)

            for d_model in d_model_options:

                config_key = (target_length, d_model)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                # ── Model — rebuilt from scratch each fold ────────────────
                tf.keras.backend.clear_session()
                model = build_transformer_model(
                    input_shape  = (target_length, X_train.shape[2]),
                    n_classes    = n_classes,
                    d_model      = d_model,
                    dropout_rate = dropout_rate,
                )

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
                y_pred_prob     = model.predict(X_test, verbose=0)
                y_pred          = np.argmax(y_pred_prob, axis=1)
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
                    "n_components":  "N/A",   # no PCA in Transformer
                    "n_clusters":    "N/A",   # no clustering
                    "compression":   "N/A",
                    "target_length": target_length,
                    "d_model":       d_model,
                    "k":             "N/A",
                    "accuracy":      accuracy,
                })

                print(f"    target_length={target_length}, d_model={d_model}"
                      f"  →  accuracy={accuracy:.4f}")

    return pd.DataFrame(all_results), global_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — mirrors run_bilstm_for_dataset from baseline_bilstm.py
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_for_dataset(domain_name: str, gestures: list,
                                cv_mode: str, output_dir: str = "results"):
    """
    Run the full Transformer sweep for one (dataset, cv_mode) combination
    and save the results. Call this from your __main__ block alongside
    run_bilstm_for_dataset.

    Parameters
    ----------
    domain_name : str  — e.g. "domain1" or "domain4"
    gestures    : list — output of load_data_domain_1 / load_data_domain_4
    cv_mode     : str  — "dependent" or "independent"
    output_dir  : str  — folder where results are written (default: "results")
    """
    from utils_saving import save_results

    config_label = f"{domain_name}_transformer_{cv_mode}"
    print(f"\nRunning: {config_label}")

    df, preds = run_transformer_pipeline(
        gestures              = gestures,
        target_length_options = [32, 64, 128],
        d_model_options       = [32, 64, 128],
        cv_mode               = cv_mode,
        epochs                = 50,
        batch_size            = 16,
    )

    group_cols  = ["target_length", "d_model"]
    summary     = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
    best_config = summary["mean"].idxmax()

    print(f"  Best config : {best_config}  "
          f"mean={summary.loc[best_config, 'mean']:.4f}  "
          f"std={summary.loc[best_config, 'std']:.4f}")

    labels = sorted(set(g["gesture_type"] for g in gestures))
    key    = best_config   # (target_length, d_model)

    y_true = preds[key]["y_true"]
    y_pred = preds[key]["y_pred"]
    cm     = confusion_matrix(y_true, y_pred, labels=labels)

    save_results(summary, best_config, cm, df,
                 config_label, output_dir=output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loading import load_data_domain_1, load_data_domain_4

    path_domain_1 = "GestureData_Mons/GestureDataDomain1_Mons/Domain1_csv"
    path_domain_4 = "GestureData_Mons/GestureDataDomain4_Mons"

    datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }

    for domain_name, gestures in datasets.items():
        for cv_mode in ["dependent", "independent"]:
            run_transformer_for_dataset(domain_name, gestures,
                                        cv_mode, output_dir="results")

    print("\nAll done — Transformer results saved in ./results/")
