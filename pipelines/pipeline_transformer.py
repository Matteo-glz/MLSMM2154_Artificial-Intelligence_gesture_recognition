import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
baseline_transformer.py
─────────────────────────────────────────────────────────────────────────────
Transformer encoder gesture recognizer — integrated into the existing pipeline.

Design decisions
----------------
• Same resampling and data-preparation helpers as baseline_bilstm.py
  (resample_trajectory) — kept local to avoid cross-file coupling.

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
  normalisation) are identical to pipeline_bilstm.py.

Relation to pipeline_bilstm.py
-------------------------------
This file is a drop-in companion to pipeline_bilstm.py. The public API mirrors
run_pipeline exactly so that the same save_results / summary code works
without modification.

Public API
----------
    resample_trajectory(traj, target_length)               → np.ndarray
    positional_encoding(seq_len, d_model)                  → np.ndarray
    build_transformer_model(input_shape, n_classes,
                            d_model, dropout_rate)         → keras.Model
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data.data_loading import load_data_domain_1, load_data_domain_4
from data.data_splitting import user_dependent_cv, user_independent_cv
from data.data_preparation import fit_normalizer, apply_normalizer
from utils.utils_saving import save_results

GROUP_COLS = ["target_length", "d_model"]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: trajectory resampling (identical to pipeline_bilstm.py)
# ─────────────────────────────────────────────────────────────────────────────

def resample_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample a trajectory to a fixed number of time-steps using linear
    interpolation along each spatial dimension independently.
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
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Returns shape (1, seq_len, d_model) — broadcast-ready for Keras addition.
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]

    angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)

    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return angles[np.newaxis, :, :].astype(np.float32)   # (1, T, d_model)


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

    pe_constant = tf.constant(positional_encoding(seq_len, d_model), dtype=tf.float32)

    inputs = layers.Input(shape=input_shape, name="trajectory")
    x = layers.Dense(d_model, name="input_projection")(inputs)
    x = x + pe_constant

    # Self-attention sub-layer
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=key_dim, dropout=dropout_rate, name="mha"
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x + attn_out)

    # Feed-forward sub-layer
    ffn = layers.Dense(d_model * 2, activation="relu", name="ffn_1")(x)
    ffn = layers.Dense(d_model,                         name="ffn_2")(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x + ffn)

    x       = layers.GlobalAveragePooling1D(name="pool")(x)
    x       = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(x)

    model = Model(inputs, outputs, name=f"Transformer_d{d_model}_h{n_heads}")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, target_length_options, d_model_options,
                 cv_mode="dependent", epochs=50, batch_size=16,
                 dropout_rate=0.3, validation_split=0.10):
    all_results        = []
    global_predictions = {}

    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for target_length in target_length_options:
            # Resample once per (fold, target_length) — reused across d_model values
            X_train = np.array([resample_trajectory(g["trajectory"], target_length)
                                 for g in train_norm], dtype=np.float32)
            y_train = np.array([g["gesture_type"] - label_offset
                                 for g in train_norm], dtype=np.int32)
            Y_train = to_categorical(y_train, num_classes=n_classes).astype(np.float32)

            X_test  = np.array([resample_trajectory(g["trajectory"], target_length)
                                 for g in test_norm], dtype=np.float32)
            y_test  = np.array([g["gesture_type"] - label_offset
                                 for g in test_norm], dtype=np.int32)

            for d_model in d_model_options:
                config_key = (target_length, d_model)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                # Rebuild from scratch each fold — no information leakage
                tf.keras.backend.clear_session()
                model = build_transformer_model(
                    input_shape  = (target_length, X_train.shape[2]),
                    n_classes    = n_classes,
                    d_model      = d_model,
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
                y_pred_original = y_pred + label_offset
                y_test_original = y_test + label_offset

                accuracy = float(np.mean(y_pred_original == y_test_original))
                global_predictions[config_key]["y_true"].extend(y_test_original.tolist())
                global_predictions[config_key]["y_pred"].extend(y_pred_original.tolist())
                all_results.append({
                    "fold_id":       fold_id,
                    "target_length": target_length,
                    "d_model":       d_model,
                    "accuracy":      accuracy,
                })
                print(f"    target_length={target_length}, d_model={d_model}"
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
    d_model_options       = [32, 64, 128]
    cv_modes              = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_transformer_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds   = run_pipeline(gestures, target_length_options,
                                       d_model_options, cv_mode)
            summary     = df.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std"])
            best_config = summary["mean"].idxmax()
            print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")

            y_true = preds[best_config]["y_true"]
            y_pred = preds[best_config]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
