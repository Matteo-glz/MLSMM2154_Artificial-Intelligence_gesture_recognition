"""
baseline_transformer.py
─────────────────────────────────────────────────────────────────────────────
Transformer encoder gesture recognizer, rebuilt to match the official
tensor2tensor implementation (Vaswani et al., 2017).
    github.com/tensorflow/tensor2tensor

Changes from a naive Keras Transformer
---------------------------------------
1. Positional encoding  →  tensor2tensor get_timing_signal_1d:
   - Frequencies spaced as  min_scale * exp(-log(max/min) / (d//2 - 1) * i)
   - Sin and cos signals are CONCATENATED (first half = sin, second half = cos),
     not interleaved as in the paper's notation.

2. Pre-normalization  (tensor2tensor default):
   - LayerNorm is applied BEFORE each sub-layer (attention and FFN).
   - A final LayerNorm is applied after the last encoder block.
   - Corresponds to tensor2tensor's layer_preprocess / layer_postprocess
     with hparams.layer_prepostprocess_sequence = "dan".

3. FFN = dense_relu_dense  (tensor2tensor common_layers.dense_relu_dense):
   - Structure: Dense(filter_size, ReLU) → Dropout(relu_dropout) → Dense(d_model)
   - Dropout is between the two Dense layers, not on the sub-layer output.
   - The residual connection additionally applies dropout (layer_postprocess_dropout).

4. filter_size = 4 × d_model  (tensor2tensor hparams.filter_size default ratio).

5. num_layers is now configurable (tensor2tensor num_encoder_layers; default 1).

Public API
----------
    get_timing_signal_1d(seq_len, channels)                → np.ndarray
    build_transformer_model(input_shape, n_classes,
                            d_model, num_layers,
                            ffn_filter_size, dropout_rate) → keras.Model
    run_transformer_pipeline(...)                          → (DataFrame, preds)
    run_transformer_for_dataset(...)                       → None
"""

import math
import time
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
# CLS token — learnable classification token prepended to each sequence
# (Devlin et al., 2019 — BERT)
# ─────────────────────────────────────────────────────────────────────────────

class PrependCLSToken(layers.Layer):
    """Prepends a learnable [CLS] token to the sequence (dim inferred at build)."""

    def build(self, input_shape):
        d_model = int(input_shape[-1])
        self.cls = self.add_weight(
            name="cls_token", shape=(1, 1, d_model),
            initializer="zeros", trainable=True,
        )

    def call(self, x):
        cls = tf.tile(self.cls, [tf.shape(x)[0], 1, 1])
        return tf.concat([cls, x], axis=1)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate schedule — transformer warmup (Vaswani et al., 2017 §5.3)
#   lr = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model: int, warmup_steps: int = 200):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step + 1, tf.float32)   # +1 avoids 0^{-0.5} at step 0
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


# ─────────────────────────────────────────────────────────────────────────────
# Utility: trajectory resampling
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
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utility: dataset preparation
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
# Positional encoding — tensor2tensor get_timing_signal_1d
# ref: tensor2tensor/layers/common_attention.py  line ~408
# ─────────────────────────────────────────────────────────────────────────────

def get_timing_signal_1d(
        seq_len: int,
        channels: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
) -> np.ndarray:
    """
    Sinusoidal positional encoding matching tensor2tensor get_timing_signal_1d.

    Timescales are spaced geometrically:
        log_increment = log(max_timescale / min_timescale) / max(T//2 - 1, 1)
        inv_timescales[i] = min_timescale * exp(-log_increment * i)

    The signal is built by concatenating all sin signals followed by all cos
    signals (NOT interleaved), then added to the input via add_timing_signal_1d.

    Returns shape (1, seq_len, channels) — broadcast-ready for Keras addition.
    """
    num_timescales = channels // 2
    positions = np.arange(seq_len, dtype=np.float32)            # (T,)

    log_timescale_increment = (
        math.log(max_timescale / min_timescale) /
        max(num_timescales - 1, 1)
    )
    inv_timescales = min_timescale * np.exp(
        -log_timescale_increment * np.arange(num_timescales, dtype=np.float32)
    )                                                            # (channels//2,)

    scaled_time = (
        positions[:, np.newaxis] * inv_timescales[np.newaxis, :]
    )                                                            # (T, channels//2)

    # Concatenate: first half = sin, second half = cos  (tensor2tensor style)
    signal = np.concatenate(
        [np.sin(scaled_time), np.cos(scaled_time)], axis=1
    )                                                            # (T, channels)

    if channels % 2 != 0:                                       # pad odd channels
        signal = np.concatenate(
            [signal, np.zeros((seq_len, 1), dtype=np.float32)], axis=1
        )

    return signal[np.newaxis].astype(np.float32)                # (1, T, channels)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition — tensor2tensor Transformer encoder
# ref: tensor2tensor/layers/common_attention.py  (multihead_attention)
#      tensor2tensor/layers/common_layers.py     (dense_relu_dense, layer_norm)
#      tensor2tensor/layers/transformer_layers.py (transformer_encoder)
# ─────────────────────────────────────────────────────────────────────────────

def _n_heads_for(d_model: int) -> int:
    """Largest power-of-2 ≤ 8 that divides d_model evenly."""
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


def build_transformer_model(
        input_shape: tuple,
        n_classes: int,
        d_model: int = 64,
        num_layers: int = 1,
        ffn_filter_size: int = None,
        dropout_rate: float = 0.1,
        warmup_steps: int = 200,
) -> Model:
    """
    Transformer encoder classifier matching tensor2tensor's architecture.

    Per-layer structure (pre-norm, tensor2tensor default)
    ------------------------------------------------------
    x_norm  = LayerNorm(x)                      ← layer_preprocess
    attn    = MultiHeadAttention(x_norm, x_norm) ← attention_dropout inside
    x       = x + Dropout(attn)                 ← layer_postprocess (residual drop)

    x_norm  = LayerNorm(x)                      ← layer_preprocess
    h       = Dense(filter_size, ReLU)(x_norm)  ┐
    h       = Dropout(h)                        ├ dense_relu_dense
    ffn     = Dense(d_model)(h)                 ┘
    x       = x + Dropout(ffn)                  ← layer_postprocess (residual drop)

    After all layers:
    x       = LayerNorm(x)                      ← final layer_preprocess
    → GlobalAvgPool → Dropout → Dense(n_classes, softmax)

    Parameters
    ----------
    input_shape    : (target_length, n_dims)
    n_classes      : number of gesture classes
    d_model        : hidden_size in tensor2tensor
    num_layers     : num_encoder_layers in tensor2tensor  (default 1)
    ffn_filter_size: filter_size in tensor2tensor; defaults to 4 × d_model
    dropout_rate   : applied to attention weights, relu intermediate, and residuals
    """
    if ffn_filter_size is None:
        ffn_filter_size = d_model * 4          # tensor2tensor default ratio

    n_heads = _n_heads_for(d_model)
    key_dim = d_model // n_heads
    seq_len = input_shape[0]

    # Positional encoding covers seq_len+1 positions (CLS + sequence)
    pe = get_timing_signal_1d(seq_len + 1, d_model)
    pe_constant = tf.constant(pe, dtype=tf.float32)

    # ── Input projection ─────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="trajectory")
    x = layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="input_projection")(inputs)
    x = PrependCLSToken(name="cls_token")(x)   # (B, T+1, d_model) — Devlin et al. 2019
    x = x + pe_constant                        # add_timing_signal_1d

    # ── Encoder layers ───────────────────────────────────────────────────────
    for i in range(num_layers):
        p = f"layer_{i}"

        # Self-attention sub-layer
        # layer_preprocess: LayerNorm before attention
        x_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_attn")(x)
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=key_dim,
            dropout=dropout_rate,              # attention_dropout on weights
            name=f"{p}_mha",
        )(x_norm, x_norm)
        # layer_postprocess: residual dropout + skip connection
        attn_out = layers.Dropout(dropout_rate, name=f"{p}_attn_residual_drop")(attn_out)
        x = x + attn_out

        # FFN sub-layer (dense_relu_dense)
        # layer_preprocess: LayerNorm before FFN
        x_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{p}_ln_ffn")(x)
        ffn = layers.Dense(ffn_filter_size, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f"{p}_ffn_1")(x_norm)
        ffn = layers.Dropout(dropout_rate, name=f"{p}_relu_drop")(ffn)   # relu_dropout
        ffn = layers.Dense(d_model,kernel_regularizer=tf.keras.regularizers.l2(1e-4),name=f"{p}_ffn_2")(ffn)
        # layer_postprocess: residual dropout + skip connection
        ffn = layers.Dropout(dropout_rate, name=f"{p}_ffn_residual_drop")(ffn)
        x = x + ffn

    # ── Final LayerNorm (tensor2tensor layer_preprocess at encoder output) ───
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    # ── Classification head — read out CLS token (position 0) ───────────────
    x = layers.Lambda(lambda t: t[:, 0, :], name="cls_readout")(x)
    x = layers.Dropout(dropout_rate, name="head_drop")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="classifier")(x)

    model = Model(inputs, outputs,
                  name=f"T2T_d{d_model}_h{n_heads}_l{num_layers}")
    lr_schedule = TransformerLRSchedule(d_model=d_model, warmup_steps=warmup_steps)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
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
        num_layers_options: list = None,
        cv_mode: str = "dependent",
        epochs: int = 50,
        batch_size: int = 16,
        dropout_rate: float = 0.1,
        validation_split: float = 0.10,
        warmup_steps: int = 200,
):
    """
    Run the full cross-validated Transformer experiment.

    Hyperparameters swept
    ---------------------
    target_length : resampling resolution
    d_model       : embedding / attention dimension (tensor2tensor hidden_size)
    num_layers    : number of encoder blocks        (tensor2tensor num_encoder_layers)

    Returns
    -------
    df                 : pd.DataFrame  — one row per (fold, target_length, d_model, num_layers)
    global_predictions : dict — key = (target_length, d_model, num_layers)
                                value = {"y_true": [...], "y_pred": [...]}
    """
    if target_length_options is None:
        target_length_options = [64]
    if d_model_options is None:
        d_model_options = [64]
    if num_layers_options is None:
        num_layers_options = [1]

    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)

    all_results        = []
    global_predictions = {}

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    experiment_start = time.time()

    for train, test, fold_id in cv_fn(gestures):
        fold_start = time.time()
        print(f"  Fold {fold_id}...", flush=True)

        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for target_length in target_length_options:

            X_train, _, Y_train = _prepare_data(
                train_norm, target_length, n_classes, label_offset)
            X_test,  y_test,  _       = _prepare_data(
                test_norm,  target_length, n_classes, label_offset)

            for d_model in d_model_options:
                for num_layers in num_layers_options:

                    config_key = (target_length, d_model, num_layers)
                    if config_key not in global_predictions:
                        global_predictions[config_key] = {"y_true": [], "y_pred": []}

                    tf.keras.backend.clear_session()
                    model = build_transformer_model(
                        input_shape  = (target_length, X_train.shape[2]),
                        n_classes    = n_classes,
                        d_model      = d_model,
                        num_layers   = num_layers,
                        dropout_rate = dropout_rate,
                        warmup_steps = warmup_steps,
                    )

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

                    y_pred_prob     = model.predict(X_test, verbose=0)
                    y_pred          = np.argmax(y_pred_prob, axis=1)
                    y_pred_original = y_pred + label_offset
                    y_test_original = y_test + label_offset

                    accuracy = float(np.mean(y_pred_original == y_test_original))

                    global_predictions[config_key]["y_true"].extend(y_test_original.tolist())
                    global_predictions[config_key]["y_pred"].extend(y_pred_original.tolist())

                    all_results.append({
                        "fold_id":       fold_id,
                        "n_components":  "N/A",
                        "n_clusters":    "N/A",
                        "compression":   "N/A",
                        "target_length": target_length,
                        "d_model":       d_model,
                        "num_layers":    num_layers,
                        "k":             "N/A",
                        "accuracy":      accuracy,
                    })

                    print(f"    target_length={target_length}, d_model={d_model},"
                          f" num_layers={num_layers}  →  accuracy={accuracy:.4f}")

        fold_elapsed = time.time() - fold_start
        print(f"  Fold {fold_id} done in {fold_elapsed:.1f}s", flush=True)

    total_elapsed = time.time() - experiment_start
    m, s = divmod(int(total_elapsed), 60)
    print(f"\nTotal experiment time: {m}m {s}s", flush=True)

    return pd.DataFrame(all_results), global_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_transformer_for_dataset(domain_name: str, gestures: list,
                                cv_mode: str, output_dir: str = "results"):
    """
    Run the full tensor2tensor-style Transformer sweep for one
    (dataset, cv_mode) combination and save results.
    """
    from utils_saving import save_results

    config_label = f"{domain_name}_transformer_{cv_mode}"
    print(f"\nRunning: {config_label}")

    df, preds = run_transformer_pipeline(
        gestures              = gestures,
        target_length_options = [32, 64],
        d_model_options       = [32, 64],
        num_layers_options    = [1, 2],
        cv_mode               = cv_mode,
        epochs                = 40,
        batch_size            = 32,
        dropout_rate          = 0.3,
        warmup_steps          = 200,
    )

    group_cols  = ["target_length", "d_model", "num_layers"]
    summary     = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
    best_config = summary["mean"].idxmax()

    print(f"  Best config : {best_config}  "
          f"mean={summary.loc[best_config, 'mean']:.4f}  "
          f"std={summary.loc[best_config, 'std']:.4f}")

    labels = sorted(set(g["gesture_type"] for g in gestures))
    y_true = preds[best_config]["y_true"]
    y_pred = preds[best_config]["y_pred"]
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
        for cv_mode in ["independent"]:  # "dependent",
            run_transformer_for_dataset(domain_name, gestures,
                                        cv_mode, output_dir="results")

    print("\nAll done — Transformer results saved in ./results/")
