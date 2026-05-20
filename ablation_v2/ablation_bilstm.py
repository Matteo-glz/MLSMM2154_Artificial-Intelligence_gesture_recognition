"""
ablation_bilstm.py
─────────────────────────────────────────────────────────────────────────────
Ablation study for Bidirectional LSTM gesture recognition.

Hyperparameters swept:
  - seq_len    : [64, 128]                    — fixed resampled sequence length
  - features   : ['raw', 'velocity', 'combined'] — input feature engineering
  - n_units    : [64, 128, 256]              — BiLSTM hidden units per direction
  - n_layers   : [1, 2, 3]                   — stacked BiLSTM layers
  - dropout    : [0.0, 0.2, 0.3, 0.5]        — dropout rate between layers
  - optimizer  : ['adam', 'adamw', 'sgd']    — training optimizer
  - lr         : [1e-3, 5e-4, 1e-4]          — learning rate
  - batch_size : [16, 32, 64]                — mini-batch size

Protocol: 2-phase cross-validation identical to main pipeline.
  Phase 1 — inner-val HP selection.
  Phase 2 — final model trained on full outer training set.

Outputs: ablation_v2/ablation_v2_results/bilstm_ablation.csv

WARNING: The full HP grid (2×3×3×3×4×3×3×3 = 5832 combos) is impractical.
         Start with a focused subset: comment out the lines marked [OPTIONAL]
         in each option list to run a fast baseline sweep first.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data.data_loading import load_data_domain_1, load_data_domain_4
from data.data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data.data_preparation import fit_normalizer, apply_normalizer

# ─────────────────────────────────────────────────────────────────────────────
# Paths & output
# ─────────────────────────────────────────────────────────────────────────────

PATH_DOMAIN_1 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain1_Mons/Domain1_csv"
PATH_DOMAIN_4 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain4_Mons"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_v2_results")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid  — comment out [OPTIONAL] rows to reduce runtime
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN_OPTIONS  = [64, 128]               # [OPTIONAL] remove 128 for speed
FEATURE_OPTIONS  = ["raw", "velocity", "combined"]
N_UNITS_OPTIONS  = [64, 128, 256]          # [OPTIONAL] remove 256 for speed
N_LAYERS_OPTIONS = [1, 2, 3]              # [OPTIONAL] remove 3 for speed
DROPOUT_OPTIONS  = [0.0, 0.2, 0.3, 0.5]  # [OPTIONAL] keep [0.2, 0.3] for speed
OPTIMIZER_OPTIONS= ["adam", "adamw", "sgd"]
LR_OPTIONS       = [1e-3, 5e-4, 1e-4]    # [OPTIONAL] keep [1e-3] for speed
BATCH_OPTIONS    = [16, 32, 64]           # [OPTIONAL] keep [32] for speed
CV_MODES         = ["dependent", "independent"]

EPOCHS       = 50
PATIENCE     = 5

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _resample_linear(traj: np.ndarray, n: int) -> np.ndarray:
    old_n = len(traj)
    if old_n == n:
        return traj.astype(np.float32)
    old_idx = np.arange(old_n, dtype=np.float64)
    new_idx = np.linspace(0, old_n - 1, n)
    return np.stack([np.interp(new_idx, old_idx, traj[:, d])
                     for d in range(traj.shape[1])], axis=1).astype(np.float32)


def _extract_features(traj: np.ndarray, seq_len: int, feature_type: str) -> np.ndarray:
    """
    Return a (seq_len, n_dims) feature array.

    'raw'      → (seq_len, 3)  — resampled x,y,z
    'velocity' → (seq_len, 3)  — 1st-order diff, resampled back to seq_len
    'combined' → (seq_len, 6)  — raw + velocity concatenated
    """
    raw = _resample_linear(traj, seq_len)           # (seq_len, 3)

    if feature_type == "raw":
        return raw

    # Velocity: diff of raw (seq_len-1 frames), resampled back to seq_len
    vel_short = np.diff(raw, axis=0).astype(np.float32)  # (seq_len-1, 3)
    vel = _resample_linear(vel_short, seq_len)            # (seq_len, 3)

    if feature_type == "velocity":
        return vel
    elif feature_type == "combined":
        return np.concatenate([raw, vel], axis=1)         # (seq_len, 6)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def _build_tensors(gestures: list, seq_len: int, feature_type: str,
                   n_classes: int, label_offset: int):
    """Convert gesture list → (X, y_int, Y_onehot)."""
    X = np.array([_extract_features(g["trajectory"], seq_len, feature_type)
                  for g in gestures], dtype=np.float32)
    y = np.array([g["gesture_type"] - label_offset for g in gestures],
                 dtype=np.int32)
    Y = to_categorical(y, num_classes=n_classes).astype(np.float32)
    return X, y, Y


# ─────────────────────────────────────────────────────────────────────────────
# Model builder: stacked BiLSTM with configurable depth
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer(name: str, lr: float):
    """Return a compiled Keras optimizer by name."""
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "adamw":
        try:
            # TF >= 2.11
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
        except AttributeError:
            try:
                # TF 2.10 experimental
                return tf.keras.optimizers.experimental.AdamW(
                    learning_rate=lr, weight_decay=1e-4)
            except AttributeError:
                # Fallback: Adam (no native AdamW available)
                return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_bilstm(input_shape: tuple, n_classes: int,
                 n_units: int, n_layers: int, dropout: float,
                 optimizer_name: str, lr: float) -> tf.keras.Model:
    """
    Stacked Bidirectional LSTM classifier.

    Layer structure for n_layers=1:
        BiLSTM(n_units) → BN → Dropout → Dense(32) → Dense(n_classes)

    For n_layers>1, intermediate BiLSTM layers use return_sequences=True and
    halve the hidden size to keep parameter count manageable.
    """
    model = Sequential(name=f"BiLSTM_{n_layers}L_{n_units}u")

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        units = n_units if i == 0 else max(16, n_units // (2 ** i))

        if i == 0:
            model.add(Bidirectional(LSTM(units, return_sequences=return_seq),
                                    input_shape=input_shape))
        else:
            model.add(Bidirectional(LSTM(units, return_sequences=return_seq)))

        model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=_make_optimizer(optimizer_name, lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Two-phase cross-validated experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(gestures: list, cv_mode: str, val_fraction: float = 0.20):
    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── Phase 1: HP selection on inner validation ─────────────────────────────
    hp_scores = defaultdict(list)

    for train, test, _ in cv_fn(gestures):
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_n, val_fraction)

        for seq_len in SEQ_LEN_OPTIONS:
            for feature_type in FEATURE_OPTIONS:
                X_it, _, Y_it    = _build_tensors(inner_train, seq_len, feature_type,
                                                   n_classes, label_offset)
                X_iv, y_iv, Y_iv = _build_tensors(inner_val,   seq_len, feature_type,
                                                   n_classes, label_offset)
                n_dims = X_it.shape[2]

                for n_units in N_UNITS_OPTIONS:
                    for n_layers in N_LAYERS_OPTIONS:
                        for dropout in DROPOUT_OPTIONS:
                            for opt_name in OPTIMIZER_OPTIONS:
                                for lr in LR_OPTIONS:
                                    for batch_size in BATCH_OPTIONS:
                                        tf.keras.backend.clear_session()
                                        model = build_bilstm(
                                            (seq_len, n_dims), n_classes,
                                            n_units, n_layers, dropout,
                                            opt_name, lr,
                                        )
                                        model.fit(
                                            X_it, Y_it,
                                            epochs=EPOCHS,
                                            batch_size=batch_size,
                                            validation_data=(X_iv, Y_iv),
                                            callbacks=[EarlyStopping(
                                                monitor="val_loss",
                                                patience=PATIENCE,
                                                restore_best_weights=True,
                                                verbose=0,
                                            )],
                                            verbose=0,
                                        )
                                        val_acc = float(np.mean(
                                            np.argmax(
                                                model(X_iv, training=False).numpy(),
                                                axis=1,
                                            ) == y_iv
                                        ))
                                        hp = (seq_len, feature_type, n_units,
                                              n_layers, dropout, opt_name, lr,
                                              batch_size)
                                        hp_scores[hp].append(val_acc)

    best_hp  = max(hp_scores, key=lambda h: np.mean(hp_scores[h]))
    best_val = float(np.mean(hp_scores[best_hp]))
    sl, ft, nu, nl, dr, op, lr_b, bs = best_hp
    print(f"  Best HP → seq_len={sl}, features={ft}, n_units={nu}, n_layers={nl}, "
          f"dropout={dr}, optimizer={op}, lr={lr_b}, batch={bs}   "
          f"val_acc={best_val:.4f}")

    # ── Phase 2: final evaluation with fixed best HP ──────────────────────────
    rows = []
    for train, test, fold_id in cv_fn(gestures):
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        test_n    = apply_normalizer(test,  mean, std)

        X_tr, _, Y_tr   = _build_tensors(train_n, sl, ft, n_classes, label_offset)
        X_te, y_te, _   = _build_tensors(test_n,  sl, ft, n_classes, label_offset)
        n_dims = X_tr.shape[2]

        tf.keras.backend.clear_session()
        model = build_bilstm(
            (sl, n_dims), n_classes, nu, nl, dr, op, lr_b,
        )
        model.fit(
            X_tr, Y_tr,
            epochs=EPOCHS,
            batch_size=bs,
            validation_split=0.10,
            callbacks=[EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )],
            verbose=0,
        )
        y_pred = np.argmax(model(X_te, training=False).numpy(), axis=1) + label_offset
        y_test = y_te + label_offset
        acc    = float(np.mean(y_pred == y_test))

        rows.append({
            "fold_id":    fold_id,
            "seq_len":    sl,
            "features":   ft,
            "n_units":    nu,
            "n_layers":   nl,
            "dropout":    dr,
            "optimizer":  op,
            "lr":         lr_b,
            "batch_size": bs,
            "val_acc":    round(best_val, 4),
            "test_acc":   round(acc, 4),
        })
        print(f"    Fold {fold_id}: test_acc={acc:.4f}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }

    all_rows = []
    total = len(datasets) * len(CV_MODES)
    done  = 0

    for domain_name, gestures in datasets.items():
        for cv_mode in CV_MODES:
            done += 1
            print(f"\n[{done}/{total}] BiLSTM ablation | {domain_name} | {cv_mode}")
            print("─" * 60)

            rows = run_ablation(gestures, cv_mode)
            for r in rows:
                r["domain"]  = domain_name
                r["cv_mode"] = cv_mode
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "bilstm_ablation.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'─' * 70}")
    print("SUMMARY — mean test accuracy per HP config")
    print("─" * 70)
    summary = (
        df.groupby(["domain", "cv_mode", "seq_len", "features", "n_units",
                    "n_layers", "dropout", "optimizer", "lr", "batch_size"])["test_acc"]
        .agg(["mean", "std"])
    )
    print(summary.to_string())
    print(f"\nSaved → {out_path}")
