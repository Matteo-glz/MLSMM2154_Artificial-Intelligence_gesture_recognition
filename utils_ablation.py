"""
ablation_study.py
─────────────────────────────────────────────────────────────────────────────
Étude d'ablation des étapes de préprocessing.

Pour chaque combinaison (méthode × domaine × cv_mode × config_preprocessing),
on fixe le préprocessing et on cherche les meilleurs HPs restants via le
protocole deux phases standard (Phase 1 inner-val, Phase 2 test).

Configs testées:
  (A) normalize=False, pca=no_pca  → baseline brut
  (B) normalize=True,  pca=no_pca  → normalisation seule
  (C) normalize=False, pca=2       → PCA 2D seule
  (D) normalize=True,  pca=2       → norm + PCA 2D  (pipeline complet)
  (E) normalize=False, pca=3       → PCA 3D seule
  (F) normalize=True,  pca=3       → norm + PCA 3D

BiLSTM: PCA non applicable → seulement (A) et (B).

Sorties: ablation_results/ablation_summary.csv  +  tableau imprimé en console.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_algorithms import compute_dtw_distance_c_speed
from utils_assessment import majority_vote

import pipeline_edit_distance as ped
import pipeline_three_cent    as ptc
import pipeline_bilstm        as pbl

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PATH_DOMAIN_1 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain1_Mons/Domain1_csv"
PATH_DOMAIN_4 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain4_Mons"

# ─────────────────────────────────────────────────────────────────────────────
# Grilles de HPs (identiques à main.py)
# ─────────────────────────────────────────────────────────────────────────────

K_OPTIONS             = [1, 3, 5, 7]
CLUSTER_OPTIONS       = [15, 20, 25, 30, 35, 40]
COMPRESSION           = [True, False]
N_POINTS_OPTIONS      = [16, 32, 64]
TARGET_LENGTH_OPTIONS = [16, 32, 64]
N_UNITS_OPTIONS       = [16, 32, 64]

# ─────────────────────────────────────────────────────────────────────────────
# Configs de préprocessing à tester
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSING_CONFIGS = [
    {"normalize": False, "pca": "no_pca"},   # (A) Baseline
    {"normalize": True,  "pca": "no_pca"},   # (B) Normalisation seule
    {"normalize": False, "pca": 2},           # (C) PCA 2D seule
    {"normalize": True,  "pca": 2},           # (D) Norm + PCA 2D  ← standard
    {"normalize": False, "pca": 3},           # (E) PCA 3D seule
    {"normalize": True,  "pca": 3},           # (F) Norm + PCA 3D
]

BILSTM_PREPROCESSING_CONFIGS = [
    {"normalize": False, "pca": "no_pca"},   # (A) Baseline
    {"normalize": True,  "pca": "no_pca"},   # (B) Normalisation seule
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper partagé : applique le préprocessing sans fuite
# ─────────────────────────────────────────────────────────────────────────────

def _apply_preprocessing(train, test, normalize, pca_option):
    """Fit sur train uniquement, applique aux deux ensembles."""
    if normalize:
        mean, std = fit_normalizer(train)
        train = apply_normalizer(train, mean, std)
        test  = apply_normalizer(test,  mean, std)
    if pca_option != "no_pca":
        pca   = fit_pca_per_gesture(train, n_components=pca_option)
        train = apply_pca_per_gesture(train, pca)
        test  = apply_pca_per_gesture(test,  pca)
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# DTW
# ─────────────────────────────────────────────────────────────────────────────

def _run_dtw(gestures, cv_mode, normalize, pca_option, val_fraction=0.20):
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # Phase 1 — sélection de k
    hp_scores = defaultdict(list)
    for train, test, _ in cv_fn(gestures):
        inner_train, inner_val = inner_val_split(train, val_fraction)
        it_proc, iv_proc = _apply_preprocessing(inner_train, inner_val, normalize, pca_option)
        cache = [
            (qg["gesture_type"],
             sorted([(compute_dtw_distance_c_speed(qg["trajectory"], rg["trajectory"]),
                      rg["gesture_type"]) for rg in it_proc], key=lambda x: x[0]))
            for qg in iv_proc
        ]
        for k in K_OPTIONS:
            y_true = [lbl for lbl, _ in cache]
            y_pred = [majority_vote(d[:k]) for _, d in cache]
            hp_scores[k].append(float(np.mean(np.array(y_true) == np.array(y_pred))))

    best_k   = max(hp_scores, key=lambda k: np.mean(hp_scores[k]))
    best_val = float(np.mean(hp_scores[best_k]))

    # Phase 2 — évaluation finale
    fold_accs = []
    for train, test, _ in cv_fn(gestures):
        tr_proc, te_proc = _apply_preprocessing(train, test, normalize, pca_option)
        cache = [
            (qg["gesture_type"],
             sorted([(compute_dtw_distance_c_speed(qg["trajectory"], rg["trajectory"]),
                      rg["gesture_type"]) for rg in tr_proc], key=lambda x: x[0]))
            for qg in te_proc
        ]
        y_true = [lbl for lbl, _ in cache]
        y_pred = [majority_vote(d[:best_k]) for _, d in cache]
        fold_accs.append(float(np.mean(np.array(y_true) == np.array(y_pred))))

    return {"best_k": best_k,
            "val_acc": best_val,
            "mean_test": float(np.mean(fold_accs)),
            "std_test":  float(np.std(fold_accs))}


# ─────────────────────────────────────────────────────────────────────────────
# Edit Distance
# ─────────────────────────────────────────────────────────────────────────────

def _run_edit_distance(gestures, cv_mode, normalize, pca_option, val_fraction=0.20):
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # Phase 1
    hp_scores = defaultdict(list)
    for train, test, _ in cv_fn(gestures):
        inner_train, inner_val = inner_val_split(train, val_fraction)
        it_proc, iv_proc = _apply_preprocessing(inner_train, inner_val, normalize, pca_option)
        for n_clusters in CLUSTER_OPTIONS:
            kmeans = ped.fit_kmeans(it_proc, n_clusters)
            it_sym = ped.apply_compression(ped.apply_symbolic_transformation(it_proc, kmeans))
            iv_sym = ped.apply_compression(ped.apply_symbolic_transformation(iv_proc, kmeans))
            for k in K_OPTIONS:
                for comp in COMPRESSION:
                    y_true = [g["gesture_type"] for g in iv_sym]
                    y_pred = [ped.predict_gesture_type_knn(g, it_sym, k=k, use_clean=comp)
                              for g in iv_sym]
                    val_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
                    hp_scores[(n_clusters, k, comp)].append(val_acc)

    best_hp          = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val         = float(np.mean(hp_scores[best_hp]))
    best_nc, best_k, best_comp = best_hp

    # Phase 2
    fold_accs = []
    for train, test, _ in cv_fn(gestures):
        tr_proc, te_proc = _apply_preprocessing(train, test, normalize, pca_option)
        kmeans  = ped.fit_kmeans(tr_proc, best_nc)
        tr_sym  = ped.apply_compression(ped.apply_symbolic_transformation(tr_proc, kmeans))
        te_sym  = ped.apply_compression(ped.apply_symbolic_transformation(te_proc, kmeans))
        y_true  = [g["gesture_type"] for g in te_sym]
        y_pred  = [ped.predict_gesture_type_knn(g, tr_sym, k=best_k, use_clean=best_comp)
                   for g in te_sym]
        fold_accs.append(float(np.mean(np.array(y_true) == np.array(y_pred))))

    return {"best_n_clusters": best_nc, "best_k": best_k, "best_comp": best_comp,
            "val_acc": best_val,
            "mean_test": float(np.mean(fold_accs)),
            "std_test":  float(np.std(fold_accs))}


# ─────────────────────────────────────────────────────────────────────────────
# Three-Cent
# ─────────────────────────────────────────────────────────────────────────────

def _run_three_cent(gestures, cv_mode, normalize, pca_option, val_fraction=0.20):
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # Phase 1
    hp_scores = defaultdict(list)
    for train, test, _ in cv_fn(gestures):
        inner_train, inner_val = inner_val_split(train, val_fraction)
        it_proc, iv_proc = _apply_preprocessing(inner_train, inner_val, normalize, pca_option)
        for n_points in N_POINTS_OPTIONS:
            templates = ptc.build_templates(it_proc, n_points)
            y_true = [g["gesture_type"] for g in iv_proc]
            y_pred = [ptc.recognize(g["trajectory"], templates, n_points) for g in iv_proc]
            hp_scores[n_points].append(float(np.mean(np.array(y_true) == np.array(y_pred))))

    best_n_points = max(hp_scores, key=lambda n: np.mean(hp_scores[n]))
    best_val      = float(np.mean(hp_scores[best_n_points]))

    # Phase 2
    fold_accs = []
    for train, test, _ in cv_fn(gestures):
        tr_proc, te_proc = _apply_preprocessing(train, test, normalize, pca_option)
        templates = ptc.build_templates(tr_proc, best_n_points)
        y_true    = [g["gesture_type"] for g in te_proc]
        y_pred    = [ptc.recognize(g["trajectory"], templates, best_n_points) for g in te_proc]
        fold_accs.append(float(np.mean(np.array(y_true) == np.array(y_pred))))

    return {"best_n_points": best_n_points,
            "val_acc": best_val,
            "mean_test": float(np.mean(fold_accs)),
            "std_test":  float(np.std(fold_accs))}


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM  (PCA non applicable)
# ─────────────────────────────────────────────────────────────────────────────

def _run_bilstm(gestures, cv_mode, normalize,
                epochs=50, batch_size=16, dropout_rate=0.3, val_fraction=0.20):
    all_types    = sorted(set(g["gesture_type"] for g in gestures))
    n_classes    = len(all_types)
    label_offset = min(all_types)
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # Phase 1
    hp_scores = defaultdict(list)
    for train, test, _ in cv_fn(gestures):
        inner_train, inner_val = inner_val_split(train, val_fraction)
        it_proc, iv_proc = _apply_preprocessing(inner_train, inner_val, normalize, "no_pca")
        for target_length in TARGET_LENGTH_OPTIONS:
            X_it, _, Y_it = pbl._build_tensors(it_proc, target_length, n_classes, label_offset)
            X_iv, y_iv, _ = pbl._build_tensors(iv_proc, target_length, n_classes, label_offset)
            for n_units in N_UNITS_OPTIONS:
                tf.keras.backend.clear_session()
                model = pbl.build_bilstm_model(
                    (target_length, X_it.shape[2]), n_classes, n_units, dropout_rate)
                model.fit(X_it, Y_it, epochs=epochs, batch_size=batch_size,
                          validation_data=(X_iv, Y_iv),
                          callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                                   restore_best_weights=True, verbose=0)],
                          verbose=0)
                val_acc = float(np.mean(
                    np.argmax(model.predict(X_iv, verbose=0), axis=1) == y_iv))
                hp_scores[(target_length, n_units)].append(val_acc)

    best_hp  = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val = float(np.mean(hp_scores[best_hp]))
    best_tl, best_nu = best_hp

    # Phase 2
    fold_accs = []
    for train, test, _ in cv_fn(gestures):
        tr_proc, te_proc = _apply_preprocessing(train, test, normalize, "no_pca")
        X_tr, y_tr, Y_tr = pbl._build_tensors(tr_proc, best_tl, n_classes, label_offset)
        X_te, y_te, _    = pbl._build_tensors(te_proc, best_tl, n_classes, label_offset)
        tf.keras.backend.clear_session()
        model = pbl.build_bilstm_model(
            (best_tl, X_tr.shape[2]), n_classes, best_nu, dropout_rate)
        model.fit(X_tr, Y_tr, epochs=epochs, batch_size=batch_size,
                  validation_split=0.10,
                  callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                           restore_best_weights=True, verbose=0)],
                  verbose=0)
        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1) + label_offset
        y_test = y_te + label_offset
        fold_accs.append(float(np.mean(y_pred == y_test)))

    return {"best_target_length": best_tl, "best_n_units": best_nu,
            "val_acc": best_val,
            "mean_test": float(np.mean(fold_accs)),
            "std_test":  float(np.std(fold_accs))}


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("ablation_results", exist_ok=True)

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }

    METHOD_RUNNERS = {
        "dtw":           (_run_dtw,           PREPROCESSING_CONFIGS),
        "edit_distance": (_run_edit_distance, PREPROCESSING_CONFIGS),
        "three_cent":    (_run_three_cent,    PREPROCESSING_CONFIGS),
        "bilstm":        (_run_bilstm,        BILSTM_PREPROCESSING_CONFIGS),
    }

    CV_MODES = ["dependent", "independent"]

    total = sum(
        len(datasets) * len(CV_MODES) * len(configs)
        for _, configs in METHOD_RUNNERS.values()
    )
    done = 0
    rows = []

    for domain_name, gestures in datasets.items():
        for method_name, (runner, prep_configs) in METHOD_RUNNERS.items():
            for cv_mode in CV_MODES:
                for prep in prep_configs:
                    done += 1
                    normalize  = prep["normalize"]
                    pca_option = prep["pca"]
                    label = (f"[{done}/{total}] {domain_name} | {method_name} | "
                             f"{cv_mode} | norm={normalize} pca={pca_option}")
                    print(f"\n{label}")

                    if method_name == "bilstm":
                        result = runner(gestures, cv_mode, normalize)
                    else:
                        result = runner(gestures, cv_mode, normalize, pca_option)

                    print(f"  val={result['val_acc']:.4f}  "
                          f"test={result['mean_test']:.4f} ± {result['std_test']:.4f}")

                    rows.append({
                        "domain":     domain_name,
                        "method":     method_name,
                        "cv_mode":    cv_mode,
                        "normalize":  normalize,
                        "pca":        str(pca_option),
                        "val_acc":    round(result["val_acc"],    4),
                        "mean_test":  round(result["mean_test"],  4),
                        "std_test":   round(result["std_test"],   4),
                    })

    df = pd.DataFrame(rows)
    out_path = "ablation_results/ablation_summary.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'─'*70}")
    print("RÉSULTATS — mean_test_accuracy par config de préprocessing")
    print('─'*70)
    pivot = df.pivot_table(
        index=["normalize", "pca"],
        columns=["domain", "method", "cv_mode"],
        values="mean_test",
        aggfunc="first",
    )
    print(pivot.to_string())
    print(f"\nSauvegardé → {out_path}")
