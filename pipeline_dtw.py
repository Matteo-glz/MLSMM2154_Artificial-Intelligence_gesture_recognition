from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_algorithms import compute_dtw_distance_c_speed
from utils_assessment import majority_vote
from utils_saving import save_results

GROUP_COLS = ["n_components", "k"]


def _dtw_distance_cache(ref_gestures, query_gestures):
    """Compute sorted DTW neighbour lists for every query gesture."""
    cache = []
    for qg in query_gestures:
        dists = sorted(
            [(compute_dtw_distance_c_speed(qg["trajectory"], rg["trajectory"]),
              rg["gesture_type"])
             for rg in ref_gestures],
            key=lambda x: x[0]
        )
        cache.append((qg["gesture_type"], dists))
    return cache


def run_pipeline(gestures, k_options, pca_options,
                 cv_mode="dependent", val_fraction=0.20):
    """
    Two-phase cross-validated DTW experiment.

    Phase 1 — HP selection GLOBALE (inner val sur tous les folds)
        Best HP = (n_components, k) avec la meilleure moyenne sur tous les folds.

    Phase 2 — Final evaluation avec le best_hp FIXE sur tous les folds.
    """
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── PHASE 1 : HP selection sur TOUS les folds ────────────────────
    hp_scores = defaultdict(list)

    for train, test, fold_id in cv_fn(gestures):
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_norm, val_fraction)

        for n_components in pca_options:
            if n_components != "no_pca":
                pca     = fit_pca_per_gesture(inner_train, n_components=n_components)
                it_proc = apply_pca_per_gesture(inner_train, pca)
                iv_proc = apply_pca_per_gesture(inner_val,   pca)
            else:
                it_proc = inner_train
                iv_proc = inner_val

            cache_inner = _dtw_distance_cache(it_proc, iv_proc)

            for k in k_options:
                y_true_iv = [lbl for lbl, _ in cache_inner]
                y_pred_iv = [majority_vote(d[:k]) for _, d in cache_inner]
                val_acc   = float(np.mean(np.array(y_true_iv) == np.array(y_pred_iv)))
                hp_scores[(n_components, k)].append(val_acc)

    # ← EN DEHORS de la boucle
    best_hp = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val_acc_global = float(np.mean(hp_scores[best_hp]))
    best_pca, best_k = best_hp
    print(f"  Global best HP: pca={best_pca}, k={best_k}")

    # ── PHASE 2 : Evaluation finale avec best_hp FIXE ────────────────
    all_results = []
    global_predictions = {"y_true": [], "y_pred": []}

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        if best_pca != "no_pca":
            pca        = fit_pca_per_gesture(train_norm, n_components=best_pca)
            train_proc = apply_pca_per_gesture(train_norm, pca)
            test_proc  = apply_pca_per_gesture(test_norm,  pca)
        else:
            train_proc = train_norm
            test_proc  = test_norm

        cache_test = _dtw_distance_cache(train_proc, test_proc)
        cache_train = _dtw_distance_cache(train_proc, train_proc)  # For majority vote on train set

        y_true = [lbl for lbl, _ in cache_test]
        y_pred = [majority_vote(d[:best_k]) for _, d in cache_test]

        y_true_train = [lbl for lbl, _ in cache_train]
        y_pred_train = [majority_vote(d[:best_k]) for _, d in cache_train]
        train_acc = float(np.mean(np.array(y_true_train) == np.array(y_pred_train)))

        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
        global_predictions["y_true"].extend(y_true)
        global_predictions["y_pred"].extend(y_pred)
        all_results.append({
            "fold_id":      fold_id,
            "n_components": best_pca,
            "k":            best_k,
            "train_accuracy": train_acc,
            "val_accuracy":   best_val_acc_global,
            "accuracy":     accuracy,


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
    k_options   = [1, 3, 5, 7]
    pca_options = ["no_pca", 1, 2, 3]
    cv_modes    = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_dtw_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds, best_config = run_pipeline(
                gestures, k_options, pca_options, cv_mode
            )

            mean_acc = df["accuracy"].mean()
            std_acc  = df["accuracy"].std()
            print(f"  Best config: {best_config}")
            print(f"  Mean accuracy : {mean_acc:.4f}")
            print(f"  Std           : {std_acc:.4f}")

            y_true = preds["y_true"]
            y_pred = preds["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            summary = df.groupby(["n_components", "k"])["accuracy"].agg(["mean", "std"])
            print(f"  Train accuracy : {df['train_accuracy'].mean():.4f}")
            print(f"  Val accuracy   : {df['val_accuracy'].mean():.4f}")
            print(f"  Test accuracy  : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")