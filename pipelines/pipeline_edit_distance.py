import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from collections import Counter, defaultdict

from data.data_loading import load_data_domain_1, load_data_domain_4
from data.data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data.data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils.utils_algorithms import edit_distance_fast
from utils.utils_saving import save_results

GROUP_COLS = ["n_components", "n_clusters", "k", "compression"]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(train_gestures, n_clusters=10):
    '''
    Learn the centroïds (the alphabet) only on the training set.
    '''
    all_points = np.vstack([g['trajectory'] for g in train_gestures])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(all_points)
    return kmeans


def apply_symbolic_transformation(gestures, kmeans):
    '''
    Trajectory transformation: from raw trajectories to "raw sequences" (ex: "AAAAABBBCCCC").
    '''
    raw_gestures = []
    for g in gestures:
        g_copy = g.copy()
        clusters = kmeans.predict(g['trajectory'])
        g_copy['seq_raw'] = "".join([chr(65 + c) for c in clusters])
        raw_gestures.append(g_copy)
    return raw_gestures


def apply_compression(gestures):
    '''
    Compress the raw sequences by removing consecutive duplicate characters.
    '''
    compressed_gestures = []
    for g in gestures:
        g_copy = g.copy()
        raw    = g['seq_raw']
        if not raw:
            g_copy['seq_clean'] = ""
        else:
            clean = [raw[0]]
            for char in raw[1:]:
                if char != clean[-1]:
                    clean.append(char)
            g_copy['seq_clean'] = "".join(clean)
        compressed_gestures.append(g_copy)
    return compressed_gestures


def predict_gesture_type_knn(test_gesture, train_gestures, k=3, use_clean=True):
    '''
    Predict the gesture type using kNN on the edit distance between sequences.
     - use_clean=True  → compressed sequences (ABC)
     - use_clean=False → raw sequences (AAABBBCCC)
    '''
    column     = 'seq_clean' if use_clean else 'seq_raw'
    target_seq = test_gesture[column]

    distances = [
        {"dist": edit_distance_fast(target_seq, g[column]),
         "gesture_type": g['gesture_type']}
        for g in train_gestures
    ]

    k_neighbors    = sorted(distances, key=lambda x: x['dist'])[:k]
    neighbor_types = [n['gesture_type'] for n in k_neighbors]
    return Counter(neighbor_types).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, k_options, pca_options, cluster_options,
                 compression, cv_mode="dependent", val_fraction=0.20):
    """
    Two-phase cross-validated edit-distance experiment.

    Phase 1 — HP selection GLOBALE (inner val sur tous les folds)
        Best HP = (n_components, n_clusters, k, compression) avec la meilleure moyenne.

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

            for n_clusters in cluster_options:
                kmeans = fit_kmeans(it_proc, n_clusters)
                it_sym = apply_compression(apply_symbolic_transformation(it_proc, kmeans))
                iv_sym = apply_compression(apply_symbolic_transformation(iv_proc, kmeans))

                for k in k_options:
                    for comp in compression:
                        y_true_iv, y_pred_iv = [], []
                        for g in iv_sym:
                            pred = predict_gesture_type_knn(g, it_sym, k=k, use_clean=comp)
                            y_true_iv.append(g["gesture_type"])
                            y_pred_iv.append(pred)

                        val_acc = float(np.mean(np.array(y_true_iv) == np.array(y_pred_iv)))
                        hp_scores[(n_components, n_clusters, k, comp)].append(val_acc)

    # ← EN DEHORS de la boucle
    best_hp = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val_acc_global = float(np.mean(hp_scores[best_hp]))
    best_pca, best_nc, best_k, best_comp = best_hp
    print(f"  Global best HP: pca={best_pca}, clusters={best_nc}, "
          f"k={best_k}, comp={best_comp}")

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

        kmeans    = fit_kmeans(train_proc, best_nc)
        train_sym = apply_compression(apply_symbolic_transformation(train_proc, kmeans))
        test_sym  = apply_compression(apply_symbolic_transformation(test_proc,  kmeans))

        y_true, y_pred = [], []
        for test_g in test_sym:
            pred = predict_gesture_type_knn(test_g, train_sym, k=best_k, use_clean=best_comp)
            y_true.append(test_g["gesture_type"])
            y_pred.append(pred)

        y_true_train, y_pred_train = [], []
        for g in train_sym:                                                          # ← nouveau
            pred = predict_gesture_type_knn(g, train_sym, k=best_k, use_clean=best_comp)
            y_true_train.append(g["gesture_type"])
            y_pred_train.append(pred)
        train_acc = float(np.mean(np.array(y_true_train) == np.array(y_pred_train)))# ← nouveau


        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
        global_predictions["y_true"].extend(y_true)
        global_predictions["y_pred"].extend(y_pred)

        
        all_results.append({
            "fold_id":      fold_id,
            "n_components": best_pca,
            "n_clusters":   best_nc,
            "k":            best_k,
            "compression":  best_comp,
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
    k_options       = [1, 3, 5, 7]
    pca_options     = ["no_pca", 2, 3]
    cluster_options = [15, 20, 25, 30, 35, 40]#, 45, 50, 55, 60]
    compression     = [True, False]
    cv_modes        = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_edit-distance_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds, best_config = run_pipeline(
                gestures, k_options, pca_options,
                cluster_options, compression, cv_mode
            )

            mean_acc = df["accuracy"].mean()
            std_acc  = df["accuracy"].std()
            print(f"  Best config: {best_config}")
            print(f"  Mean accuracy : {mean_acc:.4f}")
            print(f"  Std           : {std_acc:.4f}")

            y_true = preds["y_true"]
            y_pred = preds["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            summary = df.groupby(["n_components", "n_clusters", "k", "compression"])["accuracy"].agg(["mean", "std"])
            print(f"  Train accuracy : {df['train_accuracy'].mean():.4f}")
            print(f"  Val accuracy   : {df['val_accuracy'].mean():.4f}")
            print(f"  Test accuracy  : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
