import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from collections import Counter

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_algorithms import edit_distance_fast
from utils_saving import save_results

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
                 compression, cv_mode="dependent"):
    all_results = []
    global_predictions = {}
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for n_components in pca_options:
            pca_label = n_components if n_components != "no_pca" else "no_pca"
            if n_components != "no_pca":
                pca        = fit_pca_per_gesture(train_norm, n_components=n_components)
                train_proc = apply_pca_per_gesture(train_norm, pca)
                test_proc  = apply_pca_per_gesture(test_norm,  pca)
            else:
                train_proc = train_norm
                test_proc  = test_norm

            for n_clusters in cluster_options:
                kmeans    = fit_kmeans(train_proc, n_clusters)
                train_sym = apply_compression(apply_symbolic_transformation(train_proc, kmeans))
                test_sym  = apply_compression(apply_symbolic_transformation(test_proc,  kmeans))

                for k in k_options:
                    for comp in compression:
                        y_true, y_pred = [], []
                        config_key = (pca_label, n_clusters, k, comp)
                        if config_key not in global_predictions:
                            global_predictions[config_key] = {"y_true": [], "y_pred": []}

                        for test_g in test_sym:
                            pred = predict_gesture_type_knn(test_g, train_sym, k=k, use_clean=comp)
                            y_true.append(test_g["gesture_type"])
                            y_pred.append(pred)

                        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                        global_predictions[config_key]["y_true"].extend(y_true)
                        global_predictions[config_key]["y_pred"].extend(y_pred)
                        all_results.append({
                            "fold_id":      fold_id,
                            "n_components": pca_label,
                            "n_clusters":   n_clusters,
                            "k":            k,
                            "compression":  comp,
                            "accuracy":     accuracy,
                        })

    return pd.DataFrame(all_results), global_predictions


if __name__ == "__main__":
    PATH_DOMAIN_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    PATH_DOMAIN_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }
    k_options       = [1, 3, 5, 7]
    pca_options     = ["no_pca", 1, 2, 3]
    cluster_options = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    compression     = [True, False]
    cv_modes        = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_edit-distance_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds   = run_pipeline(gestures, k_options, pca_options,
                                       cluster_options, compression, cv_mode)
            summary     = df.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std"])
            best_config = summary["mean"].idxmax()
            print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")

            y_true = preds[best_config]["y_true"]
            y_pred = preds[best_config]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
