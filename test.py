from data_loading import load_data_domain_1
from data_splitting import  user_dependent_cv,user_independent_cv
from data_preparation import fit_normalizer, apply_normalizer, fit_pca, apply_pca
from tool_from_scratch import compute_dtw_distance,compute_dtw_distance_c_speed
from collections import Counter
import numpy as np
import pandas as pd
from tslearn.metrics import dtw as fast_dtw


def majority_vote(neighbors):
    # neighbors est une liste de tuples (distance, label)
    # On extrait juste les labels : [label1, label2, label3]
    labels = [n[1] for n in neighbors]
    # On compte les occurrences et on prend le plus fréquent
    return Counter(labels).most_common(1)[0][0]


path_domain_1 ="/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
loaded_data = load_data_domain_1(path_domain_1)
def run_dtw_knn_pipeline(gestures, k_options, pca_components_options, cv_mode="dependent"):
    """
    cv_mode: "dependent" or "independent"
    pca_components_options: list of int or None values, e.g. [None, 2, 3, 5]
        None = no PCA, just normalization
    """
    all_results = []
    global_predictions = {}

    # Choose the CV strategy
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"\n--- Fold {fold_id} ---")

        # Preprocessing — fit on train only
        mean, std = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        # Loop over PCA options
        for n_components in pca_components_options:
            label = n_components if n_components is not None else "no_pca"  # <-- add this

            if n_components != "no_pca":
                pca = fit_pca(train_norm, n_components=n_components)
                train_proc = apply_pca(train_norm, pca)
                test_proc  = apply_pca(test_norm,  pca)
            else:
                train_proc = train_norm
                test_proc  = test_norm

            # Precompute all DTW distances for this fold+pca config
            # (avoids recomputing for each k)
            distance_cache = []
            for test_g in test_proc:
                dists = []
                for train_g in train_proc:
                    dist = compute_dtw_distance_c_speed(
                        test_g["trajectory"],
                        train_g["trajectory"]
                    )
                    #dist = fast_dtw(test_g["trajectory"], train_g["trajectory"])
                    dists.append((dist, train_g["gesture_type"]))
                dists.sort(key=lambda x: x[0])
                distance_cache.append((test_g["gesture_type"], dists))

            # Loop over k — reuse the cached distances
            for k in k_options:
                y_true, y_pred = [], []
                config_key = (n_components, k)

                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                for true_label, sorted_dists in distance_cache:
                    neighbors = sorted_dists[:k]
                    pred = majority_vote(neighbors)
                    y_true.append(true_label)
                    y_pred.append(pred)

                accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)

                all_results.append({
                    "fold_id": fold_id,
                    "n_components": label,   # <-- use label instead of n_components
                    "k": k,
                    "accuracy": accuracy
                })

    df = pd.DataFrame(all_results)
    return df, global_predictions


# ---- Lancement ------
if __name__ == "__main__":
    loaded_data = load_data_domain_1(path_domain_1)

    k_options   = [1,3,5]
    pca_options = ["no_pca",2,3]  # string instead of None

    df, preds = run_dtw_knn_pipeline(
        loaded_data,
        k_options=k_options,
        pca_components_options=pca_options,
        cv_mode="dependent"
    )

    print(df)
    summary = df.groupby(["n_components", "k"])["accuracy"].agg(["mean", "std"])
    print(summary)