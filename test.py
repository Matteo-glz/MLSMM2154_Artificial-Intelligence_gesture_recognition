from sklearn.metrics import confusion_matrix
from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import fit_normalizer, apply_normalizer, fit_pca, apply_pca
from tool_from_scratch import compute_dtw_distance_c_speed
from clustering import fit_kmeans, apply_compression, apply_symbolic_transformation, predict_gesture_type_knn
from collections import Counter
import numpy as np
import pandas as pd
import os
from datetime import datetime


def majority_vote(neighbors):
    labels = [n[1] for n in neighbors]
    return Counter(labels).most_common(1)[0][0]


def run_pipeline(gestures, k_options, pca_options, cluster_options,
                 cv_mode="dependent", method="edit-distance"):
    all_results = []
    global_predictions = {}

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)

        mean, std = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for n_components in pca_options:
            label = n_components if n_components != "no_pca" else "no_pca"

            if n_components != "no_pca":
                pca        = fit_pca(train_norm, n_components=n_components)
                train_proc = apply_pca(train_norm, pca)
                test_proc  = apply_pca(test_norm,  pca)
            else:
                train_proc = train_norm
                test_proc  = test_norm

            # ── EDIT DISTANCE branch ──────────────────────────────────────
            if method == "edit-distance":
                for n_clusters in cluster_options:
                    kmeans     = fit_kmeans(train_proc, n_clusters)
                    train_sym  = apply_compression(
                                     apply_symbolic_transformation(train_proc, kmeans))
                    test_sym   = apply_compression(
                                     apply_symbolic_transformation(test_proc,  kmeans))

                    for k in k_options:
                        y_true, y_pred = [], []
                        config_key = (label, n_clusters, k)

                        if config_key not in global_predictions:
                            global_predictions[config_key] = {"y_true": [], "y_pred": []}

                        for test_g in test_sym:
                            pred = predict_gesture_type_knn(test_g, train_sym, k=k)
                            y_true.append(test_g["gesture_type"])
                            y_pred.append(pred)

                        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                        global_predictions[config_key]["y_true"].extend(y_true)
                        global_predictions[config_key]["y_pred"].extend(y_pred)

                        all_results.append({
                            "fold_id":     fold_id,
                            "n_components": label,
                            "n_clusters":  n_clusters,
                            "k":           k,
                            "accuracy":    accuracy
                        })

            # ── DTW branch ───────────────────────────────────────────────
            else:
                distance_cache = []
                for test_g in test_proc:
                    dists = []
                    for train_g in train_proc:
                        dist = compute_dtw_distance_c_speed(
                            test_g["trajectory"], train_g["trajectory"])
                        dists.append((dist, train_g["gesture_type"]))
                    dists.sort(key=lambda x: x[0])
                    distance_cache.append((test_g["gesture_type"], dists))

                for k in k_options:
                    y_true, y_pred = [], []
                    config_key = (label, k)

                    if config_key not in global_predictions:
                        global_predictions[config_key] = {"y_true": [], "y_pred": []}

                    for true_label, sorted_dists in distance_cache:
                        neighbors = sorted_dists[:k]
                        pred      = majority_vote(neighbors)
                        y_true.append(true_label)
                        y_pred.append(pred)

                    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                    global_predictions[config_key]["y_true"].extend(y_true)
                    global_predictions[config_key]["y_pred"].extend(y_pred)

                    all_results.append({
                        "fold_id":      fold_id,
                        "n_components": label,
                        "n_clusters":   "N/A",
                        "k":            k,
                        "accuracy":     accuracy
                    })

    return pd.DataFrame(all_results), global_predictions


def save_results(summary, best_config, cm, df, config_label, output_dir="results"):
    """Write summary + confusion matrix to a txt file, and raw results to csv."""
    os.makedirs(output_dir, exist_ok=True)
    safe_label = config_label.replace(" ", "_")

    # ── txt report ───────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, f"{safe_label}.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"RESULTS — {config_label}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write("FULL SUMMARY (mean accuracy ± std per config)\n")
        f.write("-"*40 + "\n")
        f.write(summary.to_string())
        f.write("\n\n")

        f.write(f"BEST CONFIG: {best_config}\n")
        best_mean = summary.loc[best_config, "mean"]
        best_std  = summary.loc[best_config, "std"]
        f.write(f"Mean accuracy : {best_mean:.4f}\n")
        f.write(f"Std           : {best_std:.4f}\n\n")

        f.write("CONFUSION MATRIX (best config)\n")
        f.write("-"*40 + "\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n")

    # ── csv raw results ───────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{safe_label}_raw.csv")
    df.to_csv(csv_path, index=False)

    print(f"  -> Saved: {txt_path}")
    print(f"  -> Saved: {csv_path}")


if __name__ == "__main__":
    path_domain_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    path_domain_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    labels = list(range(10))

    datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }

    methods = {
        "edit-distance": {
            "cluster_options": [3],#[5, 7, 9, 11, 13, 15,17,19,21 ],
            "pca_options":      [3],#["no_pca", 1,2, 3],
        },
        "dtw": {
            "cluster_options": [0],        # unused for DTW, kept for uniform signature
            "pca_options":      [3],#["no_pca",1, 2, 3],
        },
    }

    cv_modes  = ["dependent", "independent"]
    k_options = [1]#[1, 3, 5, 7]

    total = len(datasets) * len(methods) * len(cv_modes)
    done  = 0

    for domain_name, gestures in datasets.items():
        for method_name, params in methods.items():
            for cv_mode in cv_modes:

                done += 1
                config_label = f"{domain_name}_{method_name}_{cv_mode}"
                print(f"\n[{done}/{total}] Running: {config_label}")

                df, preds = run_pipeline(
                    gestures,
                    k_options        = k_options,
                    pca_options      = params["pca_options"],
                    cluster_options  = params["cluster_options"],
                    cv_mode          = cv_mode,
                    method           = method_name,
                )

                # groupby columns differ slightly by method
                group_cols = ["n_components", "n_clusters", "k"]
                summary    = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
                best_config = summary["mean"].idxmax()
                print(f"  Best config: {best_config}  "
                      f"mean={summary.loc[best_config,'mean']:.4f}")

                # Confusion matrix for best config
                pca_label  = best_config[0]
                n_clusters = best_config[1]
                k_best     = best_config[2]

                if method_name == "edit-distance":
                    key = (pca_label, n_clusters, k_best)
                else:
                    key = (pca_label, k_best)

                y_true = preds[key]["y_true"]
                y_pred = preds[key]["y_pred"]
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                save_results(summary, best_config, cm, df,
                             config_label, output_dir="results")

    print("\nAll done. Results saved in ./results/")