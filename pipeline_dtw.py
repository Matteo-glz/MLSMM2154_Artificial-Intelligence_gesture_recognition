import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_algorithms import compute_dtw_distance_c_speed
from utils_assessment import majority_vote
from utils_saving import save_results

GROUP_COLS = ["n_components", "k"]


def run_pipeline(gestures, k_options, pca_options, cv_mode="dependent"):
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

            # Compute DTW distances once, then sweep k by slicing the sorted list
            distance_cache = []
            for test_g in test_proc:
                dists = sorted(
                    [(compute_dtw_distance_c_speed(test_g["trajectory"], train_g["trajectory"]),
                      train_g["gesture_type"])
                     for train_g in train_proc],
                    key=lambda x: x[0]
                )
                distance_cache.append((test_g["gesture_type"], dists))

            for k in k_options:
                y_true, y_pred = [], []
                config_key = (pca_label, k)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                for true_label, sorted_dists in distance_cache:
                    pred = majority_vote(sorted_dists[:k])
                    y_true.append(true_label)
                    y_pred.append(pred)

                accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)
                all_results.append({
                    "fold_id":      fold_id,
                    "n_components": pca_label,
                    "k":            k,
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
    k_options   = [1, 3, 5, 7]
    pca_options = ["no_pca", 1, 2, 3]
    cv_modes    = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_dtw_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds   = run_pipeline(gestures, k_options, pca_options, cv_mode)
            summary     = df.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std"])
            best_config = summary["mean"].idxmax()
            print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")

            y_true = preds[best_config]["y_true"]
            y_pred = preds[best_config]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
