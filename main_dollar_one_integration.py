"""
main_dollar_one_integration.py
──────────────────────────────
Drop-in addition to your existing __main__ block.
Paste this AFTER your baseline loop (edit-distance / DTW).
"""

from dollar_one import run_pipeline_dollar_one
from sklearn.metrics import confusion_matrix
from saving_result import save_results
from data_loading import load_data_domain_1, load_data_domain_4

# ── Hyper-parameter grid for $1 ───────────────────────────────────────────────
#   n_points replaces n_clusters.  Paper says N=64 is adequate; 32–256 is safe.
#   PCA options stay the same as for the baselines.
path_domain_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
path_domain_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"


    # On which data we work 
datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }

dollar_one_params = {
    "n_points_options": [32, 64, 128, 256],
    "pca_options":      ["no_pca", 1, 2, 3],
}

labels    = list(range(10))
cv_modes  = ["dependent", "independent"]

for domain_name, gestures in datasets.items():            # datasets defined above
    for cv_mode in cv_modes:

        config_label = f"{domain_name}_dollar_one_{cv_mode}"
        print(f"\nRunning: {config_label}")

        df, preds = run_pipeline_dollar_one(
            gestures          = gestures,
            n_points_options  = dollar_one_params["n_points_options"],
            pca_options       = dollar_one_params["pca_options"],
            cv_mode           = cv_mode,
        )

        # ── Summary (mirrors your baseline summary block) ─────────────────
        group_cols  = ["n_components", "n_points"]
        summary     = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
        best_config = summary["mean"].idxmax()

        print(f"  Best config: {best_config}  "
              f"mean={summary.loc[best_config, 'mean']:.4f}  "
              f"std={summary.loc[best_config, 'std']:.4f}")

        # ── Confusion matrix ──────────────────────────────────────────────
        pca_label = best_config[0]
        n_points  = best_config[1]
        key       = (pca_label, n_points)

        y_true = preds[key]["y_true"]
        y_pred = preds[key]["y_pred"]
        cm     = confusion_matrix(y_true, y_pred, labels=labels)

        save_results(summary, best_config, cm, df, config_label,
                     output_dir="results")

print("\nAll done — $1 results saved in ./results/")

