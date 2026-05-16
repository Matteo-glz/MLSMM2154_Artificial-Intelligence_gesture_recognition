from sklearn.metrics import confusion_matrix

from data.data_loading import load_data_domain_1, load_data_domain_4
from utils.utils_saving import save_results

from pipelines import pipeline_edit_distance
from pipelines import pipeline_dtw
from pipelines import pipeline_three_cent
from pipelines import pipeline_bilstm


PATH_DOMAIN_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
PATH_DOMAIN_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

K_OPTIONS             = [1]#, 3, 5, 7]
PCA_OPTIONS           = ["no_pca"]#, 2, 3]
CLUSTER_OPTIONS       = [15]#, 20, 25, 30, 35, 40]#, 45, 50, 55, 60] # for edit-distance method
COMPRESSION           = [True]#, False] #  for edit-distance method
N_POINTS_OPTIONS      = [16]#, 32, 64], #128, 256] # for three-centroid method
TARGET_LENGTH_OPTIONS = [16]#, 32, 64]#, 128]
N_UNITS_OPTIONS       = [16]#, 32, 64]#, 128]
CV_MODES              = ["dependent", "independent"]


def _save_best(df, preds, group_cols, config_label, labels):
    summary     = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
    best_config = summary["mean"].idxmax()
    print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    save_results(summary, best_config, cm, df, config_label, output_dir="results")


if __name__ == "__main__":
    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        #"domain4": load_data_domain_4(PATH_DOMAIN_4),
    }

    # Each entry: (method name, pipeline runner, group columns for summary)
    method_runners = [
        (
            "edit-distance",
            lambda g, cv: pipeline_edit_distance.run_pipeline(
                g, K_OPTIONS, PCA_OPTIONS, CLUSTER_OPTIONS, COMPRESSION, cv),
            pipeline_edit_distance.GROUP_COLS,
        ),
        (
            "dtw",
            lambda g, cv: pipeline_dtw.run_pipeline(g, K_OPTIONS, PCA_OPTIONS, cv),
            pipeline_dtw.GROUP_COLS,
        ),
        (
            "three-cent",
            lambda g, cv: pipeline_three_cent.run_pipeline(
                g, PCA_OPTIONS, N_POINTS_OPTIONS, cv),
            pipeline_three_cent.GROUP_COLS,
        ),
        (
            "bilstm",
            lambda g, cv: pipeline_bilstm.run_pipeline(
                g, TARGET_LENGTH_OPTIONS, N_UNITS_OPTIONS, cv),
            pipeline_bilstm.GROUP_COLS,
        ),
    ]

    total = len(datasets) * len(method_runners) * len(CV_MODES)
    done  = 0

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for method_name, runner, group_cols in method_runners:
            for cv_mode in CV_MODES:
                done += 1
                config_label = f"{domain_name}_{method_name}_{cv_mode}"
                print(f"\n[{done}/{total}] Running: {config_label}")

                df, preds, _ = runner(gestures, cv_mode)
                _save_best(df, preds, group_cols, config_label, labels)

    print("\nAll done. Results saved in ./results/")
