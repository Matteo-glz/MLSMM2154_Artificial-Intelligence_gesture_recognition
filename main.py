import pandas  as pd
import numpy as np
from sklearn.metrics import confusion_matrix

#load data
from data_loading import load_data_domain_1,load_data_domain_4

# split data
from data_splitting import user_independent_cv,user_dependent_cv

# transform data
from data_preparation import fit_normalizer,apply_normalizer,fit_pca, apply_pca

# cluster data
from clustering import fit_kmeans, apply_symbolic_transformation, apply_compression

# assest data
from clustering import predict_gesture_type_knn
from assesment import summarize_results, compute_global_confusion_matrix

def run_experiment_user_independant(csv_name, gestures, cluster_options, k_options):

    all_results = []
    labels = list(range(0, 10))

    # stockage global pour confusion matrices et prédictions
    global_predictions = {}

    # 🔁 boucle sur nombre de clusters
    for n_clusters in cluster_options:
        print(f"\n--- Testing n_clusters = {n_clusters} ---")

        # 🔁 boucle CV (leave-one-user-out)
        for train, test, subject in user_independent_cv(gestures):
            print(f"Processing Subject {subject}...")

            # --- Préprocessing ---
            mean, std = fit_normalizer(train)
            train_norm = apply_normalizer(train, mean, std)
            test_norm = apply_normalizer(test, mean, std)

            pca = fit_pca(train_norm,3)
            train_pca = apply_pca(train_norm, pca)
            test_pca = apply_pca(test_norm, pca)

            kmeans = fit_kmeans(train_pca, n_clusters=n_clusters)
            train_sym = apply_compression(
                apply_symbolic_transformation(train_pca, kmeans)
            )
            test_sym = apply_compression(
                apply_symbolic_transformation(test_pca, kmeans)
            )

            # 🔁 boucle sur k-NN
            for k in k_options:
                y_true = []
                y_pred = []

                config_key = (n_clusters, k)

                # init stockage global
                if config_key not in global_predictions:
                    global_predictions[config_key] = {
                        "y_true": [],
                        "y_pred": [],
                        "cms": []
                    }

                # 🔁 prédictions
                for test_g in test_sym:
                    pred = predict_gesture_type_knn(test_g, train_sym, k=k)
                    y_true.append(test_g["gesture_type"])
                    y_pred.append(pred)

                # 🔥 accuracy par fold
                accuracy = np.mean(np.array(y_true) == np.array(y_pred))

                # 🔥 confusion matrix du fold
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                # 🔥 stockage global
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)
                global_predictions[config_key]["cms"].append(cm)

                # ✅ UNE ligne par fold
                all_results.append({
                    "fold_id": subject,
                    "n_clusters": n_clusters,
                    "k_neighbors": k,
                    "accuracy_fold": accuracy
                })

    # 📊 DataFrame final
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(csv_name, index=False)

    return df_final, global_predictions


def run_experiment_user_dependant(csv_name, gestures, cluster_options, k_options):

    all_results = []
    labels = list(range(0, 10))

    # stockage global pour confusion matrices et prédictions
    global_predictions = {}

    # 🔁 boucle sur nombre de clusters
    for n_clusters in cluster_options:
        print(f"\n--- Testing n_clusters = {n_clusters} ---")

        # 🔁 boucle CV (leave-one-user-out)
        for train, test, repetition in user_dependent_cv(gestures):
            print(f"Processing Subject {repetition}...")

            # --- Préprocessing ---
            mean, std = fit_normalizer(train)
            train_norm = apply_normalizer(train, mean, std)
            test_norm = apply_normalizer(test, mean, std)

            pca = fit_pca(train_norm,3)
            train_pca = apply_pca(train_norm, pca)
            test_pca = apply_pca(test_norm, pca)

            kmeans = fit_kmeans(train_pca, n_clusters=n_clusters)
            train_sym = apply_compression(
                apply_symbolic_transformation(train_pca, kmeans)
            )
            test_sym = apply_compression(
                apply_symbolic_transformation(test_pca, kmeans)
            )

            # 🔁 boucle sur k-NN
            for k in k_options:
                y_true = []
                y_pred = []

                config_key = (n_clusters, k)

                # init stockage global
                if config_key not in global_predictions:
                    global_predictions[config_key] = {
                        "y_true": [],
                        "y_pred": [],
                        "cms": []
                    }

                # 🔁 prédictions
                for test_g in test_sym:
                    pred = predict_gesture_type_knn(test_g, train_sym, k=k)
                    y_true.append(test_g["gesture_type"])
                    y_pred.append(pred)

                # 🔥 accuracy par fold
                accuracy = np.mean(np.array(y_true) == np.array(y_pred))

                # 🔥 confusion matrix du fold
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                # 🔥 stockage global
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)
                global_predictions[config_key]["cms"].append(cm)

                # ✅ UNE ligne par fold
                all_results.append({
                    "fold_id": repetition,
                    "n_clusters": n_clusters,
                    "k_neighbors": k,
                    "accuracy_fold": accuracy
                })

    # 📊 DataFrame final
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(csv_name, index=False)

    return df_final, global_predictions

if __name__ == "__main__":
    path_domain_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    path_domain_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"
    print("Chargement des données...")

    # Remets ici ton chargement de données
    gestures_dom1 = load_data_domain_1(path_domain_1)
    #gestures_dom4 = load_data_domain_4(path_domain_4) 

    #Document's names
    dom_1_independant = "results_domain_1_user_independant.csv"
    #dom_1_dependant = "results_domain_1_user_dependant"
    #dom_4_independant = "results_domain_4_user_independant"
    #dom_4_dependant = "results_domain_4_user_dependant"

    #parameters 
    clusters_options = [3,5,7]
    k_options=[1,3,5]

    # run the tests

    df_results_dom1_ind,pred_dom1_ind = run_experiment_user_dependant(dom_1_independant,gestures_dom1,clusters_options,k_options)
    summary = summarize_results(df_results_dom1_ind)
    print(summary)

    best_config = summary['mean'].idxmax()
    print("Best config:", best_config)

    labels = list(range(10))
    cm = compute_global_confusion_matrix(pred_dom1_ind, best_config[0], best_config[1], labels)

    print(cm)

