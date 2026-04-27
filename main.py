# tool 
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# 1. Data Loading 
from data_loading import load_data_domain_1, load_data_domain_4

# 2. data splitting 

from data_splitting import user_dependent_cv, user_independent_cv

# 3. data preparation 
from data_preparation import fit_normalizer, apply_normalizer, fit_pca_per_gesture, apply_pca_per_gesture

# Extra : baseline methods from scratch 
from utils_algorithms import compute_dtw_distance_c_speed

# 4. Models
from baseline_edit_distance import fit_kmeans, apply_compression, apply_symbolic_transformation, predict_gesture_type_knn
from baseline_three_cent import build_templates, recognize

# 5. Assesment
from utils_assessment import majority_vote

# 6. export results
from utils_saving import save_results


def run_pipeline(gestures, k_options, pca_options, cluster_options=0, compression=False,n_points_options= 0,
                 cv_mode="dependent", method="edit-distance"):
    
    all_results = []
    global_predictions = {}

    # ------ Splitting ----------
    # Split the dataset into two sets. One train set and one test-set regarding if we are in user dependant or independant.
    # The train and test set will change for each fold allowing us to process a "cross-fold validation" 
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ------ Normalization --------
    # run the experimentation on each fold (10 in total)
    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)

        mean, std = fit_normalizer(train)                   # Extract the mean and the std from the train test (different for each fold)
        train_norm = apply_normalizer(train, mean, std)     # Apply the noramization on the train-set with the parameters
        test_norm  = apply_normalizer(test,  mean, std)     # Apply the normalization on the test-set with the parameters 

    # ------- PCA -----------
    # For each fold, we will use (or not) a PCA. the number of the components is given in a dictonnary 
    # All of them will be tested. So we loop on the dictonnary to test with different number of components
        for n_components in pca_options:
            pca_label = n_components if n_components != "no_pca" else "no_pca"           # if no PCA we only apply a normalization (labels allows us to keep track of where are we)

            if n_components != "no_pca":
                pca        = fit_pca_per_gesture(train_norm, n_components=n_components)     # Extract the direction vector based on train test  (per each gesture per fold)
                train_proc = apply_pca_per_gesture(train_norm, pca)                         # Apply the PCA on the train-set with the parameters (per gesture)
                test_proc  = apply_pca_per_gesture(test_norm,  pca)                         # Apply the PCA on the test-set with the parameters
            else:
                train_proc = train_norm
                test_proc  = test_norm

            # ── EDIT DISTANCE branch ──────────────────────────────────────
            if method == "edit-distance":
            # ------ Clustering (K-means) -------
            # We run the experimenation several time to fin the number of clusters which provide the best results
                for n_clusters in cluster_options:
                    kmeans     = fit_kmeans(train_proc, n_clusters)                         # Extract the centroïds based on the train-set
                    train_sym  = apply_compression(                                        # We apply a symbolic transformation, we give a letter to each cluster the output is a suite of : AAAABBBBCCCC
                                     apply_symbolic_transformation(train_proc, kmeans))    # Compress the suite to have something like : ABC 
                    test_sym   = apply_compression(                                        # We apply that on the train and the test-set
                                     apply_symbolic_transformation(test_proc,  kmeans))
            #------- KNN -------------
            # In the same way to assigne a gesture to a test gesture, we needed to test several number of nearest neighbors 
                    for k in k_options: 
                        for comp in compression : 
                            y_true, y_pred = [], []                         # the real gesture and the predicted gesture. 
                            config_key = (pca_label, n_clusters, k, comp)             # the config_key save the final configuration for each fold (will be extract later)

                            if config_key not in global_predictions:        # prevent us to test twice the same configuration
                                global_predictions[config_key] = {"y_true": [], "y_pred": []}   #keep track for each tested gesture in the given configuration of its real value and its predicted value

                            for test_g in test_sym:   # For each gesture in the test-test we will                                        
                                pred = predict_gesture_type_knn(test_g, train_sym, k=k, use_clean= comp)         # Compute its prediccted value with the KNN (using edit-distance)
                                y_true.append(test_g["gesture_type"])                           # Append the list of real gesture type with its real gesture type
                                y_pred.append(pred)                                             # Append the list of predicted gesture type with its predicted gesture type

                            accuracy = np.mean(np.array(y_true) == np.array(y_pred))            # Compute the accuracy 
                            global_predictions[config_key]["y_true"].extend(y_true)             # Aggregate fold predictions into a global dictionary for cross-validation analysis
                            global_predictions[config_key]["y_pred"].extend(y_pred)

                            all_results.append({                                                # Log metadata and performance metrics for this specific hyperparameter configuration
                                "fold_id":     fold_id,
                                "n_components": pca_label,
                                "n_points":     "N/A",
                                "n_clusters":  n_clusters,
                                "compression" : comp,
                                "k":           k,
                                "accuracy":    accuracy
                            })

            # ── DTW branch ───────────────────────────────────────────────
            elif method == "dtw":
                distance_cache = []                             # Iterate through each gesture in the test set
                for test_g in test_proc:                        # Iterate through each gesture in the test set                                  
                    dists = []                                  # Temporary list to store distances for the current test gesture
                    for train_g in train_proc:                  # Compare against every gesture in the training set
                        # Calculate temporal similarity using optimized DTW (C-speed implementation)
                        dist = compute_dtw_distance_c_speed(
                            test_g["trajectory"], train_g["trajectory"])
                        dists.append((dist, train_g["gesture_type"]))       # Store distance paired with the known gesture label

                    dists.sort(key=lambda x: x[0])                          # Sort training gestures from most similar to least similar
                    distance_cache.append((test_g["gesture_type"], dists))  # Cache the sorted results for various K-value testing

                # ------- KNN Evaluation -------------
                for k in k_options:                                                 # Test different K-neighbor values to find the optimal configuration
                    y_true, y_pred = [], []                                         # Lists to track ground truth and model predictions
                    config_key = (pca_label, k)                                         # Unique identifier for this (PCA_components, K) configuration

                    if config_key not in global_predictions:                        # Initialize global results tracking if configuration is new
                        global_predictions[config_key] = {"y_true": [], "y_pred": []}

                    for true_label, sorted_dists in distance_cache:                 # Retrieve cached DTW results for each test sample
                        neighbors = sorted_dists[:k]                                # Select the top K most similar training samples
                        pred      = majority_vote(neighbors)                        # Determine predicted class via frequency of neighbor labels
                        y_true.append(true_label)                                   # Record the actual gesture type
                        y_pred.append(pred)                                         # Record the predicted gesture type

                    accuracy = np.mean(np.array(y_true) == np.array(y_pred))        # Calculate the classification success rate for this fold
                    global_predictions[config_key]["y_true"].extend(y_true)         # Merge fold truth labels into the global analysis set
                    global_predictions[config_key]["y_pred"].extend(y_pred)         # Merge fold predictions into the global analysis set

                    all_results.append({                                            # Log experimental metadata and performance for reporting
                        "fold_id":      fold_id,
                        "n_components": pca_label,
                        "n_points":     "N/A",
                        "n_clusters":   "N/A",
                        "compression" : "N/A",                                      # DTW bypasses clustering (unlike the Edit Distance branch)
                        "k":            k,
                        "accuracy":     accuracy
                    })
            

            # ── Three-cent ───────────────────────────────────────────────
            elif method == "three-cent" : 
                for n_points in n_points_options:

                    templates = build_templates(train_proc, n_points)

                    y_true, y_pred = [], []
                    config_key = (pca_label, n_points)

                    if config_key not in global_predictions:
                        global_predictions[config_key] = {"y_true": [], "y_pred": []}

                    for test_g in test_proc:
                        pred = recognize(test_g['trajectory'], templates, n_points)
                        y_true.append(test_g['gesture_type'])
                        y_pred.append(pred)

                    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                    global_predictions[config_key]["y_true"].extend(y_true)
                    global_predictions[config_key]["y_pred"].extend(y_pred)

                    all_results.append({
                        "fold_id":      fold_id,
                        "n_components": pca_label,
                        "n_points":     n_points,
                        "n_clusters":   "N/A",
                        "k":            1,
                        "compression":  "N/A",
                        "accuracy":     accuracy,
                    })

    return pd.DataFrame(all_results), global_predictions


if __name__ == "__main__":
    path_domain_1 = "C:/Users/PC/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain1_Mons/Domain1_csv"
    #path_domain_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    path_domain_4 = "C:/Users/PC/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain4_Mons"
    #path_domain_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    labels = list(range(10))

    # On which data we work 
    datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }

    # which base line method we use and with which hyperparameters
    methods = {
        "edit-distance": {
        "cluster_options": [5, 7, 9, 11, 13, 15,17,19,21],
        "compression" : [True,False],
        "n_points_options" : []
        },

        "dtw": {
            "cluster_options": [],        # unused for DTW, kept for uniform signature
            "compression" : [],
            "n_points_options" : []        # unused for DTW kept for uniform signature
        },

        "three-cent" : {
            "cluster_options": [0], 
            "compression" : [],
            "n_points_options" : [16, 32, 64, 128, 256]
        },
    }

    # PCA 
    pca_options = ["no_pca", 1, 2, 3]

    # which data splitting we use 
    cv_modes  = ["dependent", "independent"]


    # How many neighbors we look at 
    k_options = [1, 3, 5, 7]

    # the number of test we will proceed (combinatory)
    total = len(datasets) * len(methods) * len(cv_modes)
    done  = 0

    # ----- Proceed at all the tests with the good hyperprameter --------
    for domain_name, gestures in datasets.items():
        for method_name, params in methods.items():
            for cv_mode in cv_modes:

                done += 1
                config_label = f"{domain_name}_{method_name}_{cv_mode}"
                print(f"\n[{done}/{total}] Running: {config_label}")

                df, preds = run_pipeline(
                    gestures,
                    k_options        = k_options,
                    pca_options      = pca_options,
                    cluster_options  = params["cluster_options"],
                    compression      = params["compression"],
                    n_points_options = params["n_points_options"],
                    cv_mode          = cv_mode,
                    method           = method_name,
                )

                # groupby columns differ slightly by method
                group_cols = ["n_components", "n_clusters", "k", "compression", "n_points"]
                summary    = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
                best_config = summary["mean"].idxmax()
                print(f"  Best config: {best_config}  "
                      f"mean={summary.loc[best_config,'mean']:.4f}")

                # Confusion matrix for best config
                pca_label   = best_config[0]
                n_clusters  = best_config[1]
                k_best      = best_config[2]
                compression = best_config[3]
                n_points    = best_config[4]

                if method_name == "edit-distance":
                    key = (pca_label, n_clusters, k_best, compression)
                elif method_name == "dtw":
                    key = (pca_label, k_best)
                elif method_name == "three-cent" : 
                    key = (pca_label, n_points) 

                y_true = preds[key]["y_true"]
                y_pred = preds[key]["y_pred"]
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                save_results(summary, best_config, cm, df,
                             config_label, output_dir="results")

    print("\nAll done. Results saved in ./results/")
