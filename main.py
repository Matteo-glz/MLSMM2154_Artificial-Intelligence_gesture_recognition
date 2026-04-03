from data_preparation import load_data_domain_1

from data_preparation import user_independent_cv,user_dependent_cv
from data_preparation import fit_normalizer,apply_normalizer,fit_pca, apply_pca
from clustering import fit_kmeans, apply_symbolic_transformation, apply_compression, predict_gesture_type_knn

def run_user_independent_pipeline(gestures):

    results = []

    for train, test, subject in user_independent_cv(gestures):

        print(f"\n--- User {subject} ---")

        # =========================
        # 1. NORMALISATION
        # =========================
        mean, std = fit_normalizer(train)
        train = apply_normalizer(train, mean, std)
        test = apply_normalizer(test, mean, std)

        # =========================
        # 2. PCA
        # =========================
        pca = fit_pca(train)
        train = apply_pca(train, pca)
        test = apply_pca(test, pca)

        # =========================
        # 3. KMEANS (TRAIN ONLY)
        # =========================
        kmeans = fit_kmeans(train, n_clusters=10)

        # =========================
        # 4. SYMBOLIC TRANSFORMATION
        # =========================
        train_sym = apply_symbolic_transformation(train, kmeans)
        test_sym  = apply_symbolic_transformation(test, kmeans)

        # =========================
        # 5. COMPRESSION
        # =========================
        train_sym = apply_compression(train_sym)
        test_sym  = apply_compression(test_sym)

        # =========================
        # 6. MODEL (KNN CLASSIFIER)
        # =========================
        correct = 0

        for test_g in test_sym:

            pred = predict_gesture_type_knn(
                test_gesture=test_g,
                train_gestures=train_sym,
                k=3,
                use_clean= True
            )

            if pred == test_g["gesture_type"]:
                correct += 1

        accuracy = correct / len(test_sym)

        # =========================
        # 7. STORE RESULT
        # =========================
        results.append({
            "fold": subject,
            "accuracy": accuracy
        })

        print(f"User {subject} accuracy: {accuracy:.4f}")

    return results

def run_user_dependent_pipeline(gestures):

    results = []

    for train, test, subject in user_dependent_cv(gestures):

        print(f"\n--- User {subject} ---")

        # =========================
        # 1. NORMALISATION
        # =========================
        mean, std = fit_normalizer(train)
        train = apply_normalizer(train, mean, std)
        test = apply_normalizer(test, mean, std)

        # =========================
        # 2. PCA
        # =========================
        pca = fit_pca(train)
        train = apply_pca(train, pca)
        test = apply_pca(test, pca)

        # =========================
        # 3. KMEANS (TRAIN ONLY)
        # =========================
        kmeans = fit_kmeans(train, n_clusters=10)

        # =========================
        # 4. SYMBOLIC TRANSFORMATION
        # =========================
        train_sym = apply_symbolic_transformation(train, kmeans)
        test_sym  = apply_symbolic_transformation(test, kmeans)

        # =========================
        # 5. COMPRESSION
        # =========================
        train_sym = apply_compression(train_sym)
        test_sym  = apply_compression(test_sym)

        # =========================
        # 6. MODEL (KNN CLASSIFIER)
        # =========================
        correct = 0

        for test_g in test_sym:

            pred = predict_gesture_type_knn(
                test_gesture=test_g,
                train_gestures=train_sym,
                k=3,
                use_clean= True
            )

            if pred == test_g["gesture_type"]:
                correct += 1

        accuracy = correct / len(test_sym)

        # =========================
        # 7. STORE RESULT
        # =========================
        results.append({
            "fold": subject,
            "accuracy": accuracy
        })

        print(f"User {subject} accuracy: {accuracy:.4f}")

    return results

path_domaine1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
path_domaine4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

gestures_1 = load_data_domain_1(path_domaine1)

#results_independant_domain1 = run_user_independent_pipeline(gestures_1)

results_dependant_domaine1 = run_user_dependent_pipeline(gestures_1)
