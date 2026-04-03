from sklearn.cluster import KMeans
import numpy as np

def fit_kmeans(train_gestures, n_clusters=10):
    '''
    Learn the centroïds (the alphabet) only on the training set.
    '''
    # On regroupe tous les points de tous les gestes du train
    all_points = np.vstack([g['trajectory'] for g in train_gestures])
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    # n_init = 10 = run 10 times KMeans with different centroid seeds and take the best one (in terms of inertia)
    # random_state = 42 = for reproducibility
    kmeans.fit(all_points)

    print("Kmeans fitted on training data. Centroids shape:", kmeans.cluster_centers_.shape)
    return kmeans

def apply_symbolic_transformation(gestures, kmeans):
    '''
    Trajectory transformation: from raw trajectories to "raw sequences" (ex: "AAAAABBBCCCC").
    '''
    raw_gestures = []
    for g in gestures:
        g_copy = g.copy()
        
        # Clusters prediction for each point
        clusters = kmeans.predict(g['trajectory'])
        
        # Characters transformation: 0->A, 1->B, ...
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
        raw = g['seq_raw']
        
        if not raw:
            g_copy['seq_clean'] = ""
        else:
            # Keep only the character if it's different from the previous one
            clean = [raw[0]]
            for char in raw[1:]:
                if char != clean[-1]:
                    clean.append(char)
            g_copy['seq_clean'] = "".join(clean)
            
        compressed_gestures.append(g_copy)
    return compressed_gestures

def compute_edit_distance(seq1, seq2):
    ''' Compute the edit-distance (Levenshtein distance) between two sequences.'''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    
    for x in range(size_x): matrix[x, 0] = x
    for y in range(size_y): matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = matrix[x-1, y-1]
            else:
                matrix[x,y] = min(
                    matrix[x-1, y] + 1,    # Deletion
                    matrix[x, y-1] + 1,    # Insertion
                    matrix[x-1, y-1] + 1   # Substitution
                )
    return matrix[size_x - 1, size_y - 1]

from collections import Counter

def predict_gesture_type_knn(test_gesture, train_gestures, k=3, use_clean=True):
    '''
    Predict the gesture type using kNN on the edit distance between sequences.
     - use_clean=True → on utilise les séquences compressées (ABC)
     - use_clean=False → on utilise les séquences brutes (AAA)
    '''
    column = 'seq_clean' if use_clean else 'seq_raw'
    target_seq = test_gesture[column]
    
    distances = []
    
    # 1. Calculer TOUTES les distances avec le set d'entraînement
    for train_g in train_gestures:
        dist = compute_edit_distance(target_seq, train_g[column])
        distances.append({
            "dist": dist,
            "gesture_type": train_g['gesture_type']
        })
    
    # 2. Trier par distance croissante et prendre les k premiers
    # On trie la liste de dictionnaires selon la clé 'dist'
    k_neighbors = sorted(distances, key=lambda x: x['dist'])[:k]
    
    # 3. Extraire les types de gestes de ces k voisins
    neighbor_types = [n['gesture_type'] for n in k_neighbors]
    
    # 4. Vote majoritaire
    # Counter([1, 2, 1]).most_common(1) retourne [(type_majoritaire, nombre_de_votes)]
    prediction = Counter(neighbor_types).most_common(1)[0][0]
    
    return prediction


from collections import Counter

def run_user_independent_pipeline(gestures):

    results = []

    # 1. CROSS VALIDATION (LOSO)
    for train, test, subject in user_independent_cv(gestures):

        print(f"\n--- Testing subject {subject} ---")

        # =========================
        # 2. KMEANS (FIT ONLY TRAIN)
        # =========================
        kmeans = fit_kmeans(train, n_clusters=10)

        # =========================
        # 3. SYMBOLIC TRANSFORMATION
        # =========================
        train_sym = apply_symbolic_transformation(train, kmeans)
        test_sym  = apply_symbolic_transformation(test, kmeans)

        # =========================
        # 4. COMPRESSION
        # =========================
        train_sym = apply_compression(train_sym)
        test_sym  = apply_compression(test_sym)

        # =========================
        # 5. PREDICTION (KNN)
        # =========================
        correct = 0

        for test_g in test_sym:

            pred = predict_gesture_type_knn(
                test_gesture=test_g,
                train_gestures=train_sym,
                k=3,
                use_clean=True
            )

            if pred == test_g["gesture_type"]:
                correct += 1

        accuracy = correct / len(test_sym)

        # =========================
        # 6. STORE RESULTS
        # =========================
        results.append({
            "subject": subject,
            "accuracy": accuracy
        })

        print(f"Subject {subject} accuracy: {accuracy:.4f}")

    return results





