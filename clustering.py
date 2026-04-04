from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

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

def predict_gesture_type_knn(test_gesture, train_gestures, k=3, use_clean=True):
    '''
    Predict the gesture type using kNN on the edit distance between sequences.
     - use_clean=True → on utilise les séquences compressées (ABC)
     - use_clean=False → on utilise les séquences brutes (AAA)
    '''
    column = 'seq_clean' if use_clean else 'seq_raw'
    target_seq = test_gesture[column]
    
    distances = []
    
    # 1. Computes ALL distances with the strain set
    for train_g in train_gestures:
        dist = compute_edit_distance(target_seq, train_g[column])
        distances.append({
            "dist": dist,
            "gesture_type": train_g['gesture_type']
        })
    
    # 2. Sort by growing distances and take the k's first
    # sort dictionnary list by the the key 'dist'

    k_neighbors = sorted(distances, key=lambda x: x['dist'])[:k]
    
    # 3. Extract gesture types of the k neighbos
  
    neighbor_types = [n['gesture_type'] for n in k_neighbors]
    
    # 4. Take the majority
    # Vote majoritaire
    prediction = Counter(neighbor_types).most_common(1)[0][0]
    
    return prediction

def compute_class_metrics(y_true, y_pred, labels):
    '''
    Compute class-wise evaluation metrics based on the confusion matrix.

    This function evaluates the performance of a classification model 
    for each class independently using a one-vs-all approach.

    Parameters:
    - y_true: array-like, true class labels
    - y_pred: array-like, predicted class labels
    - labels: list of all class labels (ensures consistent ordering in the confusion matrix)

    Returns:
    - class_stats: dictionary mapping each class label to its evaluation metrics:
        * sensitivity (recall): TP / (TP + FN)
        * precision: TP / (TP + FP)
        * npv (negative predictive value): TN / (TN + FN)

    The confusion matrix is used to derive TP, FP, FN, TN for each class.
    '''

    # Compute confusion matrix:
    # rows = true labels, columns = predicted labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Initialize dictionary to store per-class metrics
    class_stats = {}
    
    # Iterate over each class index and corresponding label
    for i, label in enumerate(labels):

        # True Positives (TP): correctly predicted samples of class i
        tp = cm[i, i]

        # False Positives (FP): samples predicted as class i but belonging to other classes
        fp = cm[:, i].sum() - tp

        # False Negatives (FN): samples of class i predicted as another class
        fn = cm[i, :].sum() - tp

        # True Negatives (TN): all remaining samples correctly predicted as not class i
        tn = cm.sum() - (tp + fp + fn)
        
        # Sensitivity (Recall): ability to correctly identify class i
        # Measures how many actual class i samples are correctly detected
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Precision: reliability of predictions for class i
        # Measures how many predicted class i samples are correct
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value (NPV): reliability of negative predictions
        # Measures how many samples predicted as "not class i" are correct
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Store computed metrics for the current class
        class_stats[label] = {
            "sensitivity": sensitivity,
            "precision": precision,
            "npv": npv
        }

    # Return dictionary containing metrics for all classes
    return class_stats

