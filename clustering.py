from sklearn.cluster import KMeans
import numpy as np
from collections import Counter


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

import numpy as np
from numba import jit

# 1. Le "Moteur" (Core) : C'est ici que la magie opère.
# On travaille sur des entiers (codes ASCII) pour une vitesse maximale.
@jit(nopython=True)
def _edit_distance_core(s1, s2):
    n, m = len(s1), len(s2)
    
    # On utilise l'optimisation à deux lignes pour la mémoire
    prev_row = np.arange(m + 1)
    curr_row = np.zeros(m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i
        for j in range(1, m + 1):
            # Coût de substitution (comparaison d'entiers = ultra rapide)
            cost = 0 if s1[i-1] == s2[j-1] else 1
            
            # Calcul du minimum entre Insertion, Suppression et Substitution
            curr_row[j] = min(
                prev_row[j] + 1,      # Deletion
                curr_row[j - 1] + 1,  # Insertion
                prev_row[j - 1] + cost # Substitution
            )
        # Transfert de ligne
        prev_row[:] = curr_row[:]
        
    return prev_row[m]

# 2. La "Façade" (Wrapper) : C'est la fonction que tu appelles.
# Elle respecte ta contrainte : elle prend des chaînes de caractères (str).
def edit_distance_fast(str1, str2):
    # Conversion ultra-rapide de str vers array d'entiers (ASCII)
    # On utilise .encode() pour passer en bytes, puis np.frombuffer
    s1_arr = np.frombuffer(str1.encode('ascii'), dtype=np.uint8)
    s2_arr = np.frombuffer(str2.encode('ascii'), dtype=np.uint8)
    
    return _edit_distance_core(s1_arr, s2_arr)

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
        dist = edit_distance_fast(target_seq, train_g[column])
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



