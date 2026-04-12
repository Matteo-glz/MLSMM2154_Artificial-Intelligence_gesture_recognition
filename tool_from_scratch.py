def edit_distance(seq1, seq2): 
    col = len(seq2) + 1
    row = len(seq1) + 1

    change = [0,0,0]

    matrix = [[0] * col for _ in range(row)]

    for i in range(col):
        matrix[0][i] = i
    for j in range(row):
        matrix[j][0] = j

    for i in range(1, row) : 
        for j in range(1,col) : 
            if seq1[i-1] == seq2[j-1] : 
                change[0] = matrix[i-1][j-1]
            else : 
                change[0] = matrix[i-1][j-1]+1
            change[1] = matrix[i][j-1]+1
            change[2] = matrix[i-1][j]+1
            print(change)
            matrix[i][j] = min(change)
    print(matrix)
    return matrix[row-1][col-1]





import numpy as np

def euclidean_distance(p1, p2):
    #return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))
    # or more simply: 
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_dtw_distance(seq1, seq2):
    '''
    Compute Dynamic Time Warping distance between two sequences.

    Parameters:
    - seq1, seq2: lists of vectors (e.g. 3D points)

    Returns:
    - DTW distance (float)
    '''

    n, m = len(seq1), len(seq2)

    # Initialize matrix with infinity
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    # Fill matrix
    for i in range(1, n+1):
        for j in range(1, m+1):

            cost = euclidean_distance(seq1[i-1], seq2[j-1])

            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )

    return dtw[n, m]
def compute_dtw_distance_window(seq1, seq2, window=None):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0

    w = window if window is not None else max(n, m)  # unconstrained by default

    for i in range(1, n+1):
        for j in range(max(1, i-w), min(m+1, i+w+1)):
            cost = np.linalg.norm(np.array(seq1[i-1]) - np.array(seq2[j-1]))
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    return dtw[n, m]

from numba import jit

@jit(nopython=True) # This decorator makes it lightning fast
def compute_dtw_distance_c_speed(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            # Using Euclidean distance manually inside for Numba compatibility
            dist = 0.0
            for k in range(len(seq1[i-1])):
                dist += (seq1[i-1][k] - seq2[j-1][k])**2
            cost = np.sqrt(dist)

            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]


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