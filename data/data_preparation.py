
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict


def fit_normalizer(train_gestures):
    '''
    Tool used to compute the mean and std of the training set for normalization
    it will be used to normalize both the training and testing sets in the same way, preventing data leakage
    '''

    all_points = np.vstack([g['trajectory'] for g in train_gestures])
    
    mean = np.mean(all_points, axis=0)
    std = np.std(all_points, axis=0)
    
    return mean, std

def apply_normalizer(gestures, mean, std):
    '''Tool used to apply the normalization to a set of gestures, using the mean and std computed on the training set'''
    normalized = []
    
    for g in gestures:
        g_copy = g.copy()
        g_copy['trajectory'] = (g['trajectory'] - mean) / std
        normalized.append(g_copy)
    
    return normalized


def fit_pca_per_gesture(train_gestures, n_components):
    """
    Fit one PCA per gesture type, using only training data.
    Returns a dict: {gesture_type -> fitted PCA object}
    """
    groups = defaultdict(list)
    for g in train_gestures:
        groups[g['gesture_type']].append(g['trajectory'])
    
    pcas = {}
    for gesture_type, trajs in groups.items():
        all_points = np.vstack(trajs)
        pca = PCA(n_components=n_components)
        pca.fit(all_points)
        pcas[gesture_type] = pca
    return pcas

def apply_pca_per_gesture(gestures, pcas):
    """
    Apply the correct PCA to each gesture based on its type.
    Works for both train and test sets.
    """
    transformed = []
    for g in gestures:
        g_copy = g.copy()
        pca = pcas[g['gesture_type']]  # use the PCA fitted for this class
        g_copy['trajectory'] = pca.transform(g['trajectory'])
        transformed.append(g_copy)
    return transformed





   





