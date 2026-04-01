import pandas as pd
import glob
import os
from sklearn.decomposition import PCA
import numpy as np


def load_data_sequences(directory_path):
    '''Load gesture data from CSV files and return a list of gesture dictionaries. Each dictionary contains:
     - "gesture_id": unique identifier for the gesture
     - "subject": subject number (e.g., 1 for "subject1")
     - "gesture_type": type of gesture (e.g., 1, 2, 3)
     - "repetition": repetition number (e.g., 1, 2, 3)
     - "trajectory": numpy array of shape (n_samples, 3) containing the x, y, z coordinates of the gesture trajectory
    '''
    gestures = []
    files = glob.glob(os.path.join(directory_path, "*.csv"))                # get all CSV files in the directory

    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path).replace('.csv', '')
        parts = filename.split('-') 
        
        df = pd.read_csv(file_path, header=None, names=['x','y','z','t'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        trajectory = df[['x','y','z']].values                               # only keep the spatial coordinates, ignore the time
        

        gestures.append({
            "gesture_id": i,
            "subject": int(parts[0][-1]),                                   # only take the subject's number (e.g., "subject1" -> 1)
            "gesture_type": int(parts[1]),
            "repetition": int(parts[2]),
            "trajectory": trajectory                                        #numpy array of shape (n_samples, 3)
        })
    print("Data loaded successfully.")
    return gestures

def user_independent_cv(gestures):
    # we create a sorted list of unique subject IDs present in the data
    subjects = sorted(set(g["subject"] for g in gestures))
    
    # we operate on each subject, the actuel subject will be the one set aside for testing
    for subject in subjects:
        # Train : we take all gestures that do not belong to the current subject
        train = [g for g in gestures if g["subject"] != subject]
        
        # Test : we take all gestures that belong to the current subject
        test = [g for g in gestures if g["subject"] == subject]
        
        # 3. 'yield' is a generator that allows us to iterate over the folds without storing them all in memory at once.
        # It returns the training and testing sets along with the ID of the excluded subject for this iteration.
        yield train, test, subject

def user_dependent_cv(gestures):
    # we create a sorted list of unique repetition numbers present in the data
    repetitions = sorted(set(g["repetition"] for g in gestures))
    
    # 2. We operate on each repetition, the actual repetition will be the one set aside for testing
    for rep in repetitions:
        # Train : we take all gestures that do not belong to the current repetition
        train = [g for g in gestures if g["repetition"] != rep]
        
        # Test : we test on the excluded repetition for all users (e.g., rep 1)
        test = [g for g in gestures if g["repetition"] == rep]
        
        # Return the training and testing sets along with the ID of the excluded repetition
        yield train, test, rep


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

def fit_pca(train_gestures, n_components=3):
    '''
    Tool used to fit a PCA on the training set, it will be used to t
    ransform both the training and testing sets in the same way, preventing data leakage
    '''
    all_points = np.vstack([g['trajectory'] for g in train_gestures])
    
    pca = PCA(n_components=n_components)
    pca.fit(all_points)
    
    return pca

def apply_pca(gestures, pca):
    '''
    Tool used to apply the PCA transformation to a set of gestures, using the PCA object fitted on the training set
    '''
    transformed = []
    
    for g in gestures:
        g_copy = g.copy()
        g_copy['trajectory'] = pca.transform(g['trajectory'])
        transformed.append(g_copy)
        
    return transformed


# --- TEST ---
path = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/Domain1_csv"
gesture_data = load_data_sequences(path)


def run_user_independent_pipeline(gestures):
    '''Pipeline for user-independent evaluation. It includes normalization, PCA, and a placeholder 
    for the model training and evaluation. The results are stored in a list of dictionaries containing 
    the fold identifier and the accuracy.'''
    results = []

    for train, test, subject in user_independent_cv(gestures):
        
        # --- NORMALISATION ---
        mean, std = fit_normalizer(train)
        train = apply_normalizer(train, mean, std)
        test = apply_normalizer(test, mean, std)
        
        # --- PCA ---
        pca = fit_pca(train)
        train = apply_pca(train, pca)
        test = apply_pca(test, pca)
        
        # --- MODEL (placeholder) ---
        accuracy = 0  # à remplacer
        
        results.append({
            "fold": subject,
            "accuracy": accuracy
        })
        
        print(f"User {subject} done.")

    return train, test



def run_user_dependent_pipeline(gestures):
    '''Pipeline for user-dependent evaluation. It includes normalization,
      PCA, and a placeholder for the model training and evaluation. The results are stored in a 
      list of dictionaries containing 
    the fold identifier and the accuracy.'''
    results = []

    for train, test, rep in user_dependent_cv(gestures):
        
        mean, std = fit_normalizer(train)
        train = apply_normalizer(train, mean, std)
        test = apply_normalizer(test, mean, std)
        
        pca = fit_pca(train)
        train = apply_pca(train, pca)
        test = apply_pca(test, pca)
        
        accuracy = 0
        
        results.append({
            "fold": rep,
            "accuracy": accuracy
        })
        
        print(f"Repetition {rep} done.")

    return results

