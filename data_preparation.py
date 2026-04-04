
from sklearn.decomposition import PCA
import numpy as np



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
        
        '''
        accuracy = 0
        
        results.append({
            "fold": rep,
            "accuracy": accuracy
        })
        
        print(f"Repetition {rep} done.")
        '''

    return train,test





