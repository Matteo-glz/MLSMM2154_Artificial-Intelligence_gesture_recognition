
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

'''Here the functions we used at a moment sometimes it was just ideas
    Sometimes it was to test something '''

def test_raw_data_and_load_data(gestures) : 
    '''founction used to test if the raw data loaded with load_data_domain_4
      is correct by comparing it with the raw data read directly from the csv files.'''
    
    # to use this function you need to add the file path in the getsure dictionnary
    # => "file_path": file_path  #useful for debug 
    for g in gestures:
        df_raw = pd.read_csv(
        g["file_path"],
        skiprows=5,
        header=None,
        usecols=[0,1,2]
    )
    
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce').dropna()
    
        if not np.allclose(g["trajectory"], df_raw.values):
            print("Mismatch detected:", g["file_path"])
            break
    else:
        print("✅ All trajectories are correct!")


def compute_class_metrics(y_true, y_pred, labels):
    # not used because heavy function
    '''
    Compute per-class evaluation metrics (sensitivity, precision, NPV) 
    from predictions using the confusion matrix.

    Parameters:
    - y_true: list or array of true class labels
    - y_pred: list or array of predicted class labels
    - labels: list of all possible class labels (ensures consistent ordering)

    Returns:
    - class_stats: dictionary where each key is a class label and each value
      is a dictionary containing:
        * sensitivity (recall): ability to correctly identify positive samples
        * precision: reliability of positive predictions
        * npv (negative predictive value): reliability of negative predictions

    The function computes metrics independently for each class using a 
    one-vs-all strategy derived from the confusion matrix.
    '''

    # Compute confusion matrix: rows = true labels, columns = predicted labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Initialize dictionary to store metrics per class
    class_stats = {}
    
    # Loop over each class using its index and label
    for i, label in enumerate(labels):
        
        # True Positives (TP): correctly predicted samples of class i
        tp = cm[i, i]
        
        # False Positives (FP): samples predicted as class i but belonging to other classes
        fp = cm[:, i].sum() - tp
        
        # False Negatives (FN): samples of class i predicted as other classes
        fn = cm[i, :].sum() - tp
        
        # True Negatives (TN): all remaining samples correctly predicted as not belonging to class i
        tn = cm.sum() - (tp + fp + fn)
        
        # Sensitivity (Recall): proportion of actual positives correctly identified
        # Measures how well the model captures class i
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Precision: proportion of predicted positives that are actually correct
        # Measures reliability when the model predicts class i
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value (NPV): proportion of predicted negatives that are correct
        # Measures reliability when the model predicts "not class i"
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Store computed metrics for the current class
        class_stats[label] = {
            "sensitivity": sensitivity,
            "precision": precision,
            "npv": npv
        }

    # Return dictionary containing metrics for all classes
    return class_stats

def summarize_results(df):
    '''
    Aggregate cross-validation results to compute mean accuracy and standard deviation
    for each hyperparameter configuration.

    Parameters:
    - df: pandas DataFrame containing at least the following columns:
        * 'n_clusters': number of clusters used in KMeans
        * 'k_neighbors': number of neighbors used in k-NN
        * 'accuracy_fold': accuracy obtained for each fold

    Returns:
    - summary: pandas DataFrame indexed by (n_clusters, k_neighbors) containing:
        * mean: average accuracy across folds
        * std: standard deviation of accuracy across folds

    This function allows comparison of model performance and stability across
    different hyperparameter configurations.
    '''

    # Group results by hyperparameter configuration and compute statistics
    summary = df.groupby(['n_clusters', 'k_neighbors'])['accuracy_fold'].agg(['mean', 'std'])

    # Return aggregated performance metrics
    return summary

def compute_global_confusion_matrix(global_predictions, n_clusters, k, labels):
    '''
    Compute a global confusion matrix by aggregating predictions across all folds
    for a given hyperparameter configuration.

    Parameters:
    - global_predictions: dictionary storing predictions for each configuration:
        key = (n_clusters, k)
        value = {
            "y_true": list of true labels across all folds,
            "y_pred": list of predicted labels across all folds
        }
    - n_clusters: selected number of clusters
    - k: selected number of neighbors
    - labels: list of class labels (ensures consistent matrix structure)

    Returns:
    - confusion matrix (2D numpy array)

    This matrix summarizes overall classification performance across all folds,
    providing a global view of class-wise errors and correct predictions.
    '''

    # Retrieve the key corresponding to the chosen configuration
    key = (n_clusters, k)

    # Extract aggregated true and predicted labels
    y_true = global_predictions[key]["y_true"]
    y_pred = global_predictions[key]["y_pred"]

    # Compute and return confusion matrix
    return confusion_matrix(y_true, y_pred, labels=labels)

def compute_mean_std_cm(global_predictions, n_clusters, k):
    # no used difficulty to interpret and not in our requirement
    '''
    Compute the mean and standard deviation of confusion matrices across folds
    for a given hyperparameter configuration.

    Parameters:
    - global_predictions: dictionary storing results for each configuration:
        key = (n_clusters, k)
        value = {
            "cms": list of confusion matrices (one per fold)
        }
    - n_clusters: selected number of clusters
    - k: selected number of neighbors

    Returns:
    - mean_cm: average confusion matrix across folds
    - std_cm: standard deviation of confusion matrices across folds

    This function provides a more detailed analysis of model behavior by showing
    how classification patterns vary across folds. Each cell (i, j) represents
    the mean and variability of predictions from class i to class j.

    Note:
    The resulting matrices contain averaged values (not integer counts) and are
    mainly intended for analysis and visualization, not for computing metrics.
    '''

    # Retrieve list of confusion matrices for the given configuration
    cms = global_predictions[(n_clusters, k)]["cms"]

    # Compute element-wise mean across all folds
    mean_cm = np.mean(cms, axis=0)

    # Compute element-wise standard deviation across folds
    std_cm = np.std(cms, axis=0)

    # Return aggregated matrices
    return mean_cm, std_cm

