from sklearn.metrics import confusion_matrix
import numpy as np

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