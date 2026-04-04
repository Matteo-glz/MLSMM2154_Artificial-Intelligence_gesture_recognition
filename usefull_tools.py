
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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

