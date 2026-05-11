from collections import defaultdict
import numpy as np


def inner_val_split(train_gestures: list, val_fraction: float = 0.20,
                    seed: int = 42):
    """
    Carve out an inner validation set from the training portion of an outer fold.

    Used by all pipelines for hyperparameter selection — the outer test fold is
    never touched during this step, so HP selection is unbiased.

    Parameters
    ----------
    train_gestures : list — training gestures for the current outer fold
    val_fraction   : float — fraction to hold out (default 0.20 = 20 %)
    seed           : int — random seed for reproducibility

    Returns
    -------
    inner_train : list — 80 % of train_gestures (used to fit / train)
    inner_val   : list — 20 % of train_gestures (used to score HP candidates)
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(train_gestures))
    n_val   = max(1, int(len(train_gestures) * val_fraction))
    return ([train_gestures[i] for i in indices[n_val:]],
            [train_gestures[i] for i in indices[:n_val]])


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
    """
    Strict interpretation:
    For each fold, for each (user, gesture_type), remove ONE sample.
    """

    # group by (user, gesture_type)
    groups = defaultdict(list)
    for g in gestures:
        key = (g["subject"], g["gesture_type"])
        groups[key].append(g)

    # number of folds = max number of repetitions
    max_rep = max(len(v) for v in groups.values())

    for fold in range(max_rep):
        train = []
        test = []

        for key, samples in groups.items():
            if fold < len(samples):
                test.append(samples[fold])
                train.extend(samples[:fold] + samples[fold+1:])
            else:
                train.extend(samples)

        yield train, test, fold