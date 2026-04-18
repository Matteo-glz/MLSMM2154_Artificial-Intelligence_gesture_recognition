"""
three_cent.py
─────────────────────────────────────────────────────────────────────────────
3-Cent Gesture Recognizer for 3D mid-air gestures.
Based on: Caputo et al., "Comparing 3D trajectories for simple mid-air
gesture recognition", Computers & Graphics 73 (2018) 17-25.

Relation to dollar_one.py
--------------------------
This file is a direct evolution of dollar_one.py. Every helper function
is IDENTICAL except for three focused changes in _preprocess():

    $1                          3-cent
    ──────────────────────────  ──────────────────────────────────────
    rotate to indicative angle  NO rotation (orientation is kept)
    scale by bounding box       scale by trajectory length (uniform)
    translate to centroid       translate to centroid  (same)

And one change in recognize():

    $1                          3-cent
    ──────────────────────────  ──────────────────────────────────────
    Golden Section Search       direct _path_distance (no angle search
    over ±45° to find best      needed because we never rotated)
    angular alignment

Why these changes matter for 3D mid-air gestures
-------------------------------------------------
• No rotation: gesture direction is discriminative. A swipe-left and a
  swipe-right should NOT compare as identical after alignment. Removing
  rotation preserves this information.

• Length-based scaling: bounding-box scaling stretches each axis
  independently, distorting the shape of the gesture path. Scaling
  uniformly by total arc length keeps the path's proportions intact.

• No GSS: because there is no angular degree of freedom left, the
  matching reduces to a single path-distance call — faster and simpler.

Public API (mirrors dollar_one.py exactly)
------------------------------------------
    build_templates(train_gestures, n_points) → list of template dicts
    recognize(candidate_traj, templates, n_points) → gesture_type (int)
    run_pipeline_three_cent(gestures, n_points_options, pca_options,
                            cv_mode) → (DataFrame, global_predictions)
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers — all identical to dollar_one.py
# ─────────────────────────────────────────────────────────────────────────────

def _path_length(points: np.ndarray) -> float:
    """Total arc-length of a polyline (works for 2-D or 3-D)."""
    diffs = np.diff(points, axis=0)                 # Compute the vector between each succesive point (poins ABC => B-A, C-B)
    distances = np.linalg.norm(diffs, axis=1)       # Compute the eaclidian distance between each pair of points
    return float(np.sum(distances))                 # Add all the distances to get the total length of the path


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a variable-length point path into exactly n evenly-spaced points.

    Implements the RESAMPLE function from Appendix A of the paper,
    extended to arbitrary dimensionality.

    Parameters
    ----------
    points : np.ndarray, shape (m, dims)
    n      : int — target number of points

    Returns
    -------
    np.ndarray, shape (n, dims)
    """
    total   = _path_length(points)
    if total == 0:                                # degenerate gesture (single point)
        return np.tile(points[0], (n, 1))

    interval = total / (n - 1)                   # desired spacing between new points 
    D        = 0.0                               # set the acumulated distance to 0
    new_pts  = [points[0].copy()]                # start the new path with the first point of the original path 

    i = 1
    while i < len(points) and len(new_pts) < n:
        d = float(np.linalg.norm(points[i] - points[i - 1])) # distance between the two succesive point in the original path

        if D + d >= interval:                                    # if we have exceeded the desired spacing we need to add a new point by interpolation
            # Linear interpolation: place a new point at exactly `interval` distance
            t  = (interval - D) / d                              # compute the interpolation factor (between 0 and 1) to find the position of the new point along the segment between points[i-1] and points[i]
            q  = points[i - 1] + t * (points[i] - points[i - 1]) # compute the new point by linear interpolation between points[i-1] and points[i]
            new_pts.append(q)                                    # add the new point to the resampled path

            # Insert q back so we can continue from it
            points = np.insert(points, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    # Floating-point rounding may leave us one point short — duplicate the last
    while len(new_pts) < n:
        new_pts.append(points[-1].copy())

    return np.array(new_pts[:n])


def _centroid(points: np.ndarray) -> np.ndarray:
    """Mean position across all points."""
    return points.mean(axis=0)          # compute the mean of each column (X,Y,Z) to get the centroïd. 


def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Average point-to-point Euclidean distance between two equal-length paths.
    Implements PATH-DISTANCE from the paper.
    """
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — Scaling function 
# ─────────────────────────────────────────────────────────────────────────────

def _scale_by_length(points: np.ndarray) -> np.ndarray:
    """
    Uniform scale so that the total arc-length of the path becomes 1.

    This is the key difference from $1's bounding-box scaling:
    - $1 stretches each axis independently → distorts the path shape.
    - 3-cent scales all axes by the same factor → preserves proportions.

    After this step two gestures of the same shape but different sizes
    will be identical, while two gestures of different orientations
    will still differ (because we did NOT rotate first).
    """
    length = _path_length(points) # set the length of the path to the total arc-length of the path
    if length == 0:
        return points
    return points / length        # Scale all the points bu the same factor (the total length) to make the total length of the path equal to 1. 


def _translate_to_origin(points: np.ndarray) -> np.ndarray:
    """Translate so that the centroid is at the origin."""
    return points - _centroid(points) # set the centroid of the path to the origin (0,0,0)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — Preprocessing pipeline 
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(points: np.ndarray, n: int) -> np.ndarray:
    """
    Apply the 3-cent normalisation pipeline to a raw trajectory.

    Steps
    -----
    1. Resample to n equidistant points    
    2. Scale by trajectory length (uniform)
    3. Translate centroid to origin

    Parameters
    ----------
    points : np.ndarray, shape (m, dims)
    n      : int — target number of resampled points

    Returns
    -------
    np.ndarray, shape (n, dims)
    """
    pts = _resample(points, n)          # Resample the original path to n equidistant points
    pts = _scale_by_length(pts)         # Scale the resampled to a refereence length of 1 
    pts = _translate_to_origin(pts)     # Translate the scaled path to the orignin (0,0,0)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Template library 
# ─────────────────────────────────────────────────────────────────────────────

def build_templates(train_gestures: list, n_points: int) -> list:
    """
    Pre-process every training gesture and store it as a template.
    Called once per (fold × n_points) combination.

    Parameters
    ----------
    train_gestures : list of gesture dicts (standard pipeline format)
    n_points       : int — resample target (the only hyper-parameter)

    Returns
    -------
    list of dicts with keys:
        'gesture_type', 'gesture_name', 'subject', 'preprocessed'
    """
    templates = []
    for g in train_gestures:
        preprocessed = _preprocess(g['trajectory'], n_points)
        templates.append({
            'gesture_type': g['gesture_type'],
            'gesture_name': g.get('gesture_name', ''),
            'subject':      g['subject'],
            'preprocessed': preprocessed,
        })
    return templates


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3 — recognize() uses direct path_distance (no GSS angle search)
# ─────────────────────────────────────────────────────────────────────────────

def recognize(candidate_traj: np.ndarray, templates: list,
              n_points: int) -> int:
    """
    Recognise a candidate gesture against pre-built templates.

    Because there is no rotation step, there is no angular degree of
    freedom to optimise over. Recognition reduces to a single
    path-distance call per template — much simpler than $1's GSS loop.

    Parameters
    ----------
    candidate_traj : np.ndarray, shape (m, dims)
    templates      : list produced by build_templates()
    n_points       : int — must match the value used in build_templates()

    Returns
    -------
    int — predicted gesture_type (nearest template wins)
    """
    candidate = _preprocess(candidate_traj, n_points)

    best_dist = np.inf
    best_type = -1

    for tmpl in templates:
        # Direct distance — no angle search needed
        dist = _path_distance(candidate, tmpl['preprocessed'])
        if dist < best_dist:
            best_dist = dist
            best_type = tmpl['gesture_type']

    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration — mirrors run_pipeline_dollar_one exactly
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_three_cent(
        gestures: list,
        n_points_options: list,   # e.g. [16, 32, 64, 128]
        pca_options: list,        # e.g. ["no_pca", 1, 2, 3]
        cv_mode: str = "dependent",
):
    """
    Run the full cross-validated 3-cent experiment.

    Identical contract to run_pipeline_dollar_one — same return format,
    same column names, compatible with save_results / summary code.

    Hyper-parameters swept
    ----------------------
    n_points     : number of resample points (the only 3-cent parameter)
    n_components : PCA components (same as all other methods)

    Note: like $1, 3-cent is a nearest-template classifier.
    'k' is fixed at 1 and 'n_clusters'/'compression' are 'N/A'.
    """
    from data_splitting   import user_dependent_cv, user_independent_cv
    from data_preparation import (fit_normalizer, apply_normalizer,
                                  fit_pca_per_gesture, apply_pca_per_gesture)

    all_results        = []
    global_predictions = {}

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)

        # Normalisation — fitted on train only
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        # PCA sweep
        for n_components in pca_options:
            label = n_components if n_components != "no_pca" else "no_pca"

            if n_components != "no_pca":
                pca        = fit_pca_per_gesture(train_norm, n_components=n_components)
                train_proc = apply_pca_per_gesture(train_norm, pca)
                test_proc  = apply_pca_per_gesture(test_norm,  pca)
            else:
                train_proc = train_norm
                test_proc  = test_norm

            # n_points sweep
            for n_points in n_points_options:

                templates = build_templates(train_proc, n_points)

                y_true, y_pred = [], []
                config_key = (label, n_points)

                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                for test_g in test_proc:
                    pred = recognize(test_g['trajectory'], templates, n_points)
                    y_true.append(test_g['gesture_type'])
                    y_pred.append(pred)

                accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)

                all_results.append({
                    "fold_id":      fold_id,
                    "n_components": label,
                    "n_points":     n_points,
                    "n_clusters":   "N/A",
                    "k":            1,
                    "compression":  "N/A",
                    "accuracy":     accuracy,
                })

    return pd.DataFrame(all_results), global_predictions



# tool 
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# 1. Data Loading 
from data_loading import load_data_domain_1, load_data_domain_4



# 6. export results
from saving_result import save_results

if __name__ == "__main__":
    path_domain_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    path_domain_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    labels = list(range(10))

    # On which data we work 
    datasets = {
        "domain1": load_data_domain_1(path_domain_1),
        "domain4": load_data_domain_4(path_domain_4),
    }


    # which data splitting we use 
    cv_modes  = ["dependent", "independent"]

    # the number of test we will proceed (combinatory)
    total = len(datasets) * len(cv_modes)
    done  = 0

    # ----- Proceed at all the tests with the good hyperprameter --------
    for domain_name, gestures in datasets.items():
            for cv_mode in cv_modes:

                done += 1
                config_label = f"{domain_name}_{"Three-cent"}_{cv_mode}"
                print(f"\n[{done}/{total}] Running: {config_label}")

                df, preds = run_pipeline_three_cent(
                    gestures         = gestures,
                    n_points_options = [16, 32, 64, 128, 256],
                    pca_options      = ["no_pca", 1, 2, 3],
                    cv_mode          = cv_mode,
                )

                # groupby columns differ slightly by method
                group_cols = ["n_components", "n_points"]
                summary    = df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
                best_config = summary["mean"].idxmax()
                print(f"  Best config: {best_config}  "
                      f"mean={summary.loc[best_config,'mean']:.4f}")

                # Confusion matrix for best config
                pca_label  = best_config[0]
                n_points = best_config[1]

                key = (pca_label, n_points)

                y_true = preds[key]["y_true"]
                y_pred = preds[key]["y_pred"]
                cm = confusion_matrix(y_true, y_pred, labels=labels)

                save_results(summary, best_config, cm, df,
                             config_label, output_dir="results")

    print("\nAll done. Results saved in ./results/")
