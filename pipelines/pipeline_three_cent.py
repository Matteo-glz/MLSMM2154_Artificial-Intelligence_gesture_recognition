import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data.data_loading import load_data_domain_1, load_data_domain_4
from data.data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data.data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils.utils_saving import save_results

GROUP_COLS = ["n_components", "n_points"]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers — all identical to dollar_one.py
# ─────────────────────────────────────────────────────────────────────────────

def _path_length(points: np.ndarray) -> float:
    """Total arc-length of a polyline (works for 2-D or 3-D)."""
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


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
    total = _path_length(points)
    if total == 0:
        return np.tile(points[0], (n, 1))

    interval = total / (n - 1)
    D        = 0.0
    new_pts  = [points[0].copy()]

    i = 1
    while i < len(points) and len(new_pts) < n:
        d = float(np.linalg.norm(points[i] - points[i - 1]))
        if D + d >= interval:
            t = (interval - D) / d
            q = points[i - 1] + t * (points[i] - points[i - 1])
            new_pts.append(q)
            points = np.insert(points, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    while len(new_pts) < n:
        new_pts.append(points[-1].copy())

    return np.array(new_pts[:n])


def _centroid(points: np.ndarray) -> np.ndarray:
    """Mean position across all points."""
    return points.mean(axis=0)


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
    length = _path_length(points)
    if length == 0:
        return points
    return points / length


def _translate_to_origin(points: np.ndarray) -> np.ndarray:
    """Translate so that the centroid is at the origin."""
    return points - _centroid(points)


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
    pts = _resample(points, n)
    pts = _scale_by_length(pts)
    pts = _translate_to_origin(pts)
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
        dist = _path_distance(candidate, tmpl['preprocessed'])
        if dist < best_dist:
            best_dist = dist
            best_type = tmpl['gesture_type']

    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, pca_options, n_points_options,
                 cv_mode="dependent", val_fraction=0.20):

    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── PHASE 1 : HP selection sur TOUS les folds ────────────────────
    hp_scores = defaultdict(list)

    for train, test, fold_id in cv_fn(gestures):
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_norm, val_fraction)

        for n_components in pca_options:
            if n_components != "no_pca":
                pca     = fit_pca_per_gesture(inner_train, n_components)
                it_proc = apply_pca_per_gesture(inner_train, pca)
                iv_proc = apply_pca_per_gesture(inner_val,   pca)
            else:
                it_proc = inner_train
                iv_proc = inner_val

            for n_points in n_points_options:
                templates = build_templates(it_proc, n_points)
                y_pred_iv = [recognize(g["trajectory"], templates, n_points) for g in iv_proc]
                y_true_iv = [g["gesture_type"] for g in iv_proc]
                val_acc   = float(np.mean(np.array(y_true_iv) == np.array(y_pred_iv)))
                hp_scores[(n_components, n_points)].append(val_acc)

    # ← ICI, EN DEHORS de la boucle
    best_hp = max(hp_scores, key=lambda hp: np.mean(hp_scores[hp]))
    best_val_acc_global = float(np.mean(hp_scores[best_hp]))
    best_pca, best_n_points = best_hp
    print(f"  Global best HP: pca={best_pca}, n_points={best_n_points}")

    # ── PHASE 2 : Evaluation finale avec best_hp FIXE ────────────────
    all_results = []
    global_predictions = {"y_true": [], "y_pred": []}

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        if best_pca != "no_pca":
            pca        = fit_pca_per_gesture(train_norm, n_components=best_pca)
            train_proc = apply_pca_per_gesture(train_norm, pca)
            test_proc  = apply_pca_per_gesture(test_norm,  pca)
        else:
            train_proc = train_norm
            test_proc  = test_norm

        templates = build_templates(train_proc, best_n_points)
        y_true, y_pred = [], []
        for test_g in test_proc:
            pred = recognize(test_g["trajectory"], templates, best_n_points)
            y_true.append(test_g["gesture_type"])
            y_pred.append(pred)

        accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
        global_predictions["y_true"].extend(y_true)
        global_predictions["y_pred"].extend(y_pred)
        all_results.append({
            "fold_id":      fold_id,
            "n_components": best_pca,
            "n_points":     best_n_points,
            "val_accuracy":   best_val_acc_global,
            "accuracy":     accuracy,
        })
        print(f"    Test accuracy = {accuracy:.4f}")

    return pd.DataFrame(all_results), global_predictions, best_hp


if __name__ == "__main__":
    PATH_DOMAIN_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    PATH_DOMAIN_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }
    pca_options      = ["no_pca", 2, 3]
    n_points_options = [16, 32, 64]#, 128, 256]
    cv_modes         = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_three-cent_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds, best_config = run_pipeline(
                gestures, pca_options, n_points_options, cv_mode
            )

            mean_acc = df["accuracy"].mean()
            std_acc  = df["accuracy"].std()
            print(f"  Best config: {best_config}")
            print(f"  Mean accuracy : {mean_acc:.4f}")
            print(f"  Std           : {std_acc:.4f}")

            y_true = preds["y_true"]
            y_pred = preds["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            summary = df.groupby(["n_components", "n_points"])["accuracy"].agg(["mean", "std"])
            print(f"  Val accuracy   : {df['val_accuracy'].mean():.4f}")
            print(f"  Test accuracy  : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            save_results(summary, best_config, cm, df, config_label, output_dir="results")


    print("\nDone. Results saved in ./results/")
