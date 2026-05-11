"""
dollar_one.py
─────────────────────────────────────────────────────────────────────────────
$1 Gesture Recognizer — adapted for the existing pipeline.

Key design decisions vs. the original paper
--------------------------------------------
• 3-D trajectories (x, y, z) instead of 2-D (x, y).
  The paper's maths generalise trivially: just add a z-term wherever a
  distance is computed.  Rotation is still performed in the XY-plane only
  (the "indicative angle" trick), which is the standard adaptation for
  spatial / mid-air gesture data.

• Template library built from the TRAINING set at the start of each fold,
  so there is zero data leakage (same contract as the DTW branch).

• The only hyper-parameter is N (number of resample points).
  K from KNN is replaced by the "nearest template wins" logic that is
  intrinsic to $1 — no voting required.

• The function signature and return format of `run_pipeline` mirrors
  `run_pipeline` so results can be fed to the same `save_results` /
  `summary` code you already have.

Public API
----------
    build_templates(train_gestures, n_points)              → list of template dicts
    recognize(candidate_traj, templates, n_points, size)   → gesture_type (int)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_loading import load_data_domain_1, load_data_domain_4
from data_splitting import user_dependent_cv, user_independent_cv
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_saving import save_results

GROUP_COLS = ["n_components", "n_points"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _path_length(points: np.ndarray) -> float:
    """Total arc-length of a polyline (works for 2-D or 3-D)."""
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


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


def _indicative_angle(points: np.ndarray) -> float:
    """
    Angle (radians) from the centroid to the first point, measured in the XY-plane.
    This is the 'indicative angle' used as the rotation seed.
    """
    c = _centroid(points)
    return float(np.arctan2(c[1] - points[0, 1], c[0] - points[0, 0]))


def _rotate_by(points: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate points by `theta` radians around their centroid, in the XY-plane.
    The z-coordinate (if present) is left unchanged.
    """
    c     = _centroid(points)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rotated    = points.copy()
    dx         = points[:, 0] - c[0]
    dy         = points[:, 1] - c[1]
    rotated[:, 0] = dx * cos_t - dy * sin_t + c[0]
    rotated[:, 1] = dx * sin_t + dy * cos_t + c[1]
    return rotated


def _rotate_to_zero(points: np.ndarray) -> np.ndarray:
    """Rotate so that the indicative angle is at 0°."""
    return _rotate_by(points, -_indicative_angle(points))


def _scale_to_square(points: np.ndarray, size: float) -> np.ndarray:
    """
    Non-uniform scale so that the bounding box fits inside a `size`×`size` square.
    Applied per-axis (x, y, and z if present).
    """
    mins   = points.min(axis=0)
    maxs   = points.max(axis=0)
    ranges = maxs - mins

    scaled = points.copy()
    for dim in range(points.shape[1]):
        if ranges[dim] > 0:
            scaled[:, dim] = points[:, dim] * (size / ranges[dim])
    return scaled


def _translate_to_origin(points: np.ndarray) -> np.ndarray:
    """Translate so that the centroid is at the origin."""
    return points - _centroid(points)


def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Average point-to-point Euclidean distance between two equal-length paths.
    Implements PATH-DISTANCE from the paper.
    """
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _distance_at_angle(points: np.ndarray, template: np.ndarray,
                       theta: float) -> float:
    """Rotate candidate by theta, then measure path-distance to template."""
    return _path_distance(_rotate_by(points, theta), template)


_PHI = 0.5 * (-1.0 + np.sqrt(5.0))   # Golden Ratio constant


def _distance_at_best_angle(points: np.ndarray, template: np.ndarray,
                             theta_a: float = -np.pi / 4,
                             theta_b: float =  np.pi / 4,
                             theta_delta: float = np.deg2rad(2.0)) -> float:
    """
    Golden Section Search over [-45°, +45°] to find the rotation of `points`
    that minimises path-distance to `template`.
    Implements DISTANCE-AT-BEST-ANGLE from the paper.
    """
    x1 = _PHI * theta_a + (1.0 - _PHI) * theta_b
    f1 = _distance_at_angle(points, template, x1)
    x2 = (1.0 - _PHI) * theta_a + _PHI * theta_b
    f2 = _distance_at_angle(points, template, x2)

    while abs(theta_b - theta_a) > theta_delta:
        if f1 < f2:
            theta_b = x2
            x2, f2  = x1, f1
            x1 = _PHI * theta_a + (1.0 - _PHI) * theta_b
            f1 = _distance_at_angle(points, template, x1)
        else:
            theta_a = x1
            x1, f1  = x2, f2
            x2 = (1.0 - _PHI) * theta_a + _PHI * theta_b
            f2 = _distance_at_angle(points, template, x2)

    return min(f1, f2)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing pipeline (Steps 1-3) — same for templates and candidates
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SIZE = 250.0   # Reference square side length (paper default)


def _preprocess(points: np.ndarray, n: int,
                size: float = _DEFAULT_SIZE) -> np.ndarray:
    """
    Apply Steps 1-3 of the $1 algorithm to a raw trajectory.

    1. Resample to n equidistant points
    2. Rotate so indicative angle = 0°
    3. Scale to size × size square
    4. Translate centroid to origin

    Parameters
    ----------
    points : np.ndarray, shape (m, dims)  — raw trajectory
    n      : int                          — target number of points
    size   : float                        — reference square side (default 250)

    Returns
    -------
    np.ndarray, shape (n, dims) — preprocessed trajectory
    """
    pts = _resample(points, n)
    pts = _rotate_to_zero(pts)
    pts = _scale_to_square(pts, size)
    pts = _translate_to_origin(pts)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Template library
# ─────────────────────────────────────────────────────────────────────────────

def build_templates(train_gestures: list, n_points: int,
                    size: float = _DEFAULT_SIZE) -> list:
    """
    Pre-process every training gesture and store it as a template.

    This is called ONCE per (fold × n_points) combination, before recognition.
    Mirrors the role of `fit_kmeans` in the Edit-Distance branch.

    Parameters
    ----------
    train_gestures : list of gesture dicts (your standard format)
    n_points       : int — resample target (hyper-parameter)
    size           : float — reference square side

    Returns
    -------
    list of dicts, each containing:
        'gesture_type', 'gesture_name', 'subject', 'preprocessed'
    """
    templates = []
    for g in train_gestures:
        preprocessed = _preprocess(g['trajectory'], n_points, size)
        templates.append({
            'gesture_type': g['gesture_type'],
            'gesture_name': g.get('gesture_name', ''),
            'subject':      g['subject'],
            'preprocessed': preprocessed,
        })
    return templates


# ─────────────────────────────────────────────────────────────────────────────
# Recognition (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

def recognize(candidate_traj: np.ndarray, templates: list,
              n_points: int, size: float = _DEFAULT_SIZE) -> int:
    """
    Recognise a candidate gesture against a set of pre-built templates.

    Implements the RECOGNIZE function from the paper (Step 4).
    Returns the gesture_type of the nearest template.

    Parameters
    ----------
    candidate_traj : np.ndarray, shape (m, dims)
    templates      : list produced by build_templates()
    n_points       : int — must match the value used in build_templates()
    size           : float — must match the value used in build_templates()

    Returns
    -------
    int — predicted gesture_type
    """
    candidate = _preprocess(candidate_traj, n_points, size)

    best_dist = np.inf
    best_type = -1

    for tmpl in templates:
        dist = _distance_at_best_angle(candidate, tmpl['preprocessed'])
        if dist < best_dist:
            best_dist = dist
            best_type = tmpl['gesture_type']

    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(gestures, pca_options, n_points_options,
                 cv_mode="dependent", size: float = _DEFAULT_SIZE):
    all_results        = []
    global_predictions = {}
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    for train, test, fold_id in cv_fn(gestures):
        print(f"  Fold {fold_id}...", flush=True)
        mean, std  = fit_normalizer(train)
        train_norm = apply_normalizer(train, mean, std)
        test_norm  = apply_normalizer(test,  mean, std)

        for n_components in pca_options:
            pca_label = n_components if n_components != "no_pca" else "no_pca"
            if n_components != "no_pca":
                pca        = fit_pca_per_gesture(train_norm, n_components=n_components)
                train_proc = apply_pca_per_gesture(train_norm, pca)
                test_proc  = apply_pca_per_gesture(test_norm,  pca)
            else:
                train_proc = train_norm
                test_proc  = test_norm

            for n_points in n_points_options:
                y_true, y_pred = [], []
                config_key = (pca_label, n_points)
                if config_key not in global_predictions:
                    global_predictions[config_key] = {"y_true": [], "y_pred": []}

                # Build templates once per (fold, pca, n_points)
                templates = build_templates(train_proc, n_points, size)

                for test_g in test_proc:
                    pred = recognize(test_g['trajectory'], templates, n_points, size)
                    y_true.append(test_g['gesture_type'])
                    y_pred.append(pred)

                accuracy = np.mean(np.array(y_true) == np.array(y_pred))
                global_predictions[config_key]["y_true"].extend(y_true)
                global_predictions[config_key]["y_pred"].extend(y_pred)
                all_results.append({
                    "fold_id":      fold_id,
                    "n_components": pca_label,
                    "n_points":     n_points,
                    "accuracy":     accuracy,
                })

    return pd.DataFrame(all_results), global_predictions


if __name__ == "__main__":
    PATH_DOMAIN_1 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain1_Mons/Domain1_csv"
    PATH_DOMAIN_4 = "/Users/matteogalizia/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData/GestureDataDomain4_Mons"

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }
    pca_options      = ["no_pca", 1, 2, 3]
    n_points_options = [16, 32, 64, 128, 256]
    cv_modes         = ["dependent", "independent"]

    for domain_name, gestures in datasets.items():
        labels = sorted({g["gesture_type"] for g in gestures})
        for cv_mode in cv_modes:
            config_label = f"{domain_name}_dollar-one_{cv_mode}"
            print(f"\nRunning: {config_label}")

            df, preds   = run_pipeline(gestures, pca_options, n_points_options, cv_mode)
            summary     = df.groupby(GROUP_COLS)["accuracy"].agg(["mean", "std"])
            best_config = summary["mean"].idxmax()
            print(f"  Best config: {best_config}  mean={summary.loc[best_config,'mean']:.4f}")

            y_true = preds[best_config]["y_true"]
            y_pred = preds[best_config]["y_pred"]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            save_results(summary, best_config, cm, df, config_label, output_dir="results")

    print("\nDone. Results saved in ./results/")
