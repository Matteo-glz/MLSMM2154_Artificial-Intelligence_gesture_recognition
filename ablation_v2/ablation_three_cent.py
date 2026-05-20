"""
ablation_three_cent.py
─────────────────────────────────────────────────────────────────────────────
Ablation study for Three-Cent / $1-derived template matching.

Hyperparameters swept:
  - n_resample    : [32, 64, 128]                — resampling resolution
  - normalization : ['three_cent', 'none', 'bounding_box', 'zscore']
                    — spatial normalization applied after resampling
  - n_templates   : [1, 3, 5, 'all']            — templates per class
  - selection     : ['first', 'centroid', 'random']
                    — how templates are picked from training samples
  - distance      : ['path_distance', 'dtw', 'hausdorff']
                    — similarity metric between preprocessed trajectories

Protocol: 2-phase cross-validation identical to main pipeline.

Outputs: ablation_v2/ablation_v2_results/three_cent_ablation.csv

Notes:
  - 'centroid' selection builds one mean template per class (n_templates ignored).
  - 'all' templates uses every training sample as a template.
  - 'hausdorff' is symmetric max-min distance; slower than path_distance.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from data.data_loading import load_data_domain_1, load_data_domain_4
from data.data_splitting import user_dependent_cv, user_independent_cv, inner_val_split
from data.data_preparation import fit_normalizer, apply_normalizer

# ─────────────────────────────────────────────────────────────────────────────
# Paths & output
# ─────────────────────────────────────────────────────────────────────────────

PATH_DOMAIN_1 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain1_Mons/Domain1_csv"
PATH_DOMAIN_4 = "/Users/simoensm/Documents/GitHub/MLSMM2154_Artificial-Intelligence_gesture_recognition/GestureData_Mons/GestureDataDomain4_Mons"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_v2_results")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ─────────────────────────────────────────────────────────────────────────────

N_RESAMPLE_OPTIONS   = [32, 64, 128]
NORM_OPTIONS         = ["three_cent", "none", "bounding_box", "zscore"]
N_TEMPLATES_OPTIONS  = [1, 3, 5, "all"]
SELECTION_OPTIONS    = ["first", "centroid", "random"]
DISTANCE_OPTIONS     = ["path_distance", "dtw", "hausdorff"]
CV_MODES             = ["dependent", "independent"]

# ─────────────────────────────────────────────────────────────────────────────
# Resampling (uniform arc-length, identical to original three-cent)
# ─────────────────────────────────────────────────────────────────────────────

def _path_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _resample_arc(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample to n equally arc-length-spaced points (original three-cent method)."""
    total = _path_length(pts)
    if total == 0:
        return np.tile(pts[0], (n, 1))

    interval  = total / (n - 1)
    D         = 0.0
    new_pts   = [pts[0].copy()]
    i         = 1

    while i < len(pts) and len(new_pts) < n:
        d = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if D + d >= interval:
            t = (interval - D) / d
            q = pts[i - 1] + t * (pts[i] - pts[i - 1])
            new_pts.append(q)
            pts = np.insert(pts, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    while len(new_pts) < n:
        new_pts.append(pts[-1].copy())

    return np.array(new_pts[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Spatial normalization options
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_three_cent(pts: np.ndarray) -> np.ndarray:
    """Original 3-cent: scale by arc-length, translate centroid to origin."""
    length = _path_length(pts)
    if length > 0:
        pts = pts / length
    return pts - pts.mean(axis=0)


def _normalize_bounding_box(pts: np.ndarray) -> np.ndarray:
    """Scale each axis independently to [0, 1] bounding box, then center."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    rng  = maxs - mins
    rng[rng == 0] = 1.0
    pts  = (pts - mins) / rng
    return pts - pts.mean(axis=0)


def _normalize_zscore(pts: np.ndarray) -> np.ndarray:
    """Z-score each spatial axis independently."""
    std = pts.std(axis=0)
    std[std == 0] = 1.0
    return (pts - pts.mean(axis=0)) / std


def _apply_norm(pts: np.ndarray, norm: str) -> np.ndarray:
    if norm == "three_cent":
        return _normalize_three_cent(pts)
    elif norm == "none":
        return pts - pts.mean(axis=0)          # only translate to origin
    elif norm == "bounding_box":
        return _normalize_bounding_box(pts)
    elif norm == "zscore":
        return _normalize_zscore(pts)
    else:
        raise ValueError(f"Unknown normalization: {norm}")


def _preprocess(traj: np.ndarray, n: int, norm: str) -> np.ndarray:
    pts = _resample_arc(traj, n)
    return _apply_norm(pts, norm)


# ─────────────────────────────────────────────────────────────────────────────
# Template construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_templates(train: list, n: int, norm: str,
                     n_templates, selection: str,
                     seed: int = 42) -> list:
    """
    Build a template library from training gestures.

    Returns a list of dicts with keys 'gesture_type' and 'preprocessed'.

    selection='centroid' computes the mean of all preprocessed trajectories
    per class and uses that as a single template — n_templates is ignored.
    """
    rng = np.random.default_rng(seed)

    # Group by class
    groups = defaultdict(list)
    for g in train:
        preprocessed = _preprocess(g["trajectory"], n, norm)
        groups[g["gesture_type"]].append(preprocessed)

    templates = []

    for gtype, preps in groups.items():
        if selection == "centroid":
            centroid = np.mean(preps, axis=0)
            templates.append({"gesture_type": gtype, "preprocessed": centroid})

        else:
            if n_templates == "all":
                chosen = preps
            else:
                k = min(int(n_templates), len(preps))
                if selection == "first":
                    chosen = preps[:k]
                elif selection == "random":
                    idx    = rng.choice(len(preps), size=k, replace=False)
                    chosen = [preps[i] for i in sorted(idx)]
                else:
                    raise ValueError(f"Unknown selection strategy: {selection}")

            for p in chosen:
                templates.append({"gesture_type": gtype, "preprocessed": p})

    return templates


# ─────────────────────────────────────────────────────────────────────────────
# Distance metrics between preprocessed trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-point Euclidean distance (original three-cent metric)."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Unconstrained DTW between two same-length preprocessed trajectories."""
    n = len(a)
    dtw = np.full((n + 1, n + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, n])


def _hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance: max(directed_H(a→b), directed_H(b→a))."""
    from scipy.spatial.distance import cdist
    D = cdist(a, b, metric="euclidean")
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def _dist(a: np.ndarray, b: np.ndarray, dist_type: str) -> float:
    if dist_type == "path_distance":
        return _path_distance(a, b)
    elif dist_type == "dtw":
        return _dtw_distance(a, b)
    elif dist_type == "hausdorff":
        return _hausdorff_distance(a, b)
    else:
        raise ValueError(f"Unknown distance: {dist_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Recognition: nearest template wins
# ─────────────────────────────────────────────────────────────────────────────

def _recognize(traj: np.ndarray, templates: list,
               n: int, norm: str, dist_type: str) -> int:
    candidate  = _preprocess(traj, n, norm)
    best_dist  = np.inf
    best_type  = -1
    for tmpl in templates:
        d = _dist(candidate, tmpl["preprocessed"], dist_type)
        if d < best_dist:
            best_dist = d
            best_type = tmpl["gesture_type"]
    return best_type


# ─────────────────────────────────────────────────────────────────────────────
# Two-phase cross-validated experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(gestures: list, cv_mode: str, val_fraction: float = 0.20):
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── Phase 1: HP selection on inner validation ─────────────────────────────
    hp_scores = defaultdict(list)

    for train, test, _ in cv_fn(gestures):
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_n, val_fraction)

        for n_resample in N_RESAMPLE_OPTIONS:
            for norm in NORM_OPTIONS:
                for n_tmpl in N_TEMPLATES_OPTIONS:
                    for selection in SELECTION_OPTIONS:
                        templates = _build_templates(
                            inner_train, n_resample, norm, n_tmpl, selection)

                        for dist_type in DISTANCE_OPTIONS:
                            y_true, y_pred = [], []
                            for g in inner_val:
                                pred = _recognize(g["trajectory"], templates,
                                                  n_resample, norm, dist_type)
                                y_true.append(g["gesture_type"])
                                y_pred.append(pred)
                            acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
                            hp  = (n_resample, norm, n_tmpl, selection, dist_type)
                            hp_scores[hp].append(acc)

    best_hp  = max(hp_scores, key=lambda h: np.mean(hp_scores[h]))
    best_val = float(np.mean(hp_scores[best_hp]))
    n_r, norm_b, n_tmpl_b, sel_b, dist_b = best_hp
    print(f"  Best HP → n_resample={n_r}, norm={norm_b}, n_templates={n_tmpl_b}, "
          f"selection={sel_b}, distance={dist_b}   val_acc={best_val:.4f}")

    # ── Phase 2: final evaluation with fixed best HP ──────────────────────────
    rows = []
    for train, test, fold_id in cv_fn(gestures):
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        test_n    = apply_normalizer(test,  mean, std)

        templates = _build_templates(train_n, n_r, norm_b, n_tmpl_b, sel_b)

        y_true, y_pred = [], []
        for g in test_n:
            pred = _recognize(g["trajectory"], templates, n_r, norm_b, dist_b)
            y_true.append(g["gesture_type"])
            y_pred.append(pred)

        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rows.append({
            "fold_id":      fold_id,
            "n_resample":   n_r,
            "normalization": norm_b,
            "n_templates":  n_tmpl_b,
            "selection":    sel_b,
            "distance":     dist_b,
            "val_acc":      round(best_val, 4),
            "test_acc":     round(acc, 4),
        })
        print(f"    Fold {fold_id}: test_acc={acc:.4f}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    datasets = {
        "domain1": load_data_domain_1(PATH_DOMAIN_1),
        "domain4": load_data_domain_4(PATH_DOMAIN_4),
    }

    all_rows = []
    total = len(datasets) * len(CV_MODES)
    done  = 0

    for domain_name, gestures in datasets.items():
        for cv_mode in CV_MODES:
            done += 1
            print(f"\n[{done}/{total}] Three-Cent ablation | {domain_name} | {cv_mode}")
            print("─" * 60)

            rows = run_ablation(gestures, cv_mode)
            for r in rows:
                r["domain"]  = domain_name
                r["cv_mode"] = cv_mode
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "three_cent_ablation.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'─' * 70}")
    print("SUMMARY — mean test accuracy per HP config")
    print("─" * 70)
    summary = (
        df.groupby(["domain", "cv_mode", "n_resample", "normalization",
                    "n_templates", "selection", "distance"])["test_acc"]
        .agg(["mean", "std"])
    )
    print(summary.to_string())
    print(f"\nSaved → {out_path}")
