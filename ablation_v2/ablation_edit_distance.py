"""
ablation_edit_distance.py
─────────────────────────────────────────────────────────────────────────────
Ablation study for Edit-Distance gesture recognition.

Hyperparameters swept:
  - n_resample    : [32, 64, 128]             — resample trajectory before clustering
  - codebook_size : [8, 16, 32, 64]           — number of discrete symbols
  - clustering    : ['kmeans', 'kmedoids', 'uniform_grid'] — codebook method
  - sub_cost      : ['uniform', 'proportional'] — substitution cost in edit distance
  - compression   : [True, False]             — remove consecutive duplicate symbols
  - k             : [1, 3, 5, 7]             — kNN neighbours

Protocol: 2-phase cross-validation identical to main pipeline.

Outputs: ablation_v2/ablation_v2_results/edit_distance_ablation.csv

NOTE: k-medoids is O(n²) per iteration; large codebook_size × large n_resample
      will be slow. Consider reducing the grid for faster runs.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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
CODEBOOK_OPTIONS     = [8, 16, 32, 64]
CLUSTERING_OPTIONS   = ["kmeans", "kmedoids", "uniform_grid"]
SUB_COST_OPTIONS     = ["uniform", "proportional"]
COMPRESSION_OPTIONS  = [True, False]
K_OPTIONS            = [1, 3, 5, 7]
CV_MODES             = ["dependent", "independent"]

# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def _resample_linear(traj: np.ndarray, n: int) -> np.ndarray:
    old_n = len(traj)
    if old_n == n:
        return traj.astype(np.float64)
    old_idx = np.arange(old_n, dtype=np.float64)
    new_idx = np.linspace(0, old_n - 1, n)
    return np.stack([np.interp(new_idx, old_idx, traj[:, d])
                     for d in range(traj.shape[1])], axis=1)


def _apply_resample(gestures: list, n_resample: int) -> list:
    result = []
    for g in gestures:
        g_copy = g.copy()
        g_copy["trajectory"] = _resample_linear(g["trajectory"], n_resample)
        result.append(g_copy)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Codebook fitting: k-means, k-medoids, uniform grid
# ─────────────────────────────────────────────────────────────────────────────

class Codebook:
    """
    Wraps a set of cluster centers and an optional substitution cost matrix.

    cost_matrix[i, j] — cost to substitute symbol i with symbol j.
    'uniform': cost = 0 if i==j else 1.
    'proportional': cost = normalized centroid distance (in [0, 1]).
    """
    def __init__(self, centers: np.ndarray):
        self.centers = centers
        n = len(centers)
        D = cdist(centers, centers, metric="euclidean")
        max_d = D.max()
        # Proportional cost: distance-normalized; diagonal is 0
        self.proportional_cost = D / max_d if max_d > 0 else D

    def predict(self, points: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest center (integer index)."""
        return np.argmin(cdist(points, self.centers, metric="euclidean"), axis=1)


def _fit_kmeans(points: np.ndarray, n_clusters: int) -> Codebook:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    km.fit(points)
    return Codebook(km.cluster_centers_)


def _fit_kmedoids(points: np.ndarray, n_clusters: int,
                  max_iter: int = 50, seed: int = 42) -> Codebook:
    """
    Simple PAM-style k-medoids.
    - Initialize medoids randomly.
    - Assign points to nearest medoid (by Euclidean distance).
    - Update each medoid to the cluster member minimising intra-cluster distance.
    - Stop when medoids no longer change.
    """
    rng = np.random.default_rng(seed)
    n   = len(points)
    n_clusters = min(n_clusters, n)

    # Pre-compute full pairwise distance matrix (reused every iteration)
    D_full = cdist(points, points, metric="euclidean")

    medoid_idx = rng.choice(n, n_clusters, replace=False)

    for _ in range(max_iter):
        # Assignment step
        assignments = np.argmin(D_full[:, medoid_idx], axis=1)

        new_medoids = medoid_idx.copy()
        for c in range(n_clusters):
            cluster_pts = np.where(assignments == c)[0]
            if len(cluster_pts) == 0:
                continue
            # Medoid: cluster member minimising sum of distances to others in cluster
            intra = D_full[np.ix_(cluster_pts, cluster_pts)]
            new_medoids[c] = cluster_pts[np.argmin(intra.sum(axis=1))]

        if np.all(new_medoids == medoid_idx):
            break
        medoid_idx = new_medoids

    return Codebook(points[medoid_idx])


def _fit_uniform_grid(points: np.ndarray, n_clusters: int) -> Codebook:
    """
    Build a regular grid of centers that spans the bounding box of the data.

    cells_per_dim is chosen so the total number of grid points is close to
    n_clusters. The closest n_clusters grid centers to the data centroid are
    kept (discards corners that may never be activated).
    """
    n_dims = points.shape[1]
    cells_per_dim = max(2, int(round(n_clusters ** (1.0 / n_dims))))

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    axes = [np.linspace(mins[d], maxs[d], cells_per_dim) for d in range(n_dims)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(n_dims, -1).T

    # Keep the n_clusters centers closest to the data centroid
    data_center = points.mean(axis=0)
    order = np.argsort(np.linalg.norm(grid - data_center, axis=1))
    centers = grid[order[:n_clusters]]

    return Codebook(centers)


def _get_codebook(points: np.ndarray, n_clusters: int, method: str) -> Codebook:
    if method == "kmeans":
        return _fit_kmeans(points, n_clusters)
    elif method == "kmedoids":
        return _fit_kmedoids(points, n_clusters)
    elif method == "uniform_grid":
        return _fit_uniform_grid(points, n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic transformation & edit distance
# ─────────────────────────────────────────────────────────────────────────────

def _encode_gesture(g: dict, codebook: Codebook, compress: bool) -> dict:
    """Map trajectory frames to a symbol sequence (and optionally compress)."""
    indices = codebook.predict(g["trajectory"])  # integer array

    if compress:
        compressed = [indices[0]]
        for idx in indices[1:]:
            if idx != compressed[-1]:
                compressed.append(idx)
        indices = np.array(compressed)

    g_copy = g.copy()
    g_copy["symbol_seq"] = indices  # np.ndarray of int
    return g_copy


def _encode_gestures(gestures: list, codebook: Codebook, compress: bool) -> list:
    return [_encode_gesture(g, codebook, compress) for g in gestures]


def _weighted_edit_distance(s1: np.ndarray, s2: np.ndarray,
                            cost_matrix: np.ndarray) -> float:
    """
    Edit distance with per-substitution costs from cost_matrix.

    cost_matrix[i, j] = cost to substitute symbol i with symbol j.
    Insertion / deletion always cost 1.
    """
    n, m = len(s1), len(s2)
    prev = np.arange(m + 1, dtype=np.float64)
    curr = np.zeros(m + 1, dtype=np.float64)

    for i in range(1, n + 1):
        curr[0] = float(i)
        for j in range(1, m + 1):
            sub = cost_matrix[s1[i - 1], s2[j - 1]]
            curr[j] = min(prev[j] + 1.0,      # deletion
                          curr[j - 1] + 1.0,   # insertion
                          prev[j - 1] + sub)    # substitution
        prev[:] = curr[:]

    return float(prev[m])


def _predict_knn(test_g: dict, train_gestures: list,
                 k: int, sub_cost: str,
                 codebook: Codebook) -> int:
    """kNN prediction using weighted or uniform edit distance."""
    s1 = test_g["symbol_seq"]

    if sub_cost == "uniform":
        cost_fn = lambda s2: _weighted_edit_distance(
            s1, s2,
            np.where(np.eye(len(codebook.centers), dtype=bool), 0.0, 1.0)
        )
    else:  # proportional
        cost_fn = lambda s2: _weighted_edit_distance(
            s1, s2, codebook.proportional_cost
        )

    distances = sorted(
        [(cost_fn(g["symbol_seq"]), g["gesture_type"]) for g in train_gestures],
        key=lambda x: x[0],
    )
    neighbors = [label for _, label in distances[:k]]
    return Counter(neighbors).most_common(1)[0][0]


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
            it_r = _apply_resample(inner_train, n_resample)
            iv_r = _apply_resample(inner_val,   n_resample)
            all_train_pts = np.vstack([g["trajectory"] for g in it_r])

            for n_clusters in CODEBOOK_OPTIONS:
                for clustering in CLUSTERING_OPTIONS:
                    codebook = _get_codebook(all_train_pts, n_clusters, clustering)

                    for compress in COMPRESSION_OPTIONS:
                        it_enc = _encode_gestures(it_r, codebook, compress)
                        iv_enc = _encode_gestures(iv_r, codebook, compress)

                        for sub_cost in SUB_COST_OPTIONS:
                            for k in K_OPTIONS:
                                y_true, y_pred = [], []
                                for g in iv_enc:
                                    pred = _predict_knn(g, it_enc, k, sub_cost, codebook)
                                    y_true.append(g["gesture_type"])
                                    y_pred.append(pred)
                                acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
                                hp = (n_resample, n_clusters, clustering,
                                      compress, sub_cost, k)
                                hp_scores[hp].append(acc)

    best_hp  = max(hp_scores, key=lambda h: np.mean(hp_scores[h]))
    best_val = float(np.mean(hp_scores[best_hp]))
    n_r, n_cl, clust, comp, sub, k_b = best_hp
    print(f"  Best HP → n_resample={n_r}, codebook={n_cl}, clustering={clust}, "
          f"compress={comp}, sub_cost={sub}, k={k_b}   val_acc={best_val:.4f}")

    # ── Phase 2: final evaluation with fixed best HP ──────────────────────────
    rows = []
    for train, test, fold_id in cv_fn(gestures):
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        test_n    = apply_normalizer(test,  mean, std)

        tr_r = _apply_resample(train_n, n_r)
        te_r = _apply_resample(test_n,  n_r)
        all_train_pts = np.vstack([g["trajectory"] for g in tr_r])

        codebook = _get_codebook(all_train_pts, n_cl, clust)
        tr_enc   = _encode_gestures(tr_r, codebook, comp)
        te_enc   = _encode_gestures(te_r, codebook, comp)

        y_true, y_pred = [], []
        for g in te_enc:
            pred = _predict_knn(g, tr_enc, k_b, sub, codebook)
            y_true.append(g["gesture_type"])
            y_pred.append(pred)

        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        rows.append({
            "fold_id":      fold_id,
            "n_resample":   n_r,
            "codebook_size": n_cl,
            "clustering":   clust,
            "compression":  comp,
            "sub_cost":     sub,
            "k":            k_b,
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
            print(f"\n[{done}/{total}] Edit-Distance ablation | {domain_name} | {cv_mode}")
            print("─" * 60)

            rows = run_ablation(gestures, cv_mode)
            for r in rows:
                r["domain"]  = domain_name
                r["cv_mode"] = cv_mode
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "edit_distance_ablation.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'─' * 70}")
    print("SUMMARY — mean test accuracy per HP config")
    print("─" * 70)
    summary = (
        df.groupby(["domain", "cv_mode", "n_resample", "codebook_size",
                    "clustering", "compression", "sub_cost", "k"])["test_acc"]
        .agg(["mean", "std"])
    )
    print(summary.to_string())
    print(f"\nSaved → {out_path}")
