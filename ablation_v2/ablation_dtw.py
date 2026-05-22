"""
ablation_dtw.py
─────────────────────────────────────────────────────────────────────────────
Ablation study for Dynamic Time Warping (DTW) gesture recognition.

Hyperparameters swept:
  - n_resample    : [32, 64, 128]            — fixed-length resampling before DTW
  - window_frac   : [None, 0.10, 0.20, 0.30] — Sakoe-Chiba band (None = unconstrained)
  - metric        : ['euclidean', 'manhattan', 'cosine'] — inner frame distance
  - features      : ['raw', 'velocity', 'acceleration', 'pca2'] — feature type
  - derivative    : [False, True]             — apply 1st-order diff before DTW
  - k             : [1, 3, 5, 7]             — kNN neighbours

Protocol: 2-phase cross-validation identical to main pipeline.
  Phase 1 — inner-val HP selection across all folds.
  Phase 2 — final evaluation with fixed best HP.

Outputs: ablation_v2/ablation_v2_results/dtw_ablation.csv

NOTE: The full cross-product (3×4×4×3×2×4 = 1152 HP combos) is expensive.
      Comment out options you don't need to reduce runtime.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

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
# Hyperparameter grid  (comment out rows to reduce runtime)
# ─────────────────────────────────────────────────────────────────────────────

N_RESAMPLE_OPTIONS  = [32, 64]#, 128]
WINDOW_FRAC_OPTIONS = [None, 0.10, 0.20, 0.30]
METRIC_OPTIONS      = ["euclidean", "manhattan", "cosine"]
FEATURE_OPTIONS     = ["raw", "velocity", "acceleration", "pca2"]
DERIVATIVE_OPTIONS  = [False, True]
K_OPTIONS           = [1, 3, 5, 7]
CV_MODES            = ["dependent", "independent"]

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample_linear(traj: np.ndarray, n: int) -> np.ndarray:
    """Resample a trajectory to n points via per-axis linear interpolation."""
    old_n = len(traj)
    if old_n == n:
        return traj.astype(np.float64)
    old_idx = np.arange(old_n, dtype=np.float64)
    new_idx = np.linspace(0, old_n - 1, n)
    return np.stack([np.interp(new_idx, old_idx, traj[:, d])
                     for d in range(traj.shape[1])], axis=1)


def _fit_pca2(gestures: list, n_resample: int) -> dict:
    """Fit per-class PCA (2 components) on resampled trajectories — train only."""
    groups = defaultdict(list)
    for g in gestures:
        traj = _resample_linear(g["trajectory"], n_resample)
        groups[g["gesture_type"]].append(traj)
    pcas = {}
    for gtype, trajs in groups.items():
        pca = PCA(n_components=2)
        pca.fit(np.vstack(trajs))
        pcas[gtype] = pca
    return pcas


def _apply_features(gestures: list, feature_type: str,
                    n_resample: int, pca_models: dict = None) -> list:
    """
    Apply resampling + feature extraction to a gesture list.

    For 'velocity' / 'acceleration' the diff is taken after resampling so the
    number of output frames is n_resample-1 / n_resample-2 respectively.
    """
    result = []
    for g in gestures:
        traj = _resample_linear(g["trajectory"], n_resample)

        if feature_type == "raw":
            feat = traj
        elif feature_type == "velocity":
            feat = np.diff(traj, axis=0)
        elif feature_type == "acceleration":
            feat = np.diff(traj, n=2, axis=0)
        elif feature_type == "pca2":
            pca = pca_models[g["gesture_type"]]
            feat = pca.transform(traj)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        if len(feat) < 2:
            feat = traj  # fallback: too short after diff

        g_copy = g.copy()
        g_copy["trajectory"] = feat
        result.append(g_copy)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DTW with Sakoe-Chiba band and pluggable inner metric
# ─────────────────────────────────────────────────────────────────────────────

def _dtw(seq1: np.ndarray, seq2: np.ndarray,
         window_frac, metric: str, derivative: bool) -> float:
    """
    DTW distance with optional Sakoe-Chiba constraint and choice of inner metric.

    scipy.cdist pre-computes the full pairwise cost matrix; the DP then uses the
    band mask to avoid degenerate alignments.
    """
    if derivative:
        seq1 = np.diff(seq1, axis=0)
        seq2 = np.diff(seq2, axis=0)
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0

    n, m = len(seq1), len(seq2)
    w = int(max(n, m) * window_frac) if window_frac is not None else max(n, m)

    D = cdist(seq1, seq2, metric=metric)

    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m + 1, i + w + 1)
        for j in range(j_lo, j_hi):
            dtw_mat[i, j] = D[i - 1, j - 1] + min(
                dtw_mat[i - 1, j],
                dtw_mat[i, j - 1],
                dtw_mat[i - 1, j - 1],
            )

    return float(dtw_mat[n, m])


def _majority_vote(neighbors: list) -> int:
    return Counter(n[1] for n in neighbors).most_common(1)[0][0]


def _build_cache(train: list, query: list,
                 window_frac, metric: str, derivative: bool) -> list:
    """Return sorted (dist, label) lists for each query gesture."""
    cache = []
    for qg in query:
        dists = sorted(
            [(_dtw(qg["trajectory"], rg["trajectory"],
                   window_frac, metric, derivative),
              rg["gesture_type"])
             for rg in train],
            key=lambda x: x[0],
        )
        cache.append((qg["gesture_type"], dists))
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Two-phase cross-validated experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(gestures: list, cv_mode: str, val_fraction: float = 0.20):
    cv_fn = user_dependent_cv if cv_mode == "dependent" else user_independent_cv

    # ── Phase 1: HP selection on inner validation ─────────────────────────────
    hp_scores = defaultdict(list)

    _total_p1 = (len(N_RESAMPLE_OPTIONS) * len(FEATURE_OPTIONS) *
                 len(WINDOW_FRAC_OPTIONS) * len(METRIC_OPTIONS) * len(DERIVATIVE_OPTIONS))
    _fold_i = 0
    for train, test, _ in cv_fn(gestures):
        _fold_i += 1
        print(f"  [Phase 1] Fold {_fold_i} — {_total_p1} cache builds to evaluate")
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        inner_train, inner_val = inner_val_split(train_n, val_fraction)

        _combo_i = 0
        for n_resample in N_RESAMPLE_OPTIONS:
            for feature_type in FEATURE_OPTIONS:
                pca_models = None
                if feature_type == "pca2":
                    pca_models = _fit_pca2(inner_train, n_resample)

                it_feat = _apply_features(inner_train, feature_type, n_resample, pca_models)
                iv_feat = _apply_features(inner_val,   feature_type, n_resample, pca_models)

                for window_frac in WINDOW_FRAC_OPTIONS:
                    for metric in METRIC_OPTIONS:
                        for derivative in DERIVATIVE_OPTIONS:
                            _combo_i += 1
                            print(f"    [{_combo_i}/{_total_p1}] n_r={n_resample} "
                                  f"feat={feature_type} win={window_frac} "
                                  f"metric={metric} deriv={derivative}",
                                  end="\r", flush=True)
                            cache = _build_cache(it_feat, iv_feat,
                                                 window_frac, metric, derivative)
                            for k in K_OPTIONS:
                                y_true = [lbl for lbl, _ in cache]
                                y_pred = [_majority_vote(d[:k]) for _, d in cache]
                                acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
                                hp = (n_resample, feature_type,
                                      window_frac, metric, derivative, k)
                                hp_scores[hp].append(acc)
        print()  # newline after \r progress line

    best_hp  = max(hp_scores, key=lambda h: np.mean(hp_scores[h]))
    best_val = float(np.mean(hp_scores[best_hp]))
    n_r, feat, win, met, deriv, k_b = best_hp
    print(f"  Best HP → n_resample={n_r}, feature={feat}, window={win}, "
          f"metric={met}, derivative={deriv}, k={k_b}   val_acc={best_val:.4f}")

    # ── Phase 2: final evaluation with fixed best HP ──────────────────────────
    rows = []
    for train, test, fold_id in cv_fn(gestures):
        print(f"  [Phase 2] Fold {fold_id} — testing best HP "
              f"(n_r={n_r} feat={feat} win={win} metric={met} deriv={deriv} k={k_b})")
        mean, std = fit_normalizer(train)
        train_n   = apply_normalizer(train, mean, std)
        test_n    = apply_normalizer(test,  mean, std)

        pca_models = None
        if feat == "pca2":
            pca_models = _fit_pca2(train_n, n_r)

        tr_feat = _apply_features(train_n, feat, n_r, pca_models)
        te_feat = _apply_features(test_n,  feat, n_r, pca_models)

        cache  = _build_cache(tr_feat, te_feat, win, met, deriv)
        y_true = [lbl for lbl, _ in cache]
        y_pred = [_majority_vote(d[:k_b]) for _, d in cache]
        acc    = float(np.mean(np.array(y_true) == np.array(y_pred)))

        rows.append({
            "fold_id":      fold_id,
            "n_resample":   n_r,
            "feature":      feat,
            "window_frac":  str(win),
            "metric":       met,
            "derivative":   deriv,
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
            print(f"\n[{done}/{total}] DTW ablation | {domain_name} | {cv_mode}")
            print("─" * 60)

            rows = run_ablation(gestures, cv_mode)
            for r in rows:
                r["domain"]  = domain_name
                r["cv_mode"] = cv_mode
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(OUT_DIR, "dtw_ablation.csv")
    df.to_csv(out_path, index=False)

    print(f"\n{'─' * 70}")
    print("SUMMARY — mean test accuracy per HP config")
    print("─" * 70)
    summary = (
        df.groupby(["domain", "cv_mode", "n_resample", "feature",
                    "window_frac", "metric", "derivative", "k"])["test_acc"]
        .agg(["mean", "std"])
    )
    print(summary.to_string())
    print(f"\nSaved → {out_path}")
