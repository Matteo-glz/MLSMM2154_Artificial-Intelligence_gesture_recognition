"""
precompute_results.py
─────────────────────────────────────────────────────────────────────────────
Runs the FULL configuration grid in parallel, using ~90 % of all CPU cores.

How it works
────────────
  • There are 20 independent folds (10 per CV mode × 2 CV modes).
  • Each fold is dispatched to a separate worker process via joblib/loky.
  • On a 12-core / 24-thread machine all 20 folds run simultaneously
    → wall-clock time ≈ time for ONE fold  (~35 s without DTW, ~90 s with DTW)
    instead of 20 × that.

Grid covered  (mirrors main.py exactly)
────────────
  edit-distance : PCA × n_clusters × k_neighbors × compression
  DTW           : PCA × k_neighbors           (add --dtw flag)
  3-Cent        : PCA × n_points

Usage
─────
  python precompute_results.py --domain 1          # ~35-60 s
  python precompute_results.py --domain 1 --dtw    # +DTW ~90 s
  python precompute_results.py --domain 1 4        # both domains
  python precompute_results.py --jobs 8            # fix worker count
"""

from __future__ import annotations
import argparse, os, sys, time, pickle
from collections import defaultdict
from itertools   import product

import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster   import KMeans
from sklearn.metrics   import confusion_matrix as sk_cm
from joblib            import Parallel, delayed, cpu_count

# ── Make the project importable from worker processes ────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

RESULTS_DIR = os.path.join(_BASE, "results")

# ── Full grid (identical to main.py) ─────────────────────────────────────────
PCA_OPTIONS      = ["no_pca", 1, 2, 3]
K_NEIGHBORS      = [1, 3, 5, 7]
N_CLUSTERS_ED    = [5, 7, 9, 11, 13, 15, 17, 19, 21]
COMPRESSION_OPTS = [True, False]
N_POINTS_3CENT   = [16, 32, 64, 128, 256]
CV_MODES         = ["independent", "dependent"]


# ════════════════════════════════════════════════════════════════════════════
# Pure helper functions (must be importable / picklable in worker processes)
# ════════════════════════════════════════════════════════════════════════════

def _fast_resample(points: np.ndarray, n: int) -> np.ndarray:
    """
    Arc-length parameterised linear resampling.
    Equivalent to three_cent._resample but 20-50× faster
    (uses scipy interp1d instead of np.insert in a loop).
    """
    if len(points) == 1:
        return np.tile(points[0], (n, 1))
    seg   = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum   = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return np.tile(points[0], (n, 1))
    t_orig = cum / total
    t_new  = np.linspace(0.0, 1.0, n)
    out    = np.empty((n, points.shape[1]))
    for d in range(points.shape[1]):
        out[:, d] = interp1d(t_orig, points[:, d],
                             kind="linear", assume_sorted=True)(t_new)
    return out


def _preprocess_3cent(points: np.ndarray, n: int) -> np.ndarray:
    pts    = _fast_resample(points, n)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    if length > 0:
        pts = pts / length
    return pts - pts.mean(axis=0)


def _path_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _symbolic(traj: np.ndarray, km: KMeans) -> str:
    return "".join(chr(65 + c) for c in km.predict(traj))


def _compress(s: str) -> str:
    if not s:
        return s
    out = [s[0]]
    for c in s[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def _majority(votes: list) -> int:
    from collections import Counter
    return Counter(votes).most_common(1)[0][0]


def _apply_pca(train, test, pca_opt):
    if pca_opt == "no_pca":
        return train, test
    from data_preparation import fit_pca_per_gesture, apply_pca_per_gesture
    pcas = fit_pca_per_gesture(train, pca_opt)
    return apply_pca_per_gesture(train, pcas), apply_pca_per_gesture(test, pcas)


# ════════════════════════════════════════════════════════════════════════════
# Worker function — one fold
# ════════════════════════════════════════════════════════════════════════════

def _process_fold(
    train: list,
    test:  list,
    fold_id,
    cv_mode: str,
    run_dtw: bool,
) -> dict:
    """
    Compute every configuration for ONE (train, test) fold.

    Returns
    -------
    dict with keys "cv_mode", "fold_id",
         "ed"  : {(pca, nc, k, comp) → (y_true, y_pred)},
         "dtw" : {(pca, k)           → (y_true, y_pred)},
         "tc"  : {(pca, np_)         → (y_true, y_pred)},
    """
    # Late imports so each worker process initialises cleanly
    from data_preparation  import fit_normalizer, apply_normalizer
    from utils_algorithms import edit_distance_fast, compute_dtw_distance_c_speed

    mu, sig   = fit_normalizer(train)
    train_n   = apply_normalizer(train, mu, sig)
    test_n    = apply_normalizer(test,  mu, sig)
    true_lbls = [g["gesture_type"] for g in test_n]

    ed_fold  = {}   # (pca, nc, k, comp) → (list[int], list[int])
    dtw_fold = {}   # (pca, k)           → (list[int], list[int])
    tc_fold  = {}   # (pca, np_)         → (list[int], list[int])

    for pca_opt in PCA_OPTIONS:
        train_p, test_p = _apply_pca(train_n, test_n, pca_opt)
        train_pts = np.vstack([g["trajectory"] for g in train_p])
        tr_gt     = [g["gesture_type"] for g in train_p]

        # ── Edit-distance ─────────────────────────────────────────────────
        for nc in N_CLUSTERS_ED:
            km      = KMeans(n_clusters=nc, n_init=1, random_state=42,
                             max_iter=100).fit(train_pts)
            tr_raw  = [_symbolic(g["trajectory"], km) for g in train_p]
            te_raw  = [_symbolic(g["trajectory"], km) for g in test_p]
            tr_comp = [_compress(s) for s in tr_raw]
            te_comp = [_compress(s) for s in te_raw]

            for use_comp, te_seqs, tr_seqs in [
                    (True,  te_comp, tr_comp),
                    (False, te_raw,  tr_raw)]:
                # Compute pairwise distances once; sweep k with simple slicing
                cache = []
                for t_seq, t_lbl in zip(te_seqs, true_lbls):
                    dists = sorted(
                        [(edit_distance_fast(t_seq, r), rgt)
                         for r, rgt in zip(tr_seqs, tr_gt)],
                        key=lambda x: x[0])
                    cache.append((t_lbl, dists))

                for k in K_NEIGHBORS:
                    y_t = [lbl  for lbl, _ in cache]
                    y_p = [_majority([gt for _, gt in nn[:k]])
                           for _, nn in cache]
                    ed_fold[(pca_opt, nc, k, use_comp)] = (y_t, y_p)

        # ── DTW ───────────────────────────────────────────────────────────
        if run_dtw:
            dtw_cache = []
            for tg in test_p:
                d = sorted(
                    [(compute_dtw_distance_c_speed(tg["trajectory"],
                                                   trg["trajectory"]),
                      trg["gesture_type"]) for trg in train_p],
                    key=lambda x: x[0])
                dtw_cache.append((tg["gesture_type"], d))
            for k in K_NEIGHBORS:
                y_t = [lbl for lbl, _ in dtw_cache]
                y_p = [_majority([gt for _, gt in nn[:k]])
                       for _, nn in dtw_cache]
                dtw_fold[(pca_opt, k)] = (y_t, y_p)

        # ── 3-Cent ────────────────────────────────────────────────────────
        for n_pts in N_POINTS_3CENT:
            tmpls = [(g["gesture_type"],
                      _preprocess_3cent(g["trajectory"], n_pts))
                     for g in train_p]
            y_t, y_p = [], []
            for tg in test_p:
                cand = _preprocess_3cent(tg["trajectory"], n_pts)
                pred = min(tmpls, key=lambda x: _path_dist(cand, x[1]))[0]
                y_t.append(tg["gesture_type"]); y_p.append(pred)
            tc_fold[(pca_opt, n_pts)] = (y_t, y_p)

    return {"cv_mode": cv_mode, "fold_id": fold_id,
            "ed": ed_fold, "dtw": dtw_fold, "tc": tc_fold}


# ════════════════════════════════════════════════════════════════════════════
# Aggregation helper
# ════════════════════════════════════════════════════════════════════════════

def _aggregate(fold_results: list, gesture_types: list) -> dict:
    """
    Combine per-fold results into per-config accuracy + confusion matrices.
    Returns {method: {cv_mode: {config_key: result_dict}}}
    """
    # Accumulators: {cv_mode: {config_key: {fold_acc, true, pred}}}
    acc: dict = {cv: {"ed": defaultdict(lambda: {"fold_acc":[],"true":[],"pred":[]}),
                      "dtw": defaultdict(lambda: {"fold_acc":[],"true":[],"pred":[]}),
                      "tc":  defaultdict(lambda: {"fold_acc":[],"true":[],"pred":[]})}
                 for cv in CV_MODES}

    for fr in fold_results:
        cv = fr["cv_mode"]
        for key, (y_t, y_p) in fr["ed"].items():
            d = acc[cv]["ed"][key]
            fa = float(np.mean(np.array(y_t) == np.array(y_p)))
            d["fold_acc"].append(fa); d["true"].extend(y_t); d["pred"].extend(y_p)
        for key, (y_t, y_p) in fr["dtw"].items():
            d = acc[cv]["dtw"][key]
            fa = float(np.mean(np.array(y_t) == np.array(y_p)))
            d["fold_acc"].append(fa); d["true"].extend(y_t); d["pred"].extend(y_p)
        for key, (y_t, y_p) in fr["tc"].items():
            d = acc[cv]["tc"][key]
            fa = float(np.mean(np.array(y_t) == np.array(y_p)))
            d["fold_acc"].append(fa); d["true"].extend(y_t); d["pred"].extend(y_p)

    def _pack(d):
        cm = sk_cm(d["true"], d["pred"], labels=gesture_types)
        return {"per_fold_accuracy": d["fold_acc"],
                "mean_accuracy":     float(np.mean(d["fold_acc"])),
                "std_accuracy":      float(np.std(d["fold_acc"])),
                "confusion_matrix":  cm,
                "gesture_types":     gesture_types}

    out = {"edit-distance": {}, "dtw": {}, "three-cent": {}}
    for cv in CV_MODES:
        out["edit-distance"][cv] = {k: _pack(v) for k,v in acc[cv]["ed"].items()}
        if any(acc[cv]["dtw"].values()):
            out["dtw"][cv] = {k: _pack(v) for k,v in acc[cv]["dtw"].items()}
        out["three-cent"][cv]    = {k: _pack(v) for k,v in acc[cv]["tc"].items()}
    return out


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def precompute_domain(gestures: list, domain_name: str,
                      run_dtw: bool, n_jobs: int) -> dict:
    from data_splitting import user_independent_cv, user_dependent_cv

    gesture_types = sorted({g["gesture_type"] for g in gestures})
    label_map     = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                     for g in gestures}

    # Build the flat list of fold jobs (both CV modes)
    jobs = []
    for cv_mode, cv_fn in [("independent", user_independent_cv),
                            ("dependent",   user_dependent_cv)]:
        for train, test, fold_id in cv_fn(gestures):
            jobs.append((train, test, fold_id, cv_mode, run_dtw))

    total = len(jobs)
    print(f"  Dispatching {total} folds across {n_jobs} workers...")

    t0 = time.time()
    fold_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_process_fold)(train, test, fid, cv_mode, run_dtw)
        for train, test, fid, cv_mode, run_dtw in jobs
    )
    elapsed = time.time() - t0
    print(f"  All folds done in {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    aggregated = _aggregate(fold_results, gesture_types)

    output = {
        "meta": {"domain": domain_name, "gesture_types": gesture_types,
                 "label_map": label_map, "n_gestures": len(gestures)},
        **aggregated,
    }
    return output


def main():
    ap = argparse.ArgumentParser(
        description="Pre-compute all results in parallel for the interactive dashboard")
    ap.add_argument("--domain", type=int, nargs="+", default=[1], choices=[1, 4])
    ap.add_argument("--dtw",  action="store_true",
                    help="Include DTW (adds ~60 s per domain)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="Number of parallel workers (0 = auto: 90%% of CPUs)")
    args = ap.parse_args()

    n_logical = cpu_count()
    n_jobs    = args.jobs if args.jobs > 0 else max(1, int(n_logical * 0.9))
    print(f"Machine: {n_logical} logical CPUs  ->  using {n_jobs} workers")

    n_ed  = len(PCA_OPTIONS)*len(N_CLUSTERS_ED)*len(K_NEIGHBORS)*len(COMPRESSION_OPTS)
    n_tc  = len(PCA_OPTIONS)*len(N_POINTS_3CENT)
    n_dtw = len(PCA_OPTIONS)*len(K_NEIGHBORS) if args.dtw else 0
    print(f"Grid: {n_ed} edit-dist + {n_tc} 3-cent"
          + (f" + {n_dtw} DTW" if args.dtw else " (DTW skipped)"))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.dirname(os.path.abspath(__file__))

    for domain in args.domain:
        print(f"\n{'='*60}\n  DOMAIN {domain}\n{'='*60}")

        if domain == 1:
            from data_loading import load_data_domain_1
            data_dir = os.path.join(base, "GestureData_Mons",
                                    "GestureDataDomain1_Mons", "Domain1_csv")
            gestures = load_data_domain_1(data_dir)
            dname    = "Domain 1 (digits 0-9)"
        else:
            from data_loading import load_data_domain_4
            data_dir = os.path.join(base, "GestureData_Mons", "GestureDataDomain4_Mons")
            gestures = load_data_domain_4(data_dir)
            dname    = "Domain 4 (3D shapes)"

        print(f"  {len(gestures)} gestures loaded.")
        results  = precompute_domain(gestures, dname, args.dtw, n_jobs)

        out_path = os.path.join(RESULTS_DIR, f"precomputed_domain{domain}.pkl")
        with open(out_path, "wb") as fh:
            pickle.dump(results, fh)
        print(f"  Saved -> {out_path}")

        # ── Quick summary ─────────────────────────────────────────────────
        print("\n  Best configs found:")
        for method, mkey in [("edit-distance","edit-distance"),
                              ("DTW","dtw"), ("3-Cent","three-cent")]:
            if not results.get(mkey):
                continue
            for cv_mode in CV_MODES:
                if cv_mode not in results[mkey]:
                    continue
                cv_data  = results[mkey][cv_mode]
                best_key = max(cv_data, key=lambda k: cv_data[k]["mean_accuracy"])
                best     = cv_data[best_key]
                print(f"    {method:16s} / {cv_mode:12s}: "
                      f"{best['mean_accuracy']:.2%} ± {best['std_accuracy']:.2%}"
                      f"  config={best_key}")


if __name__ == "__main__":
    main()
