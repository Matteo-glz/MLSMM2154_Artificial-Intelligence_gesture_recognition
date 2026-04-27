"""
main_optimized.py
-----------------------------------------------------------------------------
100 % CPU utilisation via fine-grained task parallelism.

Problem with fold-level dispatch (old approach)
------------------------------------------------
  20 folds, 24 cores -> cores go idle as folds finish unevenly.
  predict_gesture_type_knn() recomputes edit distances for EVERY k value.

What this file does instead
----------------------------
  * Decomposes work into the smallest independent units:
      ED : (fold x PCA x n_clusters) -> 720 tasks per CV mode
      TC : (fold x PCA)              ->  80 tasks per CV mode
      DTW: (fold x PCA)              ->  80 tasks per CV mode
  * With 24 workers and 720 tasks, every core stays busy the entire time.
  * Within each task, edit distances are computed ONCE then all k values
    are answered by slicing the sorted list  (4x faster than before).
  * KMeans uses n_init=1 (same quality for batch evaluation, 10x faster fit).
  * Uses ALL logical CPUs (cpu_count()).

Progress display
----------------
  A tqdm bar shows:  tasks done | tasks left | ETA | running best accuracy

Usage
-----
  python main_optimized.py                      # domain 1+4, ed+tc
  python main_optimized.py --domain 1           # domain 1 only
  python main_optimized.py --method ed tc dtw   # include DTW (slow)
  python main_optimized.py --cv dependent       # one CV mode only
  python main_optimized.py --jobs 8             # override worker count
"""

from __future__ import annotations
import argparse, os, time
from collections  import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.cluster  import KMeans
from sklearn.metrics  import confusion_matrix
from tqdm             import tqdm

# Project imports
from data_loading    import load_data_domain_1, load_data_domain_4
from data_splitting  import user_dependent_cv, user_independent_cv
from data_preparation import (fit_normalizer, apply_normalizer,
                               fit_pca_per_gesture, apply_pca_per_gesture)
from utils_algorithms    import (edit_distance_fast,
                                  compute_dtw_distance_c_speed)
from baseline_three_cent import build_templates, recognize
from utils_assessment    import majority_vote
from utils_saving        import save_results

# -- Hyper-parameter grid -----------------------------------------------------
PCA_OPTIONS      = ["no_pca", 1, 2, 3]
K_NEIGHBORS      = [1, 3, 5, 7]
N_CLUSTERS_ED    = [5, 7, 9, 11, 13, 15, 17, 19, 21]
COMPRESSION_OPTS = [True, False]
N_POINTS_3CENT   = [16, 32, 64, 128, 256]
CV_MODES         = ["dependent", "independent"]

_BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Shared PCA helper (runs inside worker processes)
# ============================================================================

def _apply_pca(train_n, test_n, pca_opt):
    if pca_opt == "no_pca":
        return train_n, test_n
    from data_preparation import fit_pca_per_gesture, apply_pca_per_gesture
    pcas = fit_pca_per_gesture(train_n, pca_opt)
    return apply_pca_per_gesture(train_n, pcas), apply_pca_per_gesture(test_n, pcas)


# ============================================================================
# WORKER: Edit-distance - one (fold x PCA x n_clusters) task
# ============================================================================

def _task_ed(train_n, test_n, fold_id, cv_mode, pca_opt, nc):
    """
    Compute edit-distance k-NN accuracy for all (k, compression) combinations
    for a single (fold, PCA option, n_clusters) triple.

    Key optimisation: edit distances are computed ONCE per (compression type),
    the sorted neighbour list is then sliced for each k - no recomputation.

    Returns
    -------
    dict with keys: fold_id, cv_mode, pca_opt, nc,
                    results: {(k, use_comp): (y_true, y_pred)}
    """
    train_p, test_p = _apply_pca(train_n, test_n, pca_opt)

    # -- KMeans: n_init=1 is 10x faster than n_init=10 for batch evaluation --
    all_train_pts = np.vstack([g["trajectory"] for g in train_p])
    km = KMeans(n_clusters=nc, n_init=1, random_state=42,
                max_iter=100).fit(all_train_pts)

    # -- Symbolic transformation -------------------------------------------
    def _to_seqs(gestures):
        raw_seqs, comp_seqs, labels = [], [], []
        for g in gestures:
            raw = "".join(chr(65 + c) for c in km.predict(g["trajectory"]))
            # run-length compression
            cmp = raw[0] if raw else ""
            for ch in raw[1:]:
                if ch != cmp[-1]:
                    cmp += ch
            raw_seqs.append(raw)
            comp_seqs.append(cmp)
            labels.append(g["gesture_type"])
        return raw_seqs, comp_seqs, labels

    tr_raw, tr_comp, tr_lbl = _to_seqs(train_p)
    te_raw, te_comp, te_lbl = _to_seqs(test_p)

    # -- Per-test-gesture: compute distances once, store sorted list -------
    # raw_cache[i]  = (true_label, [(dist, train_label), ...] sorted)
    # comp_cache[i] = same for compressed sequences
    raw_cache, comp_cache = [], []
    for i in range(len(test_p)):
        raw_sorted = sorted(
            [(edit_distance_fast(te_raw[i], tr_raw[j]), tr_lbl[j])
             for j in range(len(train_p))],
            key=lambda x: x[0])
        comp_sorted = sorted(
            [(edit_distance_fast(te_comp[i], tr_comp[j]), tr_lbl[j])
             for j in range(len(train_p))],
            key=lambda x: x[0])
        raw_cache.append((te_lbl[i], raw_sorted))
        comp_cache.append((te_lbl[i], comp_sorted))

    # -- Sweep k - just slice the already-sorted list ----------------------
    results = {}
    for k in K_NEIGHBORS:
        for use_comp, cache in [(True, comp_cache), (False, raw_cache)]:
            y_t = [lbl for lbl, _ in cache]
            y_p = [majority_vote(nn[:k]) for _, nn in cache]
            results[(k, use_comp)] = (y_t, y_p)

    best_acc = max(
        float(np.mean(np.array(y_t) == np.array(y_p)))
        for y_t, y_p in results.values()
    )
    return dict(fold_id=fold_id, cv_mode=cv_mode, pca_opt=pca_opt,
                nc=nc, results=results, best_acc=best_acc)


# ============================================================================
# WORKER: 3-Cent - one (fold x PCA) task, sweeps all n_points inside
# ============================================================================

def _task_tc(train_n, test_n, fold_id, cv_mode, pca_opt):
    """
    Compute 3-Cent accuracy for all n_points values on one (fold, PCA) pair.

    Returns
    -------
    dict with keys: fold_id, cv_mode, pca_opt,
                    results: {n_pts: (y_true, y_pred)}
    """
    train_p, test_p = _apply_pca(train_n, test_n, pca_opt)

    results = {}
    for n_pts in N_POINTS_3CENT:
        templates = build_templates(train_p, n_pts)
        y_t, y_p  = [], []
        for tg in test_p:
            pred = recognize(tg["trajectory"], templates, n_pts)
            y_t.append(tg["gesture_type"])
            y_p.append(pred)
        results[n_pts] = (y_t, y_p)

    best_acc = max(
        float(np.mean(np.array(y_t) == np.array(y_p)))
        for y_t, y_p in results.values()
    )
    return dict(fold_id=fold_id, cv_mode=cv_mode, pca_opt=pca_opt,
                results=results, best_acc=best_acc)


# ============================================================================
# WORKER: DTW - one (fold x PCA) task, sweeps all k inside
# ============================================================================

def _task_dtw(train_n, test_n, fold_id, cv_mode, pca_opt):
    """
    Compute DTW k-NN accuracy for all k values on one (fold, PCA) pair.
    DTW distances are computed once; k is swept by slicing.

    Returns
    -------
    dict with keys: fold_id, cv_mode, pca_opt,
                    results: {k: (y_true, y_pred)}
    """
    train_p, test_p = _apply_pca(train_n, test_n, pca_opt)

    # Pre-compute full distance matrix: test x train
    dtw_cache = []
    for tg in test_p:
        sorted_dists = sorted(
            [(compute_dtw_distance_c_speed(tg["trajectory"], rg["trajectory"]),
              rg["gesture_type"])
             for rg in train_p],
            key=lambda x: x[0])
        dtw_cache.append((tg["gesture_type"], sorted_dists))

    results = {}
    for k in K_NEIGHBORS:
        y_t = [lbl for lbl, _ in dtw_cache]
        y_p = [majority_vote(nn[:k]) for _, nn in dtw_cache]
        results[k] = (y_t, y_p)

    best_acc = max(
        float(np.mean(np.array(y_t) == np.array(y_p)))
        for y_t, y_p in results.values()
    )
    return dict(fold_id=fold_id, cv_mode=cv_mode, pca_opt=pca_opt,
                results=results, best_acc=best_acc)


# ============================================================================
# Aggregation helpers
# ============================================================================

def _aggregate_ed(task_results):
    """
    Combine fine-grained ED task results into per-config accuracy rows
    and a global_predictions dict.

    Config key: (pca_opt, nc, k, use_comp)
    """
    rows  = []
    gp    = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for r in task_results:
        fold_id  = r["fold_id"]
        pca_opt  = r["pca_opt"]
        nc       = r["nc"]
        for (k, use_comp), (y_t, y_p) in r["results"].items():
            acc = float(np.mean(np.array(y_t) == np.array(y_p)))
            cfg = (pca_opt, nc, k, use_comp)
            gp[cfg]["y_true"].extend(y_t)
            gp[cfg]["y_pred"].extend(y_p)
            rows.append(dict(fold_id=fold_id, cv_mode=r["cv_mode"],
                             n_components=pca_opt, n_clusters=nc,
                             k=k, compression=use_comp,
                             n_points="N/A", accuracy=acc))

    return pd.DataFrame(rows), dict(gp)


def _aggregate_tc(task_results):
    """Config key: (pca_opt, n_pts)"""
    rows = []
    gp   = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for r in task_results:
        fold_id = r["fold_id"]
        pca_opt = r["pca_opt"]
        for n_pts, (y_t, y_p) in r["results"].items():
            acc = float(np.mean(np.array(y_t) == np.array(y_p)))
            cfg = (pca_opt, n_pts)
            gp[cfg]["y_true"].extend(y_t)
            gp[cfg]["y_pred"].extend(y_p)
            rows.append(dict(fold_id=fold_id, cv_mode=r["cv_mode"],
                             n_components=pca_opt, n_clusters="N/A",
                             k=1, compression="N/A",
                             n_points=n_pts, accuracy=acc))

    return pd.DataFrame(rows), dict(gp)


def _aggregate_dtw(task_results):
    """Config key: (pca_opt, k)"""
    rows = []
    gp   = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for r in task_results:
        fold_id = r["fold_id"]
        pca_opt = r["pca_opt"]
        for k, (y_t, y_p) in r["results"].items():
            acc = float(np.mean(np.array(y_t) == np.array(y_p)))
            cfg = (pca_opt, k)
            gp[cfg]["y_true"].extend(y_t)
            gp[cfg]["y_pred"].extend(y_p)
            rows.append(dict(fold_id=fold_id, cv_mode=r["cv_mode"],
                             n_components=pca_opt, n_clusters="N/A",
                             k=k, compression="N/A",
                             n_points="N/A", accuracy=acc))

    return pd.DataFrame(rows), dict(gp)


# ============================================================================
# Fold pre-normalisation (runs in main process, not workers)
# ============================================================================

def _build_folds(gestures):
    """
    Run both CV splits and pre-normalise each fold.
    Returns list of (fold_id, cv_mode, train_n, test_n).
    """
    folds = []
    for cv_mode, cv_fn in [("dependent",   user_dependent_cv),
                            ("independent", user_independent_cv)]:
        for train, test, fold_id in cv_fn(gestures):
            mu, sig   = fit_normalizer(train)
            train_n   = apply_normalizer(train, mu, sig)
            test_n    = apply_normalizer(test,  mu, sig)
            folds.append((fold_id, cv_mode, train_n, test_n))
    return folds


# ============================================================================
# Parallel dispatcher with progress bar
# ============================================================================

def _run_parallel(futures_map, total_tasks, method_label, n_workers):
    """
    Drive a dict of {future: task_label} through as_completed() with a
    tqdm bar showing tasks done / left / ETA / running best accuracy.

    Returns list of completed task result dicts.
    """
    results  = []
    best_acc = 0.0

    bar_fmt = ("{l_bar}{bar}| {n_fmt}/{total_fmt} tasks "
               "[{elapsed}<{remaining}, {rate_fmt}]  best={postfix[best]:.1%}")

    with tqdm(total=total_tasks, desc=f"  {method_label:20s}",
              unit="task", ncols=90, bar_format=bar_fmt,
              postfix={"best": 0.0}) as bar:
        for fut in as_completed(futures_map):
            r        = fut.result()
            best_acc = max(best_acc, r.get("best_acc", 0.0))
            results.append(r)
            bar.postfix["best"] = best_acc
            bar.update(1)

    return results


def run_method(folds, method, n_workers, cv_filter=None):
    """
    Build and dispatch all tasks for one method across both CV modes.

    Parameters
    ----------
    folds      : output of _build_folds()
    method     : "ed" | "tc" | "dtw"
    n_workers  : int
    cv_filter  : optional list to restrict CV modes, e.g. ["dependent"]

    Returns (df, global_predictions) for each cv_mode: {cv_mode: (df, gp)}
    """
    cv_modes_active = cv_filter or CV_MODES
    active_folds    = [(fid, cvm, tr, te) for fid, cvm, tr, te in folds
                       if cvm in cv_modes_active]

    # -- Build task list ---------------------------------------------------
    task_args = []
    if method == "ed":
        for fid, cvm, tr_n, te_n in active_folds:
            for pca in PCA_OPTIONS:
                for nc in N_CLUSTERS_ED:
                    task_args.append((_task_ed, tr_n, te_n, fid, cvm, pca, nc))
    elif method == "tc":
        for fid, cvm, tr_n, te_n in active_folds:
            for pca in PCA_OPTIONS:
                task_args.append((_task_tc, tr_n, te_n, fid, cvm, pca))
    elif method == "dtw":
        for fid, cvm, tr_n, te_n in active_folds:
            for pca in PCA_OPTIONS:
                task_args.append((_task_dtw, tr_n, te_n, fid, cvm, pca))

    total = len(task_args)
    label = {"ed": "Edit-distance", "tc": "3-Cent", "dtw": "DTW"}[method]
    print(f"\n  {label}: {total} tasks -> {n_workers} workers")

    # -- Dispatch ---------------------------------------------------------
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(fn, *args): i
            for i, (fn, *args) in enumerate(task_args)
        }
        task_results = _run_parallel(futures, total, label, n_workers)

    elapsed = time.time() - t0
    print(f"  {label} done in {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    # -- Aggregate per CV mode ---------------------------------------------
    agg_fn = {"ed": _aggregate_ed, "tc": _aggregate_tc, "dtw": _aggregate_dtw}[method]

    output = {}
    for cvm in cv_modes_active:
        cv_results = [r for r in task_results if r["cv_mode"] == cvm]
        df, gp     = agg_fn(cv_results)
        output[cvm] = (df, gp)

    return output


# ============================================================================
# Summary + save
# ============================================================================

_GROUP_COLS = {
    "ed":  ["n_components", "n_clusters", "k", "compression"],
    "tc":  ["n_components", "n_points"],
    "dtw": ["n_components", "k"],
}

def _summarise_and_save(df, gp, method, cv_mode, config_label, labels):
    group_cols  = _GROUP_COLS[method]
    # Keep only rows for this CV mode (df may contain both if mixed)
    cv_df       = df[df["cv_mode"] == cv_mode] if "cv_mode" in df.columns else df
    summary     = cv_df.groupby(group_cols)["accuracy"].agg(["mean", "std"])
    best_config = summary["mean"].idxmax()
    best_mean   = summary.loc[best_config, "mean"]
    best_std    = summary.loc[best_config, "std"]

    print(f"  Best  : {best_config}")
    print(f"  Acc   : {best_mean:.2%} +/- {best_std:.2%}")

    key = best_config if isinstance(best_config, tuple) else (best_config,)
    if key in gp:
        y_true = gp[key]["y_true"]
        y_pred = gp[key]["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        save_results(summary, best_config, cm, cv_df,
                     config_label, output_dir="results")
        print(f"  Saved : results/{config_label}_*")


# ============================================================================
# Entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="100%% CPU parallelised gesture recognition pipeline")
    ap.add_argument("--domain", type=int, nargs="+", default=[1, 4],
                    choices=[1, 4])
    ap.add_argument("--method", nargs="+", default=["ed", "tc"],
                    choices=["ed", "dtw", "tc"],
                    help="ed=edit-distance  tc=3-Cent  dtw=DTW (slow)")
    ap.add_argument("--cv", nargs="+", default=CV_MODES,
                    choices=CV_MODES)
    ap.add_argument("--jobs", type=int, default=0,
                    help="Worker count (0 = all logical CPUs)")
    args = ap.parse_args()

    n_logical = os.cpu_count() or 1
    n_workers = args.jobs if args.jobs > 0 else n_logical
    print(f"CPU cores: {n_logical}  ->  using {n_workers} workers (100%)")

    domain_loaders = {
        1: ("Domain 1 (digits 0-9)",
            lambda: load_data_domain_1(os.path.join(
                _BASE, "GestureData_Mons",
                "GestureDataDomain1_Mons", "Domain1_csv"))),
        4: ("Domain 4 (3D shapes)",
            lambda: load_data_domain_4(os.path.join(
                _BASE, "GestureData_Mons", "GestureDataDomain4_Mons"))),
    }

    os.makedirs("results", exist_ok=True)

    grand_start = time.time()
    done        = 0
    total_runs  = len(args.domain) * len(args.method) * len(args.cv)

    for domain_id in args.domain:
        dname, loader = domain_loaders[domain_id]
        print(f"\n{'='*62}\n  {dname}\n{'='*62}")

        gestures = loader()
        labels   = sorted({g["gesture_type"] for g in gestures})
        print(f"  {len(gestures)} gestures  |  {len(labels)} classes  "
              f"|  pre-normalising folds...")

        t_folds = time.time()
        folds   = _build_folds(gestures)
        print(f"  {len(folds)} folds ready in {time.time()-t_folds:.1f} s")

        for method in args.method:
            label = {"ed": "Edit-distance", "tc": "3-Cent", "dtw": "DTW"}[method]

            cv_results = run_method(folds, method=method,
                                    n_workers=n_workers,
                                    cv_filter=args.cv)

            for cv_mode, (df, gp) in cv_results.items():
                done += 1
                config_label = f"domain{domain_id}_{method}_{cv_mode}"
                print(f"\n[{done}/{total_runs}] {config_label}")
                _summarise_and_save(df, gp, method, cv_mode,
                                    config_label, labels)

    total_elapsed = time.time() - grand_start
    print(f"\nAll done in {total_elapsed:.1f} s  ({total_elapsed/60:.1f} min)")
    print("Results saved in ./results/")


if __name__ == "__main__":
    main()
