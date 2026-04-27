"""
statistical_assessment.py
─────────────────────────────────────────────────────────────────────────────
Non-parametric statistical comparison of gesture recognition methods.

Why Friedman test?
------------------
The Friedman test (Friedman 1940) is a non-parametric equivalent of
repeated-measures ANOVA.  We prefer it here for three reasons:
  1. Non-parametric: per-fold accuracies are bounded in [0,1] and with only
     10 folds the normality assumption of ANOVA cannot be verified.
  2. Paired across folds: every method is evaluated on the same 10 folds
     (same train/test splits), so measurements are paired — the Friedman test
     exploits this pairing to increase statistical power.
  3. More than 2 methods: the Friedman test simultaneously compares k > 2
     treatments, avoiding the inflated Type-I error of multiple t-tests.

Why post-hoc correction?
-------------------------
If the Friedman test is significant we only know that AT LEAST one method
differs from the others.  To identify WHICH pairs differ we run pairwise tests.
Running m = k(k-1)/2 = 6 pairwise tests at α=0.05 gives a family-wise
error rate of 1-(1-0.05)^6 ≈ 26%.  Post-hoc correction (Holm-Bonferroni or
Nemenyi) controls this error.

Available post-hoc methods
---------------------------
  • Nemenyi test:  exact non-parametric post-hoc; uses the critical-difference
    (CD) statistic from Demšar (2006).  CD = q_α × sqrt(k(k+1)/(6N)).
    Two methods are significantly different if |rank_i - rank_j| ≥ CD.
  • Wilcoxon + Holm-Bonferroni:  pairwise Wilcoxon signed-rank tests with
    Holm-Bonferroni correction.  Provides explicit adjusted p-values per pair.

Usage
-----
  python statistical_assessment.py                # both domains, independent CV
  python statistical_assessment.py --posthoc both # run both post-hoc methods
  python statistical_assessment.py --cv both      # independent + dependent
"""

import argparse
import itertools
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Nemenyi critical values q_α for α=0.05 (from Demšar 2006, Table 5)
# Index: number of methods k (valid for 2 ≤ k ≤ 10)
_NEMENYI_Q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
              6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# Mapping from result-file name fragment to display name
_METHOD_LABELS = {
    "dtw":           "DTW",
    "edit-distance": "Edit-Distance",
    "Three-cent":    "Three-Cent",
    "bilstm_masked": "BiLSTM",   # from RNN_simplified.py  (preferred)
    "bilstm":        "BiLSTM",   # from baseline_bilstm.py (fallback)
}

# Config columns that identify hyperparameter combinations in each raw CSV.
# "Dynamic" detection is used when the column layout varies between CSV
# versions, but these are the authoritative sets.
_CONFIG_COLS = {
    "dtw":           ["n_components", "k"],
    "edit-distance": ["n_components", "n_clusters", "compression", "k"],
    "Three-cent":    ["n_components", "n_points"],
    "bilstm_masked": [],   # fixed architecture — one accuracy per fold
    "bilstm":        ["target_length", "n_units"],
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_config_cols(df: pd.DataFrame) -> list:
    """
    Detect hyperparameter columns automatically from a raw-result DataFrame.

    A column is considered a "config" column if it has more than one distinct
    non-'N/A' value (i.e., it actually varies across configurations).

    This is robust to different CSV versions that may have different column
    layouts (e.g., presence/absence of 'n_points' or 'cv_mode').
    """
    exclude = {"fold_id", "accuracy", "cv_mode", "method", "domain", "_config"}
    config_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        unique_non_na = {str(v) for v in df[col].unique() if str(v) != "N/A"}
        if len(unique_non_na) > 1:
            config_cols.append(col)
    return config_cols


def _load_best_fold_accuracies(csv_path: str) -> tuple:
    """
    Read a raw-result CSV and return per-fold accuracies for the best
    hyperparameter configuration.

    Strategy
    --------
    1. Detect which columns actually vary (the hyperparameter grid).
    2. Group by those columns and compute mean accuracy across folds.
    3. Identify the best configuration (highest mean).
    4. Return the per-fold accuracies for that configuration.

    For methods with no hyperparameter grid (e.g. fixed-architecture BiLSTM)
    the function returns all fold accuracies directly.

    Parameters
    ----------
    csv_path : str — path to a *_raw.csv file produced by save_results()

    Returns
    -------
    fold_accs   : np.ndarray, shape (n_folds,) — accuracy per fold
    best_config : dict — best hyperparameter configuration (or {} if none)
    """
    df = pd.read_csv(csv_path)
    config_cols = _find_config_cols(df)

    if not config_cols:
        # No varying hyperparameters — one row per fold, just return them
        fold_accs = df.sort_values("fold_id")["accuracy"].values
        return fold_accs, {}

    # Find best config by mean accuracy across folds
    summary = df.groupby(config_cols)["accuracy"].mean()
    best_idx = summary.idxmax()

    # Build mask to filter rows belonging to the best configuration
    # (best_idx is a tuple when multiple config_cols, scalar otherwise)
    if isinstance(best_idx, tuple):
        mask = pd.Series([True] * len(df))
        for col, val in zip(config_cols, best_idx):
            mask &= df[col].astype(str) == str(val)
    else:
        mask = df[config_cols[0]].astype(str) == str(best_idx)

    best_df   = df[mask].sort_values("fold_id")
    fold_accs = best_df["accuracy"].values

    best_config = dict(zip(config_cols,
                           best_idx if isinstance(best_idx, tuple) else [best_idx]))
    return fold_accs, best_config


def _resolve_bilstm_csv(domain_id: int, cv_mode: str, results_dir: str) -> tuple:
    """
    Find the BiLSTM raw CSV: prefer RNN_simplified.py output, fall back to
    baseline_bilstm.py output, raise FileNotFoundError if neither exists.

    Returns (csv_path, method_key) where method_key selects the right
    config-column definition.
    """
    # Preferred: output of RNN_simplified.py (padding + masking)
    masked = os.path.join(results_dir,
                          f"domain{domain_id}_bilstm_masked_{cv_mode}_raw.csv")
    if os.path.exists(masked):
        return masked, "bilstm_masked"

    # Fallback: output of baseline_bilstm.py (resampling)
    old = os.path.join(results_dir,
                       f"domain{domain_id}_bilstm_{cv_mode}_raw.csv")
    if os.path.exists(old):
        print(f"  [WARNING] Using old BiLSTM results (baseline_bilstm.py)."
              f" Run RNN_simplified.py first for the padding+masking version.")
        return old, "bilstm"

    raise FileNotFoundError(
        f"No BiLSTM results found for domain{domain_id} / {cv_mode}.\n"
        f"  Run: python RNN_simplified.py\n"
        f"  Expected: {masked}")


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_accuracy_matrix(
        domain_id:   int,
        cv_mode:     str,
        results_dir: str,
) -> tuple:
    """
    Assemble the (n_folds × n_methods) accuracy matrix needed by the
    Friedman test.

    For each method the per-fold accuracies of the BEST configuration are used
    (highest mean accuracy across the hyperparameter grid).  For BiLSTM there
    is only one configuration (fixed architecture).

    Parameters
    ----------
    domain_id   : 1 or 4
    cv_mode     : "independent" or "dependent"
    results_dir : directory containing the raw CSV files

    Returns
    -------
    matrix       : pd.DataFrame, shape (n_folds, n_methods)
                   rows = folds, columns = method display names
    best_configs : dict {method_key: best_config_dict}
    """
    method_keys = ["dtw", "edit-distance", "Three-cent"]
    file_templates = {
        "dtw":           f"domain{domain_id}_dtw_{cv_mode}_raw.csv",
        "edit-distance": f"domain{domain_id}_edit-distance_{cv_mode}_raw.csv",
        "Three-cent":    f"domain{domain_id}_Three-cent_{cv_mode}_raw.csv",
    }

    columns      = {}
    best_configs = {}

    for key in method_keys:
        csv_path = os.path.join(results_dir, file_templates[key])
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing result file: {csv_path}\n"
                f"Run the baseline pipeline (main.py or main_optimized.py) first.")
        fold_accs, best_cfg = _load_best_fold_accuracies(csv_path)
        columns[_METHOD_LABELS[key]] = fold_accs
        best_configs[key] = best_cfg

    # BiLSTM
    bilstm_csv, bilstm_key = _resolve_bilstm_csv(domain_id, cv_mode, results_dir)
    bilstm_accs, bilstm_cfg = _load_best_fold_accuracies(bilstm_csv)
    columns[_METHOD_LABELS["bilstm_masked"]] = bilstm_accs
    best_configs["bilstm"] = bilstm_cfg

    # Verify all methods have the same number of folds
    n_folds_set = {len(v) for v in columns.values()}
    if len(n_folds_set) != 1:
        raise ValueError(
            f"Methods have different numbers of folds: "
            f"{dict(zip(columns.keys(), [len(v) for v in columns.values()]))}")

    matrix = pd.DataFrame(columns)
    return matrix, best_configs


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_friedman(matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Friedman test across k methods and N folds.

    The Friedman statistic is computed on ranks within each fold (row).
    Under H₀ (all methods equivalent), it follows a chi-squared distribution
    with k-1 degrees of freedom.

    Parameters
    ----------
    matrix : pd.DataFrame, shape (N_folds, k_methods), values = accuracies
    alpha  : significance threshold

    Returns
    -------
    dict with keys: statistic, p_value, significant, mean_ranks
    """
    # scipy expects one positional argument per method (column)
    stat, p = friedmanchisquare(*[matrix[col].values for col in matrix.columns])

    # Average rank per method (rank 1 = best within each fold)
    # scipy.stats.rankdata ranks 1=lowest, so negate accuracies to rank 1=highest
    ranks_per_fold = matrix.apply(
        lambda row: pd.Series(
            len(matrix.columns) + 1 - row.rank(method="average").values,
            index=matrix.columns), axis=1)
    mean_ranks = ranks_per_fold.mean()

    return {
        "statistic":  float(stat),
        "p_value":    float(p),
        "significant": p < alpha,
        "mean_ranks": mean_ranks,
    }


def run_nemenyi(matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Nemenyi post-hoc test following Demšar (2006).

    The critical difference CD = q_α × sqrt(k(k+1) / (6N)) defines the
    minimum average-rank difference required for statistical significance.
    Two methods are significantly different when |rank_i - rank_j| ≥ CD.

    Parameters
    ----------
    matrix : pd.DataFrame, shape (N_folds, k_methods)
    alpha  : significance threshold (only 0.05 is supported via lookup table)

    Returns
    -------
    dict with keys: cd, pairwise_matrix, significant_pairs, mean_ranks
    """
    k = len(matrix.columns)
    N = len(matrix)

    # scikit-posthocs provides Nemenyi based on Friedman ranks
    pairwise_p = sp.posthoc_nemenyi_friedman(matrix.values)
    pairwise_p.index   = matrix.columns.tolist()
    pairwise_p.columns = matrix.columns.tolist()

    # Critical difference (Demšar 2006, Equation 6)
    q_alpha = _NEMENYI_Q.get(k, _NEMENYI_Q[10])
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    # Rank each method within each fold (1 = best)
    ranks_per_fold = matrix.apply(
        lambda row: pd.Series(
            len(matrix.columns) + 1 - row.rank(method="average").values,
            index=matrix.columns), axis=1)
    mean_ranks = ranks_per_fold.mean()

    # Identify significantly different pairs
    methods = matrix.columns.tolist()
    sig_pairs = []
    for m1, m2 in itertools.combinations(methods, 2):
        if pairwise_p.loc[m1, m2] < alpha:
            sig_pairs.append((m1, m2, pairwise_p.loc[m1, m2]))

    return {
        "cd":               cd,
        "pairwise_matrix":  pairwise_p,
        "significant_pairs": sig_pairs,
        "mean_ranks":        mean_ranks,
    }


def run_wilcoxon_holm(matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    Holm-Bonferroni controls the family-wise error rate without assuming
    independence between tests (unlike Bonferroni).  Sorted p-values p[1] ≤ …
    are rejected if p[i] ≤ α / (m - i + 1) for all j ≤ i.

    Parameters
    ----------
    matrix : pd.DataFrame, shape (N_folds, k_methods)
    alpha  : significance threshold

    Returns
    -------
    pd.DataFrame with columns: method_1, method_2, statistic, p_raw,
                                p_adjusted, significant
    """
    methods = matrix.columns.tolist()
    rows = []

    for m1, m2 in itertools.combinations(methods, 2):
        diff = matrix[m1].values - matrix[m2].values
        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(matrix[m1].values, matrix[m2].values,
                               alternative="two-sided")
        rows.append({"method_1": m1, "method_2": m2,
                     "statistic": stat, "p_raw": p})

    result_df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    # Holm-Bonferroni correction: step-down procedure
    m = len(result_df)
    p_adj = []
    for i, p_raw in enumerate(result_df["p_raw"]):
        corrected = p_raw * (m - i)   # Holm multiplier
        p_adj.append(min(corrected, 1.0))

    result_df["p_adjusted"] = p_adj
    result_df["significant"] = result_df["p_adjusted"] < alpha
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Critical-difference diagram (Demšar 2006 style)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_cd_diagram(
        mean_ranks: pd.Series,
        cd:         float,
        title:      str,
        save_path:  str = None,
):
    """
    Draw a Demšar (2006) critical-difference diagram.

    Layout:
      • Horizontal axis runs from rank k (worst) on the left to rank 1
        (best) on the right.
      • Each method is shown as a black dot at its average rank with a label.
      • A blue 'CD' bracket in the top-left corner shows the critical distance.
      • Thick black horizontal bars connect methods whose rank difference is
        strictly less than CD (i.e., not significantly different).

    Parameters
    ----------
    mean_ranks : pd.Series — {method_name: average_rank}
    cd         : float     — critical difference
    title      : str
    save_path  : optional file path (PNG)
    """
    sorted_items = sorted(mean_ranks.items(), key=lambda x: x[1])
    names  = [m for m, _ in sorted_items]
    ranks  = np.array([r for _, r in sorted_items])
    k      = len(names)

    fig, ax = plt.subplots(figsize=(max(9, k * 2), 3.5))

    # Axis runs left=worst to right=best (reverse x)
    x_min, x_max = 0.5, k + 0.5
    ax.set_xlim(x_max, x_min)   # reversed
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")

    # Main horizontal axis line
    ax.hlines(0, x_min, x_max, colors="black", lw=1.5)
    for r in range(1, k + 1):
        ax.vlines(r, -0.06, 0.06, colors="black", lw=1.5)
        ax.text(r, -0.18, str(r), ha="center", va="top", fontsize=10)
    ax.text(x_max + 0.05, -0.18, "Rank →", ha="left", va="top",
            fontsize=9, style="italic", color="gray")

    # Method markers — alternate labels above/below to avoid overlap
    for i, (name, rank) in enumerate(zip(names, ranks)):
        y_sign = 1 if i % 2 == 0 else -1
        y_dot  = 0.0
        y_stem = y_sign * 0.08
        y_text = y_sign * 0.65
        ax.vlines(rank, y_dot, y_stem, colors="black", lw=1)
        ax.plot(rank, y_dot, "ko", ms=7, zorder=5)
        ax.text(rank, y_text, name, ha="center",
                va="bottom" if y_sign > 0 else "top",
                fontsize=10, fontweight="bold")

    # CD scale bracket in top-left corner of the axis
    cd_x, cd_y = x_max - 0.3, 1.35
    ax.annotate("", xy=(cd_x - cd, cd_y), xytext=(cd_x, cd_y),
                arrowprops=dict(arrowstyle="<->", color="steelblue", lw=2))
    ax.text(cd_x - cd / 2, cd_y + 0.18,
            f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=9, color="steelblue", fontweight="bold")

    # Clique bars: connect methods not significantly different
    # Greedy left-to-right scan (best rank = leftmost on reversed axis)
    bar_y_base = -1.1
    bar_level  = 0
    i = 0
    while i < k:
        j = i
        # Extend clique while next method is within CD of clique start
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i:
            bar_y = bar_y_base - bar_level * 0.22
            ax.hlines(bar_y, ranks[i], ranks[j], colors="black", lw=3.5)
            ax.vlines(ranks[i], bar_y, bar_y + 0.06, colors="black", lw=2)
            ax.vlines(ranks[j], bar_y, bar_y + 0.06, colors="black", lw=2)
            bar_level += 1
        i += 1

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  CD diagram saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main assessment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_assessment(
        domain_id:    int,
        cv_mode:      str     = "independent",
        results_dir:  str     = "results",
        alpha:        float   = 0.05,
        posthoc:      str     = "both",   # "nemenyi" | "wilcoxon" | "both"
) -> dict:
    """
    Run the full statistical comparison for one (domain, cv_mode) pair.

    Steps
    -----
    1. Load per-fold best-config accuracies for each method.
    2. Friedman test (non-parametric, paired, k > 2 methods).
    3. If significant: Nemenyi test and/or Wilcoxon+Holm post-hoc.
    4. Critical-difference diagram (Demšar 2006).
    5. Save results table and diagram.

    Parameters
    ----------
    domain_id   : 1 or 4
    cv_mode     : "independent" (default — required by project guidelines) or
                  "dependent"
    results_dir : directory containing raw CSV files
    alpha       : significance level (default 0.05)
    posthoc     : which post-hoc method(s) to run

    Returns
    -------
    dict with keys: friedman, nemenyi (if run), wilcoxon (if run),
                    accuracy_matrix, domain_id, cv_mode
    """
    domain_label = f"Domain {domain_id}"
    print(f"\n{'='*62}")
    print(f"  {domain_label}  |  CV mode: {cv_mode}")
    print(f"{'='*62}")

    # 1. Build accuracy matrix
    try:
        matrix, best_cfgs = _build_accuracy_matrix(domain_id, cv_mode, results_dir)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return {}

    print(f"\n  Per-fold accuracies (best config per method):")
    print(matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Mean accuracies:")
    for col in matrix.columns:
        print(f"    {col:20s}: {matrix[col].mean():.4f} ± {matrix[col].std():.4f}")
    print(f"\n  Best configs used:")
    for k, v in best_cfgs.items():
        print(f"    {k:20s}: {v}")

    output = {"accuracy_matrix": matrix, "domain_id": domain_id,
              "cv_mode": cv_mode}
    stat_rows = []   # rows for the results CSV

    # 2. Friedman test
    print(f"\n  ── Friedman test (α = {alpha}) ──")
    friedman_result = run_friedman(matrix, alpha)
    stat = friedman_result["statistic"]
    p    = friedman_result["p_value"]
    sig  = friedman_result["significant"]

    print(f"    χ² = {stat:.4f},  p = {p:.6f}")
    print(f"    Result: {'SIGNIFICANT' if sig else 'NOT significant'} at α = {alpha}")
    print(f"    Average ranks: {friedman_result['mean_ranks'].to_dict()}")

    stat_rows.append({
        "domain": domain_label, "cv_mode": cv_mode,
        "test_name": "Friedman", "statistic": stat,
        "p_value": p, "significant": sig,
    })
    output["friedman"] = friedman_result

    # 3. Post-hoc tests (only if Friedman is significant)
    if not sig:
        print(f"\n  Friedman test not significant — no post-hoc tests needed.")
    else:
        print(f"\n  Friedman test significant — running post-hoc tests...")

        # Nemenyi
        if posthoc in ("nemenyi", "both"):
            print(f"\n  ── Nemenyi post-hoc ──")
            nem = run_nemenyi(matrix, alpha)
            cd  = nem["cd"]
            k   = len(matrix.columns)
            N   = len(matrix)
            print(f"    k={k} methods, N={N} folds  →  CD = {cd:.4f}")
            print(f"    Nemenyi p-value matrix:")
            print(nem["pairwise_matrix"].to_string(float_format=lambda x: f"{x:.4f}"))
            if nem["significant_pairs"]:
                print(f"\n    Significantly different pairs (p < {alpha}):")
                for m1, m2, pv in nem["significant_pairs"]:
                    print(f"      {m1}  vs  {m2}  →  p = {pv:.4f}")
            else:
                print(f"    No significantly different pairs found.")

            for m1, m2 in itertools.combinations(matrix.columns, 2):
                pv = nem["pairwise_matrix"].loc[m1, m2]
                stat_rows.append({
                    "domain": domain_label, "cv_mode": cv_mode,
                    "test_name": f"Nemenyi: {m1} vs {m2}",
                    "statistic": abs(nem["mean_ranks"][m1] - nem["mean_ranks"][m2]),
                    "p_value": float(pv), "significant": pv < alpha,
                })

            # CD diagram using Nemenyi ranks
            diag_path = os.path.join(
                results_dir, f"cd_diagram_domain{domain_id}_{cv_mode}.png")
            _draw_cd_diagram(
                mean_ranks=nem["mean_ranks"],
                cd=cd,
                title=f"Critical Difference Diagram — {domain_label} ({cv_mode} CV)",
                save_path=diag_path,
            )
            output["nemenyi"] = nem

        # Wilcoxon + Holm-Bonferroni
        if posthoc in ("wilcoxon", "both"):
            print(f"\n  ── Wilcoxon signed-rank + Holm-Bonferroni ──")
            wx = run_wilcoxon_holm(matrix, alpha)
            print(wx.to_string(index=False,
                               float_format=lambda x: f"{x:.4f}"))
            sig_pairs_wx = wx[wx["significant"]]
            if len(sig_pairs_wx):
                print(f"\n    Significantly different pairs (adj-p < {alpha}):")
                for _, row in sig_pairs_wx.iterrows():
                    print(f"      {row['method_1']}  vs  {row['method_2']}"
                          f"  →  adj-p = {row['p_adjusted']:.4f}")
            else:
                print(f"    No significantly different pairs after correction.")

            for _, row in wx.iterrows():
                stat_rows.append({
                    "domain": domain_label, "cv_mode": cv_mode,
                    "test_name": f"Wilcoxon+Holm: {row['method_1']} vs {row['method_2']}",
                    "statistic": row["statistic"],
                    "p_value": row["p_raw"],
                    "significant": row["significant"],
                })
            output["wilcoxon_holm"] = wx

    # 4. Save results table
    os.makedirs(results_dir, exist_ok=True)
    test_csv = os.path.join(results_dir, "statistical_tests.csv")
    stat_df  = pd.DataFrame(stat_rows)

    # Append to existing file if it already has rows for other domains/modes
    if os.path.exists(test_csv):
        existing = pd.read_csv(test_csv)
        # Drop rows for this exact (domain, cv_mode) combination before appending
        existing = existing[~((existing["domain"] == domain_label) &
                              (existing["cv_mode"] == cv_mode))]
        stat_df = pd.concat([existing, stat_df], ignore_index=True)
    stat_df.to_csv(test_csv, index=False)
    print(f"\n  Results saved → {test_csv}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(all_outputs: list) -> None:
    """Print a concise human-readable summary across all runs."""
    print(f"\n{'='*62}")
    print("  SUMMARY")
    print(f"{'='*62}")

    for out in all_outputs:
        if not out:
            continue
        domain    = f"Domain {out['domain_id']}"
        cv_mode   = out["cv_mode"]
        friedman  = out.get("friedman", {})
        p         = friedman.get("p_value", float("nan"))
        sig       = friedman.get("significant", False)

        print(f"\n  {domain} | {cv_mode} CV"
              f"  —  Friedman p = {p:.4f} "
              f"({'significant' if sig else 'NOT significant'})")

        # Identify best method by lowest mean rank
        mean_ranks = friedman.get("mean_ranks")
        if mean_ranks is not None:
            best_method = mean_ranks.idxmin()
            print(f"    Best method (lowest avg rank): {best_method}")

        # Which pairs are significantly different (Nemenyi preferred)
        if "nemenyi" in out and sig:
            pairs = out["nemenyi"]["significant_pairs"]
            if pairs:
                print(f"    Significantly different pairs (Nemenyi, α=0.05):")
                for m1, m2, pv in pairs:
                    print(f"      {m1}  ≠  {m2}  (p={pv:.4f})")
            else:
                print(f"    No pair significantly different after Nemenyi correction.")
        elif "wilcoxon_holm" in out and sig:
            wx = out["wilcoxon_holm"]
            sig_wx = wx[wx["significant"]]
            if len(sig_wx):
                for _, row in sig_wx.iterrows():
                    print(f"      {row['method_1']}  ≠  {row['method_2']}"
                          f"  (adj-p={row['p_adjusted']:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Statistical comparison of gesture recognition methods")
    ap.add_argument("--domain", type=int, nargs="+", default=[1, 4],
                    choices=[1, 4], help="Domain IDs to process")
    ap.add_argument("--cv", nargs="+", default=["independent"],
                    choices=["independent", "dependent"],
                    help="CV mode(s) to evaluate")
    ap.add_argument("--posthoc", default="both",
                    choices=["nemenyi", "wilcoxon", "both"],
                    help="Post-hoc test(s) to run if Friedman is significant")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level (default: 0.05)")
    ap.add_argument("--results_dir", default="results",
                    help="Directory containing raw CSV result files")
    args = ap.parse_args()

    all_outputs = []
    for domain_id in args.domain:
        for cv_mode in args.cv:
            out = run_assessment(
                domain_id   = domain_id,
                cv_mode     = cv_mode,
                results_dir = args.results_dir,
                alpha       = args.alpha,
                posthoc     = args.posthoc,
            )
            all_outputs.append(out)

    _print_summary(all_outputs)

    # List all files created
    print(f"\nFiles written:")
    results_dir = args.results_dir
    if os.path.exists(os.path.join(results_dir, "statistical_tests.csv")):
        print(f"  {results_dir}/statistical_tests.csv")
    for domain_id in args.domain:
        for cv_mode in args.cv:
            diag = os.path.join(results_dir,
                                f"cd_diagram_domain{domain_id}_{cv_mode}.png")
            if os.path.exists(diag):
                print(f"  {diag}")


if __name__ == "__main__":
    main()
