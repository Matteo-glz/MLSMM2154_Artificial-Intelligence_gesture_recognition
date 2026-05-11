"""
results_explorer.py
─────────────────────────────────────────────────────────────────────────────
Interactive Streamlit dashboard that loads pre-computed results
(from precompute_results.py) and lets you explore & compare any
configuration across domains, methods, CV modes and hyperparameters.

Prerequisites
─────────────
    pip install streamlit
    python precompute_results.py --domain 1        # (run once)

Launch
──────
    streamlit run results_explorer.py
"""

import os, sys, pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

GESTURE_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#808080","#b8860b","#17becf",
]

st.set_page_config(
    page_title="Gesture Recognition — Results Explorer",
    page_icon="🤚",
    layout="wide",
)

# ─── Load pickles ─────────────────────────────────────────────────────────────
@st.cache_data
def load_results(domain: int):
    path = os.path.join(RESULTS_DIR, f"precomputed_domain{domain}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─── Helpers ──────────────────────────────────────────────────────────────────
METHOD_LABELS = {
    "edit-distance": "Edit-Distance (k-NN)",
    "dtw":           "DTW (k-NN)",
    "three-cent":    "3-Cent (nearest template)",
}

def _key_label(method, key):
    """Human-readable label for a config key tuple."""
    if method == "edit-distance":
        pca, nc, k, comp = key
        return f"PCA={pca}  clusters={nc}  k={k}  compress={'Y' if comp else 'N'}"
    elif method == "dtw":
        pca, k = key
        return f"PCA={pca}  k={k}"
    else:  # three-cent
        pca, npts = key
        return f"PCA={pca}  n_points={npts}"


def _config_selectors(method, cv_data, prefix=""):
    """Streamlit sidebar widgets to select one configuration."""
    keys = list(cv_data.keys())

    if method == "edit-distance":
        pca_opts   = sorted({k[0] for k in keys}, key=lambda x: (x=="no_pca", x))
        nc_opts    = sorted({k[1] for k in keys})
        k_opts     = sorted({k[2] for k in keys})
        comp_opts  = [True, False]

        pca  = st.selectbox(f"{prefix} PCA",        pca_opts,  key=f"{prefix}_pca")
        nc   = st.selectbox(f"{prefix} n_clusters",  nc_opts,  key=f"{prefix}_nc")
        k    = st.selectbox(f"{prefix} k neighbors", k_opts,   key=f"{prefix}_k")
        comp = st.selectbox(f"{prefix} Compression", comp_opts,
                            format_func=lambda x: "Yes" if x else "No",
                            key=f"{prefix}_comp")
        return (pca, nc, k, comp)

    elif method == "dtw":
        pca_opts = sorted({k[0] for k in keys}, key=lambda x: (x=="no_pca", x))
        k_opts   = sorted({k[1] for k in keys})
        pca = st.selectbox(f"{prefix} PCA",         pca_opts, key=f"{prefix}_pca")
        k   = st.selectbox(f"{prefix} k neighbors", k_opts,   key=f"{prefix}_k")
        return (pca, k)

    else:  # three-cent
        pca_opts  = sorted({k[0] for k in keys}, key=lambda x: (x=="no_pca", x))
        npt_opts  = sorted({k[1] for k in keys})
        pca = st.selectbox(f"{prefix} PCA",      pca_opts, key=f"{prefix}_pca")
        npt = st.selectbox(f"{prefix} n_points", npt_opts, key=f"{prefix}_npt")
        return (pca, npt)


def _confusion_heatmap(result, title, gesture_types, label_map, height=400):
    cm = result["confusion_matrix"].astype(float)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    cm_pct = cm / rs * 100

    tick = [f"{gt}<br>({label_map.get(gt,gt)})" for gt in gesture_types]

    hover = [[
        f"True: {label_map.get(gesture_types[i], gesture_types[i])}<br>"
        f"Pred: {label_map.get(gesture_types[j], gesture_types[j])}<br>"
        f"Count: {int(result['confusion_matrix'][i,j])}  ({cm_pct[i,j]:.1f}%)"
        for j in range(len(gesture_types))]
        for i in range(len(gesture_types))]

    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=tick, y=tick,
        colorscale="Blues", zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in cm_pct],
        texttemplate="%{text}", textfont=dict(size=10),
        customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickangle=40),
        height=height,
        margin=dict(l=0, r=0, b=60, t=60),
    )
    return fig


def _per_class_bar(result, title, gesture_types, label_map, color="#377eb8", height=350):
    cm  = result["confusion_matrix"]
    rs  = cm.sum(axis=1)
    pc  = np.where(rs > 0, cm.diagonal() / rs, 0.0)
    labels = [f"{gt} ({label_map.get(gt,gt)})" for gt in gesture_types]
    colors = [GESTURE_COLORS[gt % len(GESTURE_COLORS)] for gt in gesture_types]

    fig = go.Figure(go.Bar(
        x=labels, y=pc,
        marker_color=colors,
        text=[f"{v:.0%}" for v in pc],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1.18]),
        xaxis=dict(tickangle=35),
        height=height,
        margin=dict(l=0, r=0, b=80, t=50),
    )
    return fig


def _fold_box(result, title, height=320):
    fig = go.Figure(go.Box(
        y=result["per_fold_accuracy"],
        boxpoints="all", jitter=0.3, pointpos=-1.6,
        marker=dict(size=8, color="#377eb8"),
        line=dict(color="#1a1a2e"),
        name="Fold accuracy",
    ))
    mean = result["mean_accuracy"]
    std  = result["std_accuracy"]
    fig.add_hline(y=mean, line_dash="dash", line_color="red",
                  annotation_text=f"mean={mean:.2%}", annotation_position="right")
    fig.update_layout(
        title=f"{title}<br><sup>mean={mean:.2%}  std={std:.2%}</sup>",
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1.05]),
        height=height,
        margin=dict(l=0, r=60, b=40, t=70),
        showlegend=False,
    )
    return fig


def _accuracy_overview(cv_data, method, gesture_types, height=420):
    """Scatter plot of all configs: mean accuracy vs std, size=n_configs."""
    keys  = list(cv_data.keys())
    means = [cv_data[k]["mean_accuracy"] for k in keys]
    stds  = [cv_data[k]["std_accuracy"]  for k in keys]
    labels = [_key_label(method, k) for k in keys]

    fig = go.Figure(go.Scatter(
        x=stds, y=means, mode="markers",
        marker=dict(size=8, color=means, colorscale="RdYlGn",
                    cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title="Accuracy")),
        text=labels, hovertemplate="%{text}<br>mean=%{y:.2%}  std=%{x:.2%}<extra></extra>",
    ))

    # Highlight best
    best_i = int(np.argmax(means))
    fig.add_trace(go.Scatter(
        x=[stds[best_i]], y=[means[best_i]], mode="markers+text",
        marker=dict(size=14, color="gold", symbol="star",
                    line=dict(color="black", width=1.5)),
        text=[f"Best: {means[best_i]:.2%}"],
        textposition="top right",
        name="Best config",
    ))

    fig.update_layout(
        title="All configurations: accuracy vs std  (top-right = best)",
        xaxis=dict(title="Std across folds", tickformat=".1%"),
        yaxis=dict(title="Mean accuracy", tickformat=".0%"),
        height=height,
        margin=dict(l=0, r=0, b=50, t=60),
    )
    return fig


def _comparison_bar(res_a, res_b, label_a, label_b, gesture_types, label_map,
                    height=380):
    """Side-by-side per-class accuracy bar for two configs."""
    def _per_class(res):
        cm = res["confusion_matrix"]
        rs = cm.sum(axis=1)
        return np.where(rs > 0, cm.diagonal() / rs, 0.0)

    pc_a = _per_class(res_a)
    pc_b = _per_class(res_b)
    diff = pc_b - pc_a   # positive = B is better

    xlabels = [f"{gt} ({label_map.get(gt,gt)})" for gt in gesture_types]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels, y=pc_a, name=label_a,
        marker_color="#377eb8",
        text=[f"{v:.0%}" for v in pc_a], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=xlabels, y=pc_b, name=label_b,
        marker_color="#e41a1c",
        text=[f"{v:.0%}" for v in pc_b], textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1.25]),
        xaxis=dict(tickangle=35),
        legend=dict(title="Config"),
        height=height,
        margin=dict(l=0, r=0, b=80, t=30),
    )
    return fig


def _delta_bar(res_a, res_b, label_a, label_b, gesture_types, label_map,
               height=320):
    """Delta bar: B accuracy minus A accuracy per class."""
    def _per_class(res):
        cm = res["confusion_matrix"]
        rs = cm.sum(axis=1)
        return np.where(rs > 0, cm.diagonal() / rs, 0.0)

    diff = _per_class(res_b) - _per_class(res_a)
    xlabels = [f"{gt} ({label_map.get(gt,gt)})" for gt in gesture_types]
    colors  = ["#4daf4a" if d >= 0 else "#e41a1c" for d in diff]

    fig = go.Figure(go.Bar(
        x=xlabels, y=diff,
        marker_color=colors,
        text=[f"{v:+.0%}" for v in diff], textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Delta per class: ({label_b}) − ({label_a})",
        yaxis=dict(title="Accuracy delta", tickformat="+.0%"),
        xaxis=dict(tickangle=35),
        height=height,
        margin=dict(l=0, r=0, b=80, t=50),
    )
    return fig


def _method_summary_bar(data, gesture_types, label_map, height=420):
    """
    For all methods present in 'data', plot best accuracy per
    (method, cv_mode) as a grouped bar chart.
    data: the full loaded pickle dict
    """
    methods  = [m for m in ["edit-distance","dtw","three-cent"]
                if data.get(m)]
    cv_modes = ["independent","dependent"]
    bar_col  = {"independent":"#e41a1c","dependent":"#377eb8"}

    fig = go.Figure()
    for cv_m in cv_modes:
        x_lbl, y_mean, y_err = [], [], []
        for mth in methods:
            cv_data = data[mth].get(cv_m, {})
            if not cv_data:
                continue
            best = max(cv_data.values(), key=lambda r: r["mean_accuracy"])
            x_lbl.append(METHOD_LABELS.get(mth, mth))
            y_mean.append(best["mean_accuracy"])
            y_err.append(best["std_accuracy"])
        fig.add_trace(go.Bar(
            x=x_lbl, y=y_mean,
            error_y=dict(type="data", array=y_err, visible=True),
            name="User-Indep." if cv_m=="independent" else "User-Dep.",
            marker_color=bar_col[cv_m],
            text=[f"{v:.1%}" for v in y_mean], textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Best accuracy per method & CV mode",
        yaxis=dict(title="Mean accuracy", tickformat=".0%", range=[0, 1.20]),
        legend=dict(title="CV mode"),
        height=height,
        margin=dict(l=0, r=0, b=60, t=60),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# App layout
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🤚 Gesture Recognition — Interactive Results Explorer")
    st.caption("MLSMM2154 Artificial Intelligence  |  Load pre-computed results to compare any configuration.")

    # ── Check available pickles ───────────────────────────────────────────────
    available = [d for d in [1, 4]
                 if os.path.exists(os.path.join(RESULTS_DIR,
                                                f"precomputed_domain{d}.pkl"))]
    if not available:
        st.error(
            "No pre-computed results found. Run first:\n\n"
            "```\npython precompute_results.py --domain 1\n```"
        )
        st.stop()

    # ── Sidebar — global selectors ────────────────────────────────────────────
    with st.sidebar:
        st.header("Global settings")
        domain   = st.selectbox("Domain", available,
                                format_func=lambda d: f"Domain {d}")
        data     = load_results(domain)
        meta     = data["meta"]
        gt       = meta["gesture_types"]
        lmap     = meta["label_map"]

        st.caption(f"{meta['n_gestures']} gestures, {len(gt)} classes")

        methods_avail = [m for m in ["edit-distance","dtw","three-cent"]
                         if data.get(m)]
        method = st.selectbox("Method", methods_avail,
                              format_func=lambda m: METHOD_LABELS.get(m,m))
        cv_mode = st.selectbox("CV mode",
                               ["independent","dependent"],
                               format_func=lambda m: "User-Independent" if m=="independent"
                                                     else "User-Dependent")

        cv_data = data[method].get(cv_mode, {})
        if not cv_data:
            st.warning("No results for this combination.")
            st.stop()

        st.divider()
        mode = st.radio("View mode", ["Single config", "Compare two configs",
                                      "Method overview"])

    # ── Tabs ─────────────────────────────────────────────────────────────────
    if mode == "Method overview":
        st.subheader(f"Domain {domain} — All methods overview")
        st.plotly_chart(_method_summary_bar(data, gt, lmap), use_container_width=True)

        st.subheader(f"All configurations for '{METHOD_LABELS[method]}' "
                     f"/ {cv_mode}")
        st.plotly_chart(_accuracy_overview(cv_data, method, gt), use_container_width=True)

        # Top-5 configs table
        sorted_keys = sorted(cv_data, key=lambda k: cv_data[k]["mean_accuracy"],
                             reverse=True)
        st.subheader("Top-10 configurations")
        rows = []
        for k in sorted_keys[:10]:
            r = cv_data[k]
            rows.append({
                "Config": _key_label(method, k),
                "Mean accuracy": f"{r['mean_accuracy']:.2%}",
                "Std": f"{r['std_accuracy']:.2%}",
                "Per-fold": "  ".join(f"{v:.0%}" for v in r["per_fold_accuracy"]),
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif mode == "Single config":
        with st.sidebar:
            st.subheader("Configuration")
            cfg_key = _config_selectors(method, cv_data, prefix="A")

        result = cv_data.get(cfg_key)
        if result is None:
            st.warning(f"Config {cfg_key} not found.")
            st.stop()

        st.subheader(f"Domain {domain} · {METHOD_LABELS[method]} "
                     f"· {cv_mode} · {_key_label(method, cfg_key)}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean accuracy",
                    f"{result['mean_accuracy']:.2%}",
                    f"± {result['std_accuracy']:.2%}")
        col2.metric("Best fold",
                    f"{max(result['per_fold_accuracy']):.2%}")
        col3.metric("Worst fold",
                    f"{min(result['per_fold_accuracy']):.2%}")

        c1, c2 = st.columns([6, 4])
        with c1:
            st.plotly_chart(
                _confusion_heatmap(result,
                    f"Confusion matrix — {cv_mode} CV",
                    gt, lmap, height=460),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                _fold_box(result, "Accuracy per fold", height=460),
                use_container_width=True)

        st.plotly_chart(
            _per_class_bar(result,
                f"Per-class accuracy — {_key_label(method, cfg_key)}",
                gt, lmap, height=380),
            use_container_width=True)

    elif mode == "Compare two configs":
        with st.sidebar:
            st.subheader("Config A")
            method_a  = st.selectbox("Method A", methods_avail,
                                     format_func=lambda m: METHOD_LABELS.get(m,m),
                                     key="ma")
            cv_mode_a = st.selectbox("CV mode A",
                                     ["independent","dependent"],
                                     format_func=lambda m: "User-Indep." if m=="independent"
                                                           else "User-Dep.",
                                     key="cva")
            cv_data_a = data[method_a].get(cv_mode_a, {})
            cfg_a     = _config_selectors(method_a, cv_data_a, prefix="A")

            st.divider()
            st.subheader("Config B")
            method_b  = st.selectbox("Method B", methods_avail,
                                     format_func=lambda m: METHOD_LABELS.get(m,m),
                                     key="mb")
            cv_mode_b = st.selectbox("CV mode B",
                                     ["independent","dependent"],
                                     format_func=lambda m: "User-Indep." if m=="independent"
                                                           else "User-Dep.",
                                     key="cvb")
            cv_data_b = data[method_b].get(cv_mode_b, {})
            cfg_b     = _config_selectors(method_b, cv_data_b, prefix="B")

        res_a = cv_data_a.get(cfg_a)
        res_b = cv_data_b.get(cfg_b)
        if res_a is None or res_b is None:
            st.warning("One or both configs not found in the pre-computed data.")
            st.stop()

        label_a = f"A: {METHOD_LABELS[method_a]} | {cv_mode_a} | {_key_label(method_a, cfg_a)}"
        label_b = f"B: {METHOD_LABELS[method_b]} | {cv_mode_b} | {_key_label(method_b, cfg_b)}"

        # Header metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy A", f"{res_a['mean_accuracy']:.2%}",
                    f"± {res_a['std_accuracy']:.2%}")
        col2.metric("Accuracy B", f"{res_b['mean_accuracy']:.2%}",
                    f"± {res_b['std_accuracy']:.2%}",
                    delta=f"{res_b['mean_accuracy']-res_a['mean_accuracy']:+.2%}")
        col3.metric("Winner",
                    "A" if res_a["mean_accuracy"] > res_b["mean_accuracy"] else "B",
                    f"|Δ| = {abs(res_a['mean_accuracy']-res_b['mean_accuracy']):.2%}")

        # Confusion matrices side by side
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                _confusion_heatmap(res_a, label_a, gt, lmap, height=400),
                use_container_width=True)
        with c2:
            st.plotly_chart(
                _confusion_heatmap(res_b, label_b, gt, lmap, height=400),
                use_container_width=True)

        # Per-class comparison + delta
        st.plotly_chart(
            _comparison_bar(res_a, res_b, "A", "B", gt, lmap),
            use_container_width=True)
        st.plotly_chart(
            _delta_bar(res_a, res_b, "A", "B", gt, lmap),
            use_container_width=True)

        # Fold-by-fold comparison
        fig_folds = go.Figure()
        fig_folds.add_trace(go.Scatter(
            x=list(range(len(res_a["per_fold_accuracy"]))),
            y=res_a["per_fold_accuracy"],
            mode="lines+markers", name="A",
            marker=dict(size=8, color="#377eb8"),
        ))
        fig_folds.add_trace(go.Scatter(
            x=list(range(len(res_b["per_fold_accuracy"]))),
            y=res_b["per_fold_accuracy"],
            mode="lines+markers", name="B",
            marker=dict(size=8, color="#e41a1c"),
        ))
        fig_folds.update_layout(
            title="Per-fold accuracy",
            xaxis_title="Fold", yaxis_title="Accuracy",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            height=320, margin=dict(l=0,r=0,b=40,t=50),
        )
        st.plotly_chart(fig_folds, use_container_width=True)


if __name__ == "__main__":
    main()
