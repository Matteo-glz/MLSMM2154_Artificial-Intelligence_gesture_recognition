"""
visualize_pipeline.py  —  Interactive 3D Visualization Dashboard
MLSMM2154 Artificial Intelligence · Gesture Recognition Pipeline

Sections:
  §1  Raw 3D Trajectories          — browse data, filter by class
  §2  PCA Explorer                 — global PCA of trajectory points
  §3  K-Means Cluster Space        — dropdown to switch k ∈ {5,7,9,11,13,15,17,19,21}
  §4  Gesture Path through Clusters— one gesture's cluster-visit sequence
  §5  DTW Alignment                — 3D warping lines + cost-matrix heatmap
  §6  3-Cent Preprocessing         — resample → scale → translate pipeline
  §7  Symbolic Sequence Explorer   — cluster-visit strings, raw & compressed
  §8  Baseline Comparison (kNN)    — nearest neighbours per method for one sample
  §9  Confusion Matrices           — 2×3 grid: CV modes × baselines
  §10 Accuracy & User-dep vs Indep — bar charts with error bars

Usage:
    python visualize_pipeline.py                       # Domain 1, defaults
    python visualize_pipeline.py --domain 4 --open
    python visualize_pipeline.py --skip-baselines      # skip slow CV (no §8-§10)
    python visualize_pipeline.py --skip-dtw            # skip only DTW (fast)
"""

import argparse, os, sys, webbrowser, time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SkPCA
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from data_loading      import load_data_domain_1, load_data_domain_4
from data_preparation  import fit_normalizer, apply_normalizer
from data_splitting    import user_independent_cv, user_dependent_cv
from clustering        import (fit_kmeans, apply_symbolic_transformation,
                               apply_compression, predict_gesture_type_knn)
from tool_from_scratch import compute_dtw_distance_c_speed
from three_cent        import build_templates, recognize, _preprocess, _resample, _scale_by_length, _translate_to_origin

# ── Palettes ─────────────────────────────────────────────────────────────────
GESTURE_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#808080","#b8860b","#17becf",
]
CLUSTER_PAL = px.colors.qualitative.Alphabet   # 26 colours
K_OPTIONS   = [5, 7, 9, 11, 13, 15, 17, 19, 21]
DEFAULT_K   = 7


# ════════════════════════════════════════════════════════════════════════════════
# §1 — Raw 3D Trajectories
# ════════════════════════════════════════════════════════════════════════════════
def fig_trajectories_3d(gestures, max_per_class=6, domain_name="Domain 1"):
    gesture_types = sorted({g["gesture_type"] for g in gestures})
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                 for g in gestures}

    fig = go.Figure()
    trace_gt = []

    for gt in gesture_types:
        examples = [g for g in gestures if g["gesture_type"] == gt][:max_per_class]
        color = GESTURE_COLORS[gt % len(GESTURE_COLORS)]
        for ex_i, g in enumerate(examples):
            traj = g["trajectory"]
            x, y, z = traj[:,0], traj[:,1], traj[:,2]
            n = len(x)
            sizes   = [8] + [2]*(n-2) + [8]
            symbols = ["diamond"] + ["circle"]*(n-2) + ["square"]
            hover = [f"<b>Class {gt} — {label_map[gt]}</b><br>S{g['subject']} Rep{g['repetition']}<br>"
                     f"t={j}  ({xi:.3f}, {yi:.3f}, {zi:.3f})"
                     for j,(xi,yi,zi) in enumerate(zip(x,y,z))]

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=sizes, color=color, symbol=symbols,
                            line=dict(color="black", width=0.4)),
                name=f"Class {gt} — {label_map[gt]}",
                legendgroup=f"gt{gt}", showlegend=(ex_i==0),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
            trace_gt.append(gt)

    buttons = [dict(label="All classes", method="update",
                    args=[{"visible":[True]*len(trace_gt)}])]
    for fgt in gesture_types:
        buttons.append(dict(
            label=f"Class {fgt} — {label_map[fgt]}", method="update",
            args=[{"visible":[t==fgt for t in trace_gt]}]))

    fig.update_layout(
        title=f"<b>§1 — Raw 3D Gesture Trajectories ({domain_name})</b>"
              "<br><sup>◆ = start  ■ = end  |  dropdown filters by class</sup>",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0, y=1.1, xanchor="left", bgcolor="white",
                          bordercolor="#bbb")],
        legend=dict(title="Gesture class", itemsizing="constant"),
        margin=dict(l=0,r=0,b=0,t=90), height=680,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §2 — PCA Explorer
# ════════════════════════════════════════════════════════════════════════════════
def fig_pca_explorer(gestures, max_pts=8000):
    """Global PCA of all trajectory points → 3D scatter + explained-variance bar."""
    all_pts   = np.vstack([g["trajectory"] for g in gestures])
    all_gt    = np.concatenate([[g["gesture_type"]]*len(g["trajectory"]) for g in gestures])
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"])) for g in gestures}

    pca = SkPCA(n_components=3)
    pca.fit(all_pts)
    evr = pca.explained_variance_ratio_

    # Sub-sample for rendering
    if len(all_pts) > max_pts:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pts), max_pts, replace=False)
        pts_d, gt_d = all_pts[idx], all_gt[idx]
    else:
        pts_d, gt_d = all_pts, all_gt

    pcs = pca.transform(pts_d)   # (N, 3)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"scatter3d"},{"type":"bar"}]],
        column_widths=[0.72, 0.28],
        subplot_titles=["Trajectory points in PCA space (PC1–PC3)",
                        "Explained variance per component"],
    )

    gesture_types = sorted(set(gt_d))
    legend_added  = set()
    for gt in gesture_types:
        mask  = gt_d == gt
        color = GESTURE_COLORS[gt % len(GESTURE_COLORS)]
        sl    = gt not in legend_added
        if sl: legend_added.add(gt)
        fig.add_trace(go.Scatter3d(
            x=pcs[mask,0], y=pcs[mask,1], z=pcs[mask,2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.5),
            name=f"Class {gt} — {label_map[gt]}",
            legendgroup=f"gt{gt}", showlegend=sl,
            hovertemplate=f"Class {gt}<extra></extra>",
        ), row=1, col=1)

    # Explained variance bar (cumulative overlay)
    labels = [f"PC{i+1}" for i in range(3)]
    cumul  = np.cumsum(evr)
    fig.add_trace(go.Bar(
        x=labels, y=evr*100,
        marker_color=["#377eb8","#4daf4a","#e41a1c"],
        text=[f"{v:.1f}%" for v in evr*100], textposition="inside",
        name="Individual", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=labels, y=cumul*100, mode="lines+markers",
        line=dict(color="orange", width=2),
        marker=dict(size=8),
        name="Cumulative", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title="<b>§2 — Global PCA of Trajectory Points</b>"
              "<br><sup>Each dot = one 3D coordinate from any gesture, "
              "projected to the first 3 principal components. "
              "Colour = gesture class.</sup>",
        scene=dict(xaxis_title=f"PC1 ({evr[0]*100:.1f}%)",
                   yaxis_title=f"PC2 ({evr[1]*100:.1f}%)",
                   zaxis_title=f"PC3 ({evr[2]*100:.1f}%)"),
        margin=dict(l=0,r=0,b=0,t=90), height=660,
        legend=dict(title="Class", itemsizing="constant"),
    )
    fig.update_yaxes(title_text="Variance explained (%)", row=1, col=2)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §3 — K-Means Cluster Space  (k switcher)
# ════════════════════════════════════════════════════════════════════════════════
def fig_cluster_space_all_k(gestures, all_kmeans: dict, max_pts=6000):
    """
    Pre-computes traces for every k value and uses a dropdown to toggle visibility.
    all_kmeans: {k_value → fitted KMeans}
    """
    all_pts = np.vstack([g["trajectory"] for g in gestures])
    # Use the same random sub-sample for all k (same display points, different colours)
    if len(all_pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(all_pts), max_pts, replace=False)
        pts_d = all_pts[idx]
    else:
        pts_d = all_pts

    fig = go.Figure()
    trace_k = []   # k value of each trace

    for k in K_OPTIONS:
        km         = all_kmeans[k]
        lbl_d      = km.predict(pts_d)
        centroids  = km.cluster_centers_
        is_default = (k == DEFAULT_K)

        for ki in range(k):
            mask  = lbl_d == ki
            color = CLUSTER_PAL[ki % len(CLUSTER_PAL)]

            fig.add_trace(go.Scatter3d(
                x=pts_d[mask,0], y=pts_d[mask,1], z=pts_d[mask,2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.40),
                name=f"Cluster {ki} ({chr(65+ki)})",
                legendgroup=f"cl{ki}", showlegend=is_default,
                visible=is_default,
                hovertemplate=f"k={k}, Cluster {ki} ({chr(65+ki)})<extra></extra>",
            ))
            trace_k.append(k)

            fig.add_trace(go.Scatter3d(
                x=[centroids[ki,0]], y=[centroids[ki,1]], z=[centroids[ki,2]],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol="diamond",
                            line=dict(color="black", width=2)),
                text=[f"<b>{chr(65+ki)}</b>"], textposition="top center",
                showlegend=False, visible=is_default,
                hovertemplate=(f"<b>Centroid {chr(65+ki)}</b> (k={k})<br>"
                               f"({centroids[ki,0]:.3f}, {centroids[ki,1]:.3f}, "
                               f"{centroids[ki,2]:.3f})<extra></extra>"),
            ))
            trace_k.append(k)

    n_tr = len(trace_k)
    buttons = []
    for sel_k in K_OPTIONS:
        vis = [tk == sel_k for tk in trace_k]
        buttons.append(dict(
            label=f"k = {sel_k}  ({sel_k} clusters)",
            method="update",
            args=[{"visible": vis},
                  {"title": (f"<b>§3 — K-Means Cluster Space  (k = {sel_k})</b>"
                             "<br><sup>Trajectory points coloured by cluster. "
                             "Large ◆ = centroid (A, B, C…). "
                             "This is the codebook for symbolic encoding.</sup>")}],
        ))

    fig.update_layout(
        title=f"<b>§3 — K-Means Cluster Space  (k = {DEFAULT_K})</b>"
              "<br><sup>Use the dropdown to change k. "
              "Each colour = one cluster. Large ◆ = centroid.</sup>",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0, y=1.12, xanchor="left",
                          bgcolor="white", bordercolor="#bbb")],
        legend=dict(title=f"Cluster (k={DEFAULT_K})", itemsizing="constant"),
        margin=dict(l=0,r=0,b=0,t=100), height=700,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §4 — Gesture Path through Clusters
# ════════════════════════════════════════════════════════════════════════════════
def fig_gesture_cluster_overlay(gestures, kmeans: KMeans,
                                gesture_class=0, n_examples=4):
    k          = kmeans.n_clusters
    centroids  = kmeans.cluster_centers_
    label_map  = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                  for g in gestures}
    targets    = [g for g in gestures if g["gesture_type"] == gesture_class][:n_examples]
    class_name = label_map.get(gesture_class, str(gesture_class))

    fig = go.Figure()
    first_info = ""
    leg_added  = set()

    for ex_i, g in enumerate(targets):
        traj   = g["trajectory"]
        clbl   = kmeans.predict(traj)
        x,y,z  = traj[:,0], traj[:,1], traj[:,2]

        fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode="lines",
            line=dict(color="rgba(170,170,170,0.5)", width=1.5),
            showlegend=False, hoverinfo="none"))

        for ki in range(k):
            mask  = clbl == ki
            if not np.any(mask): continue
            color = CLUSTER_PAL[ki % len(CLUSTER_PAL)]
            sl    = ki not in leg_added
            if sl: leg_added.add(ki)
            idx   = np.where(mask)[0]
            fig.add_trace(go.Scatter3d(
                x=traj[mask,0], y=traj[mask,1], z=traj[mask,2],
                mode="markers",
                marker=dict(size=5, color=color),
                name=f"Cluster {ki} ({chr(65+ki)})",
                legendgroup=f"cl{ki}", showlegend=sl,
                text=[f"t={ti}, cluster {ki} ({chr(65+ki)})" for ti in idx],
                hovertemplate="%{text}<extra></extra>",
            ))

        if ex_i == 0:
            raw  = "".join(chr(65+l) for l in clbl)
            comp = "".join(c for i,c in enumerate(raw) if i==0 or c!=raw[i-1])
            first_info = (f"S{g['subject']} Rep{g['repetition']} — "
                          f"raw: <b>{raw}</b>  →  compressed: <b>{comp}</b>")

    for ki in range(k):
        color = CLUSTER_PAL[ki % len(CLUSTER_PAL)]
        fig.add_trace(go.Scatter3d(
            x=[centroids[ki,0]], y=[centroids[ki,1]], z=[centroids[ki,2]],
            mode="markers+text",
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(color="black",width=1)),
            text=[f"<b>{chr(65+ki)}</b>"], textposition="top center",
            showlegend=False,
            hovertemplate=f"Centroid {chr(65+ki)}<extra></extra>"))

    fig.update_layout(
        title=(f"<b>§4 — Class '{class_name}' path through cluster space "
               f"(k={k})</b><br><sup>{first_info}</sup>"),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(title="Cluster", itemsizing="constant"),
        margin=dict(l=0,r=0,b=0,t=100), height=680,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §5 — DTW Alignment
# ════════════════════════════════════════════════════════════════════════════════
def _dtw_full(s1, s2):
    n,m = len(s1), len(s2)
    D   = np.full((n+1,m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost   = float(np.linalg.norm(s1[i-1]-s2[j-1]))
            D[i,j] = cost + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
    path = []
    i,j  = n,m
    while i>1 or j>1:
        path.append((i-1,j-1))
        if i==1:   j-=1
        elif j==1: i-=1
        else:
            step = int(np.argmin([D[i-1,j-1], D[i-1,j], D[i,j-1]]))
            if step==0: i-=1; j-=1
            elif step==1: i-=1
            else: j-=1
    path.append((0,0))
    return D[1:,1:], list(reversed(path))


def _dtw_panel(fig, g1, g2, label_map, row, col, col3d, colhm):
    s1,s2 = g1["trajectory"], g2["trajectory"]
    cost_mat, path = _dtw_full(s1,s2)
    dist = cost_mat[-1,-1]
    c1 = GESTURE_COLORS[g1["gesture_type"] % len(GESTURE_COLORS)]
    c2 = GESTURE_COLORS[g2["gesture_type"] % len(GESTURE_COLORS)]
    x1,y1,z1 = s1[:,0],s1[:,1],s1[:,2]
    x2,y2,z2 = s2[:,0],s2[:,1],s2[:,2]

    name1 = f"Class {g1['gesture_type']} ({label_map[g1['gesture_type']]}) S{g1['subject']}"
    name2 = f"Class {g2['gesture_type']} ({label_map[g2['gesture_type']]}) S{g2['subject']}"

    fig.add_trace(go.Scatter3d(x=x1,y=y1,z=z1, mode="lines+markers",
        line=dict(color=c1,width=4), marker=dict(size=3,color=c1),
        name=name1), row=row, col=col3d)
    fig.add_trace(go.Scatter3d(x=x2,y=y2,z=z2, mode="lines+markers",
        line=dict(color=c2,width=4), marker=dict(size=3,color=c2),
        name=name2), row=row, col=col3d)

    step = max(1, len(path)//50)
    for pi,pj in path[::step]:
        fig.add_trace(go.Scatter3d(
            x=[x1[pi],x2[pj]], y=[y1[pi],y2[pj]], z=[z1[pi],z2[pj]],
            mode="lines", line=dict(color="rgba(80,80,80,0.2)",width=1),
            showlegend=False, hoverinfo="none"), row=row, col=col3d)

    fig.add_trace(go.Heatmap(z=cost_mat, colorscale="Viridis",
        showscale=(col==1), name="DTW matrix",
        hovertemplate="i=%{y}, j=%{x}, cost=%{z:.3f}<extra></extra>"),
        row=row, col=colhm)
    fig.add_trace(go.Scatter(
        x=[p[1] for p in path], y=[p[0] for p in path],
        mode="lines", line=dict(color="red",width=2),
        name=f"Warping path (dist={dist:.3f})"), row=row, col=colhm)
    return dist


def fig_dtw_alignment(gestures, class_a=0, class_b=1):
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                 for g in gestures}
    def pick(gt, subj_rank=0):
        cs = sorted({g["subject"] for g in gestures if g["gesture_type"]==gt})
        s  = cs[subj_rank % len(cs)]
        return [g for g in gestures if g["gesture_type"]==gt and g["subject"]==s][0]

    g_aa1 = pick(class_a, 0)
    g_aa2 = pick(class_a, 1)
    g_ab1 = pick(class_a, 0)
    g_ab2 = pick(class_b, 0)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"scatter3d"},{"type":"heatmap"}],
               [{"type":"scatter3d"},{"type":"heatmap"}]],
        column_widths=[0.60,0.40],
        subplot_titles=[
            f"SAME CLASS {class_a} vs {class_a} (two subjects) — 3D alignment",
            "DTW cost matrix",
            f"DIFFERENT classes {class_a} vs {class_b} — 3D alignment",
            "DTW cost matrix",
        ],
        vertical_spacing=0.10,
    )
    d1 = _dtw_panel(fig, g_aa1, g_aa2, label_map, row=1, col=1, col3d=1, colhm=2)
    d2 = _dtw_panel(fig, g_ab1, g_ab2, label_map, row=2, col=1, col3d=1, colhm=2)

    fig.update_layout(
        title=(f"<b>§5 — DTW Alignment</b>"
               f"<br><sup>Top: same class (intra-class dist={d1:.3f})  "
               f"Bottom: different classes (inter-class dist={d2:.3f})  "
               f"→ intra < inter means DTW is discriminative.</sup>"),
        height=900, margin=dict(l=0,r=50,b=0,t=110),
    )
    for r in [1,2]:
        fig.update_xaxes(title_text="Sequence 2 (time)", row=r, col=2)
        fig.update_yaxes(title_text="Sequence 1 (time)", row=r, col=2)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §6 — 3-Cent Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════════════════════
def fig_three_cent_preprocessing(gestures, gesture_class=0, n_points=32, n_examples=3):
    """
    4-panel row for one gesture:
      Original → Resampled → Length-Scaled → Translated to origin
    Repeated for n_examples repetitions so you can see consistency after preprocessing.
    """
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                 for g in gestures}
    targets   = [g for g in gestures if g["gesture_type"]==gesture_class][:n_examples]
    class_name = label_map.get(gesture_class, str(gesture_class))

    steps = ["Original (raw)", f"Resampled ({n_points} pts)", "Length-scaled", "Translated to origin"]
    fig   = make_subplots(rows=1, cols=4,
                          specs=[[{"type":"scatter3d"}]*4],
                          subplot_titles=steps,
                          horizontal_spacing=0.02)

    for ex_i, g in enumerate(targets):
        raw  = g["trajectory"]
        color = GESTURE_COLORS[ex_i % len(GESTURE_COLORS)]
        sl    = (ex_i == 0)

        # Step 0: original
        s0 = raw
        # Step 1: resample
        s1 = _resample(raw, n_points)
        # Step 2: scale by length
        s2 = _scale_by_length(s1)
        # Step 3: translate to origin
        s3 = _translate_to_origin(s2)

        name = f"S{g['subject']} Rep{g['repetition']}"
        for col_i, traj in enumerate([s0,s1,s2,s3], start=1):
            x,y,z = traj[:,0], traj[:,1], traj[:,2]
            fig.add_trace(go.Scatter3d(
                x=x,y=y,z=z, mode="lines+markers",
                line=dict(color=color,width=2),
                marker=dict(size=3, color=color),
                name=name, legendgroup=name,
                showlegend=(col_i==1),
                hovertemplate=f"{name}, step {col_i-1}<extra></extra>",
            ), row=1, col=col_i)

    # Path-distance between examples 0 and 1 after preprocessing
    pdist_raw, pdist_proc = None, None
    if len(targets) >= 2:
        s1_proc = _preprocess(targets[0]["trajectory"], n_points)
        s2_proc = _preprocess(targets[1]["trajectory"], n_points)
        from three_cent import _path_distance
        pdist_proc = _path_distance(s1_proc, s2_proc)
        # raw: resample both to same n_points then compute
        r1 = _resample(targets[0]["trajectory"], n_points)
        r2 = _resample(targets[1]["trajectory"], n_points)
        pdist_raw  = float(np.mean(np.linalg.norm(r1-r2, axis=1)))

    sup = ""
    if pdist_raw is not None:
        sup = (f"Path-distance between S{targets[0]['subject']} and "
               f"S{targets[1]['subject']}:  "
               f"raw={pdist_raw:.4f}  →  after 3-cent={pdist_proc:.4f}")

    fig.update_layout(
        title=f"<b>§6 — 3-Cent Preprocessing Pipeline — Class '{class_name}'</b>"
              f"<br><sup>{sup}</sup>",
        height=500, margin=dict(l=0,r=0,b=0,t=110),
        legend=dict(title="Repetition", itemsizing="constant"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §7 — Symbolic Sequence Explorer
# ════════════════════════════════════════════════════════════════════════════════
def fig_symbolic_sequences(gestures, kmeans: KMeans, gesture_class=0, n_examples=10):
    k         = kmeans.n_clusters
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                 for g in gestures}
    targets   = [g for g in gestures if g["gesture_type"]==gesture_class][:n_examples]
    class_name = label_map.get(gesture_class, str(gesture_class))

    fig        = go.Figure()
    annotations = []
    leg_added   = set()

    for ex_i, g in enumerate(targets):
        clbl      = kmeans.predict(g["trajectory"])
        row_label = f"S{g['subject']} R{g['repetition']}"

        for t_idx, cl in enumerate(clbl):
            color = CLUSTER_PAL[cl % len(CLUSTER_PAL)]
            sl    = cl not in leg_added
            if sl: leg_added.add(cl)
            fig.add_trace(go.Bar(
                x=[1], y=[row_label], orientation="h", base=[t_idx],
                marker=dict(color=color, line=dict(width=0)),
                name=f"Cluster {cl} ({chr(65+cl)})",
                legendgroup=f"cl{cl}", showlegend=sl,
                hovertemplate=f"t={t_idx}, cluster {cl} ({chr(65+cl)})<extra></extra>",
            ))

        raw  = "".join(chr(65+l) for l in clbl)
        comp = "".join(c for i,c in enumerate(raw) if i==0 or c!=raw[i-1])
        annotations.append(dict(
            text=f"<span style='font-family:monospace'>{comp}</span>",
            x=len(clbl)+1, y=row_label,
            xanchor="left", showarrow=False,
            font=dict(size=10),
        ))

    fig.update_layout(
        title=(f"<b>§7 — Symbolic Sequence Explorer — Class '{class_name}' "
               f"(k={k})</b>"
               "<br><sup>Each row = one repetition. Colour = cluster at each "
               "time step. Right = compressed string.</sup>"),
        xaxis_title="Time step",
        yaxis=dict(title="", autorange="reversed"),
        barmode="stack",
        height=max(420, 58*len(targets)+130),
        margin=dict(l=0,r=320,b=60,t=100),
        annotations=annotations,
        legend=dict(title=f"Cluster (k={k})", traceorder="normal",
                    itemsizing="constant"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §8 — Baseline kNN Comparison  (one sample gesture, 3 methods)
# ════════════════════════════════════════════════════════════════════════════════
def fig_baseline_knn_comparison(gestures, kmeans: KMeans,
                                test_gesture_idx: int = 0, k: int = 3,
                                n_points_3cent: int = 32):
    """
    For one test gesture, show the top-k nearest neighbours found by
    edit-distance, DTW, and 3-Cent in a single 3D figure.
    Uses the FULL gesture list as the train set (no CV — just for illustration).
    """
    label_map = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                 for g in gestures}
    test_g    = gestures[test_gesture_idx]
    train_gs  = [g for g in gestures if g["gesture_id"] != test_g["gesture_id"]]

    # ── Edit-distance neighbours ──────────────────────────────────────────────
    train_sym = apply_compression(apply_symbolic_transformation(train_gs, kmeans))
    test_sym  = apply_compression(apply_symbolic_transformation([test_g], kmeans))[0]
    from tool_from_scratch import edit_distance_fast
    ed_dists  = sorted([(edit_distance_fast(test_sym["seq_clean"], tg["seq_clean"]),
                          tg) for tg in train_sym], key=lambda x: x[0])
    ed_nn     = [g for _,g in ed_dists[:k]]

    # ── DTW neighbours ────────────────────────────────────────────────────────
    dtw_dists = sorted([(compute_dtw_distance_c_speed(test_g["trajectory"],
                                                       tg["trajectory"]),
                          tg) for tg in train_gs], key=lambda x: x[0])
    dtw_nn    = [g for _,g in dtw_dists[:k]]

    # ── 3-Cent neighbours ─────────────────────────────────────────────────────
    templates = build_templates(train_gs, n_points_3cent)
    cand_proc = _preprocess(test_g["trajectory"], n_points_3cent)
    from three_cent import _path_distance
    tc_dists  = sorted([(_path_distance(cand_proc, tmpl["preprocessed"]), tmpl)
                         for tmpl in templates], key=lambda x: x[0])
    # match back to original gesture (gesture_type + subject + repetition)
    tc_nn_types = [tmpl["gesture_type"] for _,tmpl in tc_dists[:k]]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()
    METHOD_STYLES = {
        "Test gesture":   dict(color="black",  width=5),
        "Edit-distance":  dict(color="#e41a1c", width=2),
        "DTW":            dict(color="#377eb8", width=2),
        "3-Cent":         dict(color="#4daf4a", width=2),
    }
    traj = test_g["trajectory"]
    fig.add_trace(go.Scatter3d(
        x=traj[:,0],y=traj[:,1],z=traj[:,2], mode="lines+markers",
        line=dict(color="black",width=5),
        marker=dict(size=4,color="black"),
        name=f"Test: class {test_g['gesture_type']} "
             f"({label_map[test_g['gesture_type']]}) "
             f"S{test_g['subject']}",
    ))

    for method, nn_list in [("Edit-distance", ed_nn), ("DTW", dtw_nn)]:
        color = METHOD_STYLES[method]["color"]
        for ri, nn in enumerate(nn_list):
            traj_nn = nn["trajectory"]
            fig.add_trace(go.Scatter3d(
                x=traj_nn[:,0],y=traj_nn[:,1],z=traj_nn[:,2],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                name=f"{method} NN{ri+1}: class {nn['gesture_type']} "
                     f"({label_map[nn['gesture_type']]}) S{nn['subject']}",
                legendgroup=method, showlegend=(ri==0),
                opacity=0.7,
            ))

    # 3-Cent: find original gestures with matching type/subject from templates
    tc_used = set()
    for _, tmpl in tc_dists[:k]:
        matches = [g for g in train_gs
                   if g["gesture_type"]==tmpl["gesture_type"]
                   and g["subject"]==tmpl["subject"]
                   and id(g) not in tc_used]
        if matches:
            g_nn = matches[0]; tc_used.add(id(g_nn))
            fig.add_trace(go.Scatter3d(
                x=g_nn["trajectory"][:,0],
                y=g_nn["trajectory"][:,1],
                z=g_nn["trajectory"][:,2],
                mode="lines",
                line=dict(color="#4daf4a", width=2, dash="dot"),
                name=f"3-Cent NN: class {g_nn['gesture_type']} "
                     f"({label_map[g_nn['gesture_type']]}) S{g_nn['subject']}",
                legendgroup="3-Cent",
                showlegend=(len(tc_used)==1),
                opacity=0.7,
            ))

    from collections import Counter
    pred_ed  = Counter([g["gesture_type"] for g in ed_nn]).most_common(1)[0][0]
    pred_dtw = Counter([g["gesture_type"] for g in dtw_nn]).most_common(1)[0][0]
    pred_tc  = Counter(tc_nn_types).most_common(1)[0][0]
    truth    = test_g["gesture_type"]

    preds_str = (f"Truth: class {truth}  |  "
                 f"Edit-dist pred: {pred_ed} {'OK' if pred_ed==truth else 'WRONG'}  |  "
                 f"DTW pred: {pred_dtw} {'OK' if pred_dtw==truth else 'WRONG'}  |  "
                 f"3-Cent pred: {pred_tc} {'OK' if pred_tc==truth else 'WRONG'}")

    fig.update_layout(
        title=(f"<b>§8 — Baseline kNN Comparison  (k={k})</b>"
               f"<br><sup>{preds_str}</sup>"),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        legend=dict(title="Method / Neighbour", itemsizing="constant"),
        margin=dict(l=0,r=0,b=0,t=110), height=680,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §9 — Confusion Matrices  (2 rows CV modes × 3 cols methods)
# ════════════════════════════════════════════════════════════════════════════════
def fig_confusion_grid(results: dict, gesture_types: list, label_map: dict):
    """
    results format:
      {method: {cv_mode: {"cm": np.ndarray, "accuracy": float, "std": float}}}
    """
    methods  = [m for m in ["edit-distance","dtw","three-cent"] if m in results]
    cv_modes = ["independent","dependent"]
    n_cols   = len(methods)

    titles = []
    for cv_m in cv_modes:
        cv_label = "User-Independent" if cv_m=="independent" else "User-Dependent"
        for mth in methods:
            r = results.get(mth,{}).get(cv_m,{})
            acc = r.get("accuracy", float("nan"))
            std = r.get("std", 0.0)
            titles.append(f"{mth} | {cv_label}<br>acc={acc:.1%}±{std:.1%}")

    fig = make_subplots(
        rows=2, cols=n_cols,
        specs=[[{"type":"heatmap"}]*n_cols]*2,
        subplot_titles=titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
    )

    tick_text = [label_map.get(gt, str(gt)) for gt in gesture_types]

    for r_i, cv_m in enumerate(cv_modes):
        for c_i, mth in enumerate(methods):
            row = r_i+1; col = c_i+1
            r   = results.get(mth,{}).get(cv_m,{})
            cm  = r.get("cm", None)
            if cm is None:
                fig.add_trace(go.Heatmap(z=[[0]],showscale=False,
                    hovertemplate="No data<extra></extra>"), row=row, col=col)
                continue

            # Normalise rows to %
            cm_norm = cm.astype(float)
            row_sum = cm_norm.sum(axis=1, keepdims=True)
            row_sum[row_sum==0] = 1
            cm_pct  = cm_norm / row_sum * 100

            hover_text = [[
                f"True: {label_map.get(gesture_types[i], i)}<br>"
                f"Pred: {label_map.get(gesture_types[j], j)}<br>"
                f"Count: {cm[i,j]}  ({cm_pct[i,j]:.1f}%)"
                for j in range(len(gesture_types))]
                for i in range(len(gesture_types))]

            fig.add_trace(go.Heatmap(
                z=cm_pct,
                x=tick_text, y=tick_text,
                colorscale="Blues", zmin=0, zmax=100,
                text=[[f"{v:.0f}%" for v in row_v] for row_v in cm_pct],
                texttemplate="%{text}",
                textfont=dict(size=9),
                customdata=hover_text,
                hovertemplate="%{customdata}<extra></extra>",
                showscale=(col==n_cols and row==1),
                colorbar=dict(title="%", len=0.45, y=0.75),
            ), row=row, col=col)

            fig.update_xaxes(tickangle=45, row=row, col=col)
            fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title="<b>§9 — Confusion Matrices (row-normalised to %)</b>"
              "<br><sup>Diagonal = correct predictions. "
              "Off-diagonal = confusions. Darker = more frequent.</sup>",
        height=700, margin=dict(l=0,r=50,b=60,t=110),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# §10 — Accuracy & User-dep vs User-indep
# ════════════════════════════════════════════════════════════════════════════════
def fig_accuracy_comparison(results: dict, gesture_types: list, label_map: dict):
    """
    3 sub-figures in one card:
      Left   — grouped bar: method × CV mode (overall accuracy)
      Middle — per-class accuracy, user-independent
      Right  — per-class accuracy, user-dependent
    """
    methods  = [m for m in ["edit-distance","dtw","three-cent"] if m in results]
    cv_modes = ["independent","dependent"]
    cv_labels = {"independent":"User-Indep.","dependent":"User-Dep."}
    bar_col   = {"independent":"#e41a1c","dependent":"#377eb8"}

    tick_text = [f"Class {gt}\n({label_map.get(gt,gt)})" for gt in gesture_types]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type":"bar"},{"type":"bar"},{"type":"bar"}]],
        subplot_titles=[
            "Overall accuracy (mean ± std)",
            "Per-class accuracy — User-Independent",
            "Per-class accuracy — User-Dependent",
        ],
        horizontal_spacing=0.08,
    )

    # ── Left: overall accuracy grouped bar ───────────────────────────────────
    for cv_m in cv_modes:
        x_labels, y_vals, err_vals = [], [], []
        for mth in methods:
            r = results.get(mth,{}).get(cv_m,{})
            x_labels.append(mth)
            y_vals.append(r.get("accuracy", 0.0))
            err_vals.append(r.get("std", 0.0))
        fig.add_trace(go.Bar(
            x=x_labels, y=y_vals,
            error_y=dict(type="data", array=err_vals, visible=True),
            name=cv_labels[cv_m],
            marker_color=bar_col[cv_m],
            text=[f"{v:.1%}" for v in y_vals], textposition="outside",
            legendgroup=cv_m,
        ), row=1, col=1)

    # ── Middle & Right: per-class accuracy ───────────────────────────────────
    best_method = methods[0]   # edit-distance if available
    for col_i, cv_m in enumerate(cv_modes, start=2):
        r   = results.get(best_method,{}).get(cv_m,{})
        cm  = r.get("cm", None)
        if cm is None: continue
        row_sum = cm.sum(axis=1)
        per_cls = np.where(row_sum>0, cm.diagonal()/row_sum, 0.0)
        colors  = [GESTURE_COLORS[gt % len(GESTURE_COLORS)] for gt in gesture_types]
        fig.add_trace(go.Bar(
            x=tick_text, y=per_cls,
            marker_color=colors,
            text=[f"{v:.0%}" for v in per_cls], textposition="outside",
            name=cv_labels[cv_m],
            showlegend=False,
        ), row=1, col=col_i)

    fig.update_layout(
        title=(f"<b>§10 — Accuracy Comparison & User-Dep vs User-Indep</b>"
               f"<br><sup>Per-class breakdown uses '{best_method}'. "
               "User-dependent is easier: the model has seen this user before.</sup>"),
        barmode="group",
        height=520, margin=dict(l=0,r=0,b=100,t=110),
        legend=dict(title="CV Mode"),
        yaxis=dict(title="Mean accuracy", tickformat=".0%", range=[0,1.18]),
        yaxis2=dict(title="Accuracy", tickformat=".0%", range=[0,1.18]),
        yaxis3=dict(title="Accuracy", tickformat=".0%", range=[0,1.18]),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Baseline CV runner
# ════════════════════════════════════════════════════════════════════════════════
def run_baselines(gestures, n_clusters=DEFAULT_K, k_neighbors=3,
                  n_points_3cent=32, run_dtw=True, run_3cent=True):
    """
    Run all baselines for both CV modes.
    Returns:
      {method: {cv_mode: {"cm", "accuracy", "std", "per_fold"}}}
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    gesture_types = sorted({g["gesture_type"] for g in gestures})
    cv_fns = {"independent": user_independent_cv,
              "dependent":   user_dependent_cv}

    results = {}

    # ── Edit-distance ─────────────────────────────────────────────────────────
    print("  Running edit-distance CV…")
    results["edit-distance"] = {}
    for cv_mode, cv_fn in cv_fns.items():
        all_true, all_pred, fold_accs = [], [], []
        for train, test, _ in cv_fn(gestures):
            mu,sigma = fit_normalizer(train)
            train_n  = apply_normalizer(train,mu,sigma)
            test_n   = apply_normalizer(test,mu,sigma)
            km       = fit_kmeans(train_n, n_clusters)
            train_s  = apply_compression(apply_symbolic_transformation(train_n,km))
            test_s   = apply_compression(apply_symbolic_transformation(test_n,km))
            y_t, y_p = [], []
            for tg in test_s:
                pred = predict_gesture_type_knn(tg, train_s, k=k_neighbors)
                y_t.append(tg["gesture_type"]); y_p.append(pred)
            fold_accs.append(np.mean(np.array(y_t)==np.array(y_p)))
            all_true.extend(y_t); all_pred.extend(y_p)
        cm  = sk_cm(all_true, all_pred, labels=gesture_types)
        acc = float(np.mean(fold_accs))
        std = float(np.std(fold_accs))
        results["edit-distance"][cv_mode] = {"cm":cm,"accuracy":acc,"std":std,"per_fold":fold_accs}
        print(f"    edit-distance / {cv_mode}: {acc:.2%} ± {std:.2%}")

    # ── DTW ───────────────────────────────────────────────────────────────────
    if run_dtw:
        print("  Running DTW CV (may take ~30-60 s)…")
        results["dtw"] = {}
        for cv_mode, cv_fn in cv_fns.items():
            all_true, all_pred, fold_accs = [], [], []
            for train, test, _ in cv_fn(gestures):
                mu,sigma = fit_normalizer(train)
                train_n  = apply_normalizer(train,mu,sigma)
                test_n   = apply_normalizer(test,mu,sigma)
                y_t, y_p = [], []
                for tg in test_n:
                    dists = [(compute_dtw_distance_c_speed(tg["trajectory"],
                                                            tr["trajectory"]),
                              tr["gesture_type"]) for tr in train_n]
                    dists.sort(key=lambda x: x[0])
                    nbrs  = [gt for _,gt in dists[:k_neighbors]]
                    from collections import Counter
                    pred  = Counter(nbrs).most_common(1)[0][0]
                    y_t.append(tg["gesture_type"]); y_p.append(pred)
                fold_accs.append(np.mean(np.array(y_t)==np.array(y_p)))
                all_true.extend(y_t); all_pred.extend(y_p)
            cm  = sk_cm(all_true, all_pred, labels=gesture_types)
            acc = float(np.mean(fold_accs))
            std = float(np.std(fold_accs))
            results["dtw"][cv_mode] = {"cm":cm,"accuracy":acc,"std":std,"per_fold":fold_accs}
            print(f"    DTW / {cv_mode}: {acc:.2%} ± {std:.2%}")

    # ── 3-Cent ────────────────────────────────────────────────────────────────
    if run_3cent:
        print("  Running 3-Cent CV…")
        results["three-cent"] = {}
        for cv_mode, cv_fn in cv_fns.items():
            all_true, all_pred, fold_accs = [], [], []
            for train, test, _ in cv_fn(gestures):
                mu,sigma = fit_normalizer(train)
                train_n  = apply_normalizer(train,mu,sigma)
                test_n   = apply_normalizer(test,mu,sigma)
                tmpls    = build_templates(train_n, n_points_3cent)
                y_t, y_p = [], []
                for tg in test_n:
                    pred = recognize(tg["trajectory"], tmpls, n_points_3cent)
                    y_t.append(tg["gesture_type"]); y_p.append(pred)
                fold_accs.append(np.mean(np.array(y_t)==np.array(y_p)))
                all_true.extend(y_t); all_pred.extend(y_p)
            cm  = sk_cm(all_true, all_pred, labels=gesture_types)
            acc = float(np.mean(fold_accs))
            std = float(np.std(fold_accs))
            results["three-cent"][cv_mode] = {"cm":cm,"accuracy":acc,"std":std,"per_fold":fold_accs}
            print(f"    3-Cent / {cv_mode}: {acc:.2%} ± {std:.2%}")

    return results


# ════════════════════════════════════════════════════════════════════════════════
# HTML builder
# ════════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
*{box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#eef0f3;margin:0;padding:22px;color:#222}
h1{text-align:center;font-size:26px;margin-bottom:4px;color:#1a1a2e}
.sub{text-align:center;color:#666;font-size:13px;margin-bottom:22px}
nav{background:#1a1a2e;border-radius:10px;padding:10px 18px;
    display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;
    box-shadow:0 3px 10px rgba(0,0,0,.18)}
nav a{color:#99aacc;text-decoration:none;font-size:13px;
      padding:3px 7px;border-radius:4px;transition:background .15s}
nav a:hover{color:#fff;background:rgba(255,255,255,.12)}
.card{background:#fff;border-radius:12px;
      box-shadow:0 2px 10px rgba(0,0,0,.07);
      margin-bottom:24px;padding:18px 22px}
.hint{color:#666;font-size:12px;border-left:3px solid #dde;
      padding-left:9px;margin:-2px 0 10px}
</style>
"""

_HINTS = {
    "sec1":  "Rotate: left-drag. Zoom: scroll. Pan: right-drag. Hover for coordinates.",
    "sec2":  "Global PCA projects all trajectory points into the space of maximum variance.",
    "sec3":  "Switch k with the dropdown — watch centroids rearrange as granularity changes.",
    "sec4":  "The sequence of cluster letters is what the edit-distance algorithm compares.",
    "sec5":  "Top row: intra-class (same gesture, two subjects). Bottom: inter-class. "
             "Good DTW: intra-distance < inter-distance.",
    "sec6":  "After preprocessing, gestures of the same class overlap closely — "
             "easier to match by path distance.",
    "sec7":  "Each coloured segment = one 'letter' in the symbolic string. "
             "Right side shows the compressed version.",
    "sec8":  "Black = test gesture. Dashed = nearest neighbours. "
             "Each method may pick different neighbours.",
    "sec9":  "Row = true class. Column = predicted class. "
             "Diagonal = correct. Off-diagonal = confusion.",
    "sec10": "User-dependent is easier: training and test data share the same user. "
             "The gap shows how much gesture style varies between users.",
}

def build_html(sections, output_path):
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
             f"<title>Gesture Recognition Dashboard</title>{_CSS}</head><body>"]
    parts.append("<h1>Gesture Recognition — Interactive 3D Dashboard</h1>")
    parts.append("<p class='sub'>MLSMM2154 Artificial Intelligence · "
                 "Gesture Recognition Pipeline</p>")
    parts.append("<nav>")
    for anchor, title, fig in sections:
        if fig is not None:
            parts.append(f"<a href='#{anchor}'>{title}</a>")
    parts.append("</nav>")

    first = True
    for anchor, title, fig in sections:
        if fig is None: continue
        hint = _HINTS.get(anchor, "")
        parts.append(f"<div class='card' id='{anchor}'>")
        if hint:
            parts.append(f"<p class='hint'>{hint}</p>")
        inc = "cdn" if first else False
        parts.append(pio.to_html(fig, full_html=False,
                                 include_plotlyjs=inc,
                                 config={"displayModeBar":True,"scrollZoom":True}))
        first = False
        parts.append("</div>")

    parts.append("</body></html>")
    with open(output_path,"w",encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"Dashboard saved -> {output_path}")


# ════════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Gesture Recognition 3D Dashboard")
    ap.add_argument("--domain",          type=int,   default=1,  choices=[1,4])
    ap.add_argument("--n-clusters",      type=int,   default=DEFAULT_K)
    ap.add_argument("--n-per-class",     type=int,   default=5)
    ap.add_argument("--seq-class",       type=int,   default=0,
                    help="Gesture class shown in §4/§6/§7")
    ap.add_argument("--dtw-class-a",     type=int,   default=0)
    ap.add_argument("--dtw-class-b",     type=int,   default=1)
    ap.add_argument("--n-points-3cent",  type=int,   default=32)
    ap.add_argument("--k-neighbors",     type=int,   default=3)
    ap.add_argument("--skip-baselines",  action="store_true",
                    help="Skip §8–§10 (no CV run)")
    ap.add_argument("--skip-dtw",        action="store_true",
                    help="Skip DTW in §9 (fast mode)")
    ap.add_argument("--output",          type=str,   default="dashboard.html")
    ap.add_argument("--open",            action="store_true")
    args = ap.parse_args()

    base = os.path.dirname(__file__)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.domain == 1:
        data_dir = os.path.join(base,"GestureData_Mons",
                                "GestureDataDomain1_Mons","Domain1_csv")
        print("Loading Domain 1 (digits 0-9)...")
        gestures     = load_data_domain_1(data_dir)
        domain_name  = "Domain 1 — Digits 0-9"
    else:
        data_dir = os.path.join(base,"GestureData_Mons","GestureDataDomain4_Mons")
        print("Loading Domain 4 (3D shapes)...")
        gestures    = load_data_domain_4(data_dir)
        domain_name = "Domain 4 — 3D Shapes"
    print(f"  {len(gestures)} gestures loaded.")

    mean, std   = fit_normalizer(gestures)
    gestures_n  = apply_normalizer(gestures, mean, std)
    gesture_types = sorted({g["gesture_type"] for g in gestures_n})
    label_map   = {g["gesture_type"]: g.get("gesture_name", str(g["gesture_type"]))
                   for g in gestures_n}

    # Clamp user-supplied classes to valid range
    valid_gts     = set(gesture_types)
    seq_class     = args.seq_class     if args.seq_class     in valid_gts else gesture_types[0]
    dtw_class_a   = args.dtw_class_a   if args.dtw_class_a   in valid_gts else gesture_types[0]
    dtw_class_b   = args.dtw_class_b   if args.dtw_class_b   in valid_gts else gesture_types[1]

    # ── Pre-compute K-Means for all k values ──────────────────────────────────
    print(f"Fitting K-Means for k in {K_OPTIONS}...")
    all_pts   = np.vstack([g["trajectory"] for g in gestures_n])
    all_kmeans = {}
    for k in K_OPTIONS:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(all_pts)
        all_kmeans[k] = km
        print(f"  k={k} done")

    default_km = all_kmeans[min(K_OPTIONS, key=lambda x: abs(x-args.n_clusters))]

    # ── Build visualisation figures ───────────────────────────────────────────
    print("Building figures...")

    print("  §1 Raw trajectories...")
    f1 = fig_trajectories_3d(gestures_n, args.n_per_class, domain_name)

    print("  §2 PCA explorer...")
    f2 = fig_pca_explorer(gestures_n)

    print("  §3 Cluster space (all k)...")
    f3 = fig_cluster_space_all_k(gestures_n, all_kmeans)

    print("  §4 Gesture path through clusters...")
    f4 = fig_gesture_cluster_overlay(gestures_n, default_km,
                                     gesture_class=seq_class,
                                     n_examples=min(args.n_per_class, 4))

    print(f"  §5 DTW alignment (classes {dtw_class_a} vs {dtw_class_b})...")
    f5 = fig_dtw_alignment(gestures_n, class_a=dtw_class_a, class_b=dtw_class_b)

    print(f"  §6 3-Cent preprocessing (class {seq_class})...")
    f6 = fig_three_cent_preprocessing(gestures_n, gesture_class=seq_class,
                                      n_points=args.n_points_3cent,
                                      n_examples=min(args.n_per_class, 3))

    print(f"  §7 Symbolic sequences (class {seq_class})...")
    f7 = fig_symbolic_sequences(gestures_n, default_km,
                                gesture_class=seq_class, n_examples=10)

    f8 = f9 = f10 = None
    if not args.skip_baselines:
        print("Running baselines CV (may take 1-3 min)...")
        t0 = time.time()
        baseline_results = run_baselines(
            gestures_n,
            n_clusters      = args.n_clusters,
            k_neighbors     = args.k_neighbors,
            n_points_3cent  = args.n_points_3cent,
            run_dtw         = not args.skip_dtw,
            run_3cent       = True,
        )
        print(f"  CV done in {time.time()-t0:.1f}s")

        print("  §8 Baseline kNN comparison...")
        # Pick a test gesture that all methods should be able to classify
        sample_idx = 0
        f8 = fig_baseline_knn_comparison(gestures_n, default_km,
                                         test_gesture_idx=sample_idx,
                                         k=args.k_neighbors,
                                         n_points_3cent=args.n_points_3cent)

        print("  §9 Confusion matrices...")
        f9 = fig_confusion_grid(baseline_results, gesture_types, label_map)

        print("  §10 Accuracy comparison...")
        f10 = fig_accuracy_comparison(baseline_results, gesture_types, label_map)
    else:
        print("  Skipping §8-§10 (--skip-baselines).")

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    sections = [
        ("sec1",  "§1 Trajectories",          f1),
        ("sec2",  "§2 PCA",                   f2),
        ("sec3",  "§3 Cluster Space (k)",      f3),
        ("sec4",  "§4 Path+Clusters",          f4),
        ("sec5",  "§5 DTW Alignment",          f5),
        ("sec6",  "§6 3-Cent Preprocessing",   f6),
        ("sec7",  "§7 Symbolic Sequences",     f7),
        ("sec8",  "§8 Baseline kNN",           f8),
        ("sec9",  "§9 Confusion Matrices",     f9),
        ("sec10", "§10 Accuracy",              f10),
    ]

    output_path = os.path.join(base, args.output)
    print(f"Writing HTML to {output_path}...")
    build_html(sections, output_path)

    if args.open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
