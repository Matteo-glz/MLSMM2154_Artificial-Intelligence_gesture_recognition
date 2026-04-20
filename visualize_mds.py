import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import MDS

sys.path.insert(0, os.path.dirname(__file__))

from data_loading import load_data_domain_1, load_data_domain_4
from data_preparation import fit_normalizer, apply_normalizer
from tool_from_scratch import compute_dtw_distance_c_speed

# ─────────────────────── CONFIG ──────────────────────────────────────────────
PATH_DOMAIN_1 = r"C:\Users\PC\Documents\GitHub\MLSMM2154_Artificial-Intelligence_gesture_recognition\GestureData_Mons\GestureDataDomain1_Mons\Domain1_csv"
PATH_DOMAIN_4 = r"C:\Users\PC\Documents\GitHub\MLSMM2154_Artificial-Intelligence_gesture_recognition\GestureData_Mons\GestureDataDomain4_Mons"

DOMAIN       = 1     # 1 or 4
MAX_GESTURES = None  # e.g. 80 to limit computation; None = all
# ─────────────────────────────────────────────────────────────────────────────

GESTURE_NAMES = {
    0: "Cuboid",       1: "Cylinder",      2: "Sphere",
    3: "Rect. Pipe",   4: "Hemisphere",    5: "Cylinder Pipe",
    6: "Pyramid",      7: "Tetrahedron",   8: "Cone",
    9: "Toroid",
}


def compute_dtw_matrix(gestures):
    n = len(gestures)
    D = np.zeros((n, n))
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_dtw_distance_c_speed(
                gestures[i]["trajectory"].astype(np.float64),
                gestures[j]["trajectory"].astype(np.float64),
            )
            D[i, j] = D[j, i] = d
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  DTW {done}/{total}  ({100 * done / total:.1f}%)", end="\r", flush=True)
    print()
    return D


def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                  angle=angle, **kwargs)
    ax.add_patch(ell)


def plot_mds(coords, gestures, stress, title_suffix=""):
    labels   = np.array([g["gesture_type"] for g in gestures])
    subjects = np.array([g["subject"]      for g in gestures])

    unique_classes  = sorted(set(labels))
    unique_subjects = sorted(set(subjects))

    cmap_class = plt.cm.get_cmap("tab10", len(unique_classes))
    cmap_subj  = plt.cm.get_cmap("Set1",  len(unique_subjects))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"MDS on DTW distance matrix{title_suffix}\n"
        f"Stress: {stress:.4f}  ({len(gestures)} gestures)",
        fontsize=13, fontweight="bold",
    )

    # ── Left: colored by gesture class ───────────────────────────────────
    ax = axes[0]
    for i, cls in enumerate(unique_classes):
        mask = labels == cls
        name = GESTURE_NAMES.get(cls, f"Class {cls}")
        color = cmap_class(i)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=color, label=name, s=55, alpha=0.85,
            edgecolors="white", linewidths=0.4, zorder=3,
        )
        confidence_ellipse(
            coords[mask, 0], coords[mask, 1], ax,
            n_std=1.5, facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.2, zorder=2,
        )

    ax.set_title("Colored by Gesture Class", fontsize=11)
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", "datalim")

    # ── Right: colored by subject ─────────────────────────────────────────
    ax = axes[1]
    for i, subj in enumerate(unique_subjects):
        mask = subjects == subj
        color = cmap_subj(i)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=color, label=f"Subject {subj}", s=55, alpha=0.85,
            edgecolors="white", linewidths=0.4, zorder=3,
        )
        confidence_ellipse(
            coords[mask, 0], coords[mask, 1], ax,
            n_std=1.5, facecolor=color, alpha=0.10,
            edgecolor=color, linewidth=1.2, zorder=2,
        )

    ax.set_title("Colored by Subject", fontsize=11)
    ax.set_xlabel("MDS Dimension 1")
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", "datalim")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mds_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    # 1. Load
    print(f"Loading Domain {DOMAIN}…")
    if DOMAIN == 4:
        gestures = load_data_domain_4(PATH_DOMAIN_4)
        title_suffix = " — Domain 4"
    else:
        gestures = load_data_domain_1(PATH_DOMAIN_1)
        title_suffix = " — Domain 1"

    if MAX_GESTURES:
        gestures = gestures[:MAX_GESTURES]

    print(f"  {len(gestures)} gestures loaded.")

    # 2. Normalize (full-dataset stats — fine for visualization, no CV needed)
    mean, std = fit_normalizer(gestures)
    gestures = apply_normalizer(gestures, mean, std)

    # 3. Pairwise DTW distance matrix
    print(f"Computing DTW matrix ({len(gestures)} × {len(gestures)})…")
    D = compute_dtw_matrix(gestures)

    # 4. MDS embedding
    print("Fitting MDS…")
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, n_init=4, max_iter=600,
                  normalized_stress="auto")
    except TypeError:
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, n_init=4, max_iter=600)

    coords = mds.fit_transform(D)
    print(f"  Stress: {mds.stress_:.4f}")

    # 5. Plot
    plot_mds(coords, gestures, mds.stress_, title_suffix)
