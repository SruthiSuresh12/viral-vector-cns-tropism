"""
plots.py
--------
Visualization functions for the AAV tropism predictor:
1. ROC + PR curves (classifier performance)
2. Mutational scan heatmap (capsid surface sensitivity map)
3. UMAP of capsid embedding space (colored by tropism label)
4. VR-region bar chart (variable region sensitivity)
5. Per-serotype prediction bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Consistent color palette
LABEL_COLORS = {
    "cns": "#2196F3",        # Blue
    "peripheral": "#FF5722", # Orange-red
    "broad": "#4CAF50",      # Green
    "bbb": "#9C27B0",        # Purple
}

VP1_VR_COLORS = {
    "VR-I": "#E91E63",
    "VR-II": "#9C27B0",
    "VR-III": "#3F51B5",
    "VR-IV": "#2196F3",
    "VR-V": "#00BCD4",
    "VR-VI": "#4CAF50",
    "VR-VII": "#FFEB3B",
    "VR-VIII": "#FF9800",
    "VR-IX": "#F44336",
    "other": "#E0E0E0",
}


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    output_path: str,
    title: str = "ROC Curves — AAV Tropism Classifier",
):
    """Plot per-label ROC curves with AUROC in legend."""
    from sklearn.metrics import roc_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curves
    ax = axes[0]
    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=LABEL_COLORS.get(label, "gray"),
                label=f"{label} (AUC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # PR curves
    from sklearn.metrics import precision_recall_curve, average_precision_score
    ax = axes[1]
    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap = average_precision_score(y_true[:, i], y_prob[:, i])
        ax.plot(rec, prec, lw=2, color=LABEL_COLORS.get(label, "gray"),
                label=f"{label} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC/PR curves → {output_path}")


def plot_mutational_scan_heatmap(
    scan_df: pd.DataFrame,
    sequence: str,
    output_path: str,
    label: str = "bbb",
    title: Optional[str] = None,
    max_positions: int = 200,
):
    """
    Heatmap: positions (x-axis) × mutations (y-axis), colored by delta score.
    Annotates VP1 variable regions.
    """
    AAS = list("ACDEFGHIKLMNPQRSTVWY")

    # Pivot to matrix
    scan_df = scan_df[scan_df["position"] <= max_positions]
    positions = sorted(scan_df["position"].unique())
    n_pos = len(positions)
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    matrix = np.zeros((20, n_pos))
    wt_row = np.full(n_pos, np.nan)

    for _, row in scan_df.iterrows():
        p_idx = pos_to_idx.get(row["position"])
        if p_idx is None:
            continue
        aa_idx = AAS.index(row["mutant_aa"]) if row["mutant_aa"] in AAS else -1
        if aa_idx >= 0:
            matrix[aa_idx, p_idx] = row["delta_score"]

    # Mark WT position in matrix
    for _, row in scan_df.groupby("position").first().reset_index().iterrows():
        p_idx = pos_to_idx.get(row["position"])
        if p_idx is not None and row["wt_aa"] in AAS:
            wt_row[p_idx] = AAS.index(row["wt_aa"])

    vmax = np.percentile(np.abs(matrix), 95)

    fig, (ax_heat, ax_vr) = plt.subplots(
        2, 1, figsize=(min(n_pos * 0.08 + 4, 24), 8),
        height_ratios=[10, 1],
        gridspec_kw={"hspace": 0.05}
    )

    im = ax_heat.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    # Mark WT residues
    for p_idx, aa_idx in enumerate(wt_row):
        if not np.isnan(aa_idx):
            ax_heat.plot(p_idx, int(aa_idx), "k.", markersize=3, alpha=0.6)

    ax_heat.set_yticks(range(20))
    ax_heat.set_yticklabels(AAS, fontsize=7)
    ax_heat.set_ylabel("Substitution", fontsize=10)
    ax_heat.set_xticks([])

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.01)
    cbar.set_label(f"Δ {label.upper()} score", fontsize=9)

    title_str = title or f"Mutational Scan — {label.upper()} tropism\n(Blue = gain, Red = loss)"
    ax_heat.set_title(title_str, fontsize=12, fontweight="bold")

    # VR region annotation bar
    vr_colors_array = []
    from src.models.mutational_scan import assign_vr
    for pos in positions:
        vr = assign_vr(pos - 1)
        vr_colors_array.append(mcolors.to_rgb(VP1_VR_COLORS.get(vr, "#E0E0E0")))

    ax_vr.imshow(
        [vr_colors_array],
        aspect="auto",
        interpolation="nearest",
    )
    ax_vr.set_yticks([0])
    ax_vr.set_yticklabels(["VR"], fontsize=8)
    ax_vr.set_xlabel(f"Residue position (VP1, 1–{max_positions})", fontsize=10)
    ax_vr.set_xticks(range(0, n_pos, max(1, n_pos // 20)))
    ax_vr.set_xticklabels(
        [str(positions[i]) for i in range(0, n_pos, max(1, n_pos // 20))],
        fontsize=7, rotation=45
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mutational scan heatmap → {output_path}")


def plot_embedding_umap(
    embeddings: np.ndarray,
    names: List[str],
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    output_path: str = "results/figures/umap.png",
    title: str = "UMAP of AAV Capsid Embedding Space",
):
    """UMAP projection of capsid sequence space, colored by tropism."""
    try:
        from umap import UMAP
    except ImportError:
        print("[WARN] umap-learn not installed. Skipping UMAP. pip install umap-learn")
        return

    from sklearn.preprocessing import StandardScaler

    print("Computing UMAP projection...")
    X_scaled = StandardScaler().fit_transform(embeddings)
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.3)
    coords = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None and label_names is not None:
        # Color by dominant label
        label_colors_arr = list(LABEL_COLORS.values())
        for i, name in enumerate(names):
            dominant_label_idx = int(np.argmax(labels[i])) if labels[i].max() > 0 else -1
            color = (label_colors_arr[dominant_label_idx]
                     if dominant_label_idx >= 0 else "#9E9E9E")
            ax.scatter(coords[i, 0], coords[i, 1], c=color, s=80, alpha=0.8, zorder=3)
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.7, ha="center", va="bottom")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c="#2196F3", s=80, alpha=0.8)
        for i, name in enumerate(names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.7)

    # Legend
    if label_names:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=LABEL_COLORS[l], label=l)
            for l in label_names if l in LABEL_COLORS
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP → {output_path}")


def plot_vr_sensitivity(
    vr_df: pd.DataFrame,
    output_path: str,
    label: str = "bbb",
    title: Optional[str] = None,
):
    """Bar chart of mean sensitivity per variable region."""
    fig, ax = plt.subplots(figsize=(10, 5))

    vr_df = vr_df.sort_values("mean_abs_delta", ascending=False)
    colors = [VP1_VR_COLORS.get(vr, "#E0E0E0") for vr in vr_df["vr_region"]]

    bars = ax.bar(vr_df["vr_region"], vr_df["mean_abs_delta"],
                  color=colors, edgecolor="black", linewidth=0.8)

    ax.set_xlabel("VP1 Variable Region", fontsize=12)
    ax.set_ylabel("Mean |Δ Score|", fontsize=12)
    ax.set_title(title or f"Per-Region Sensitivity to Mutation\n(Target: {label.upper()} tropism)",
                 fontsize=13, fontweight="bold")

    # Value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"VR sensitivity plot → {output_path}")


def plot_serotype_predictions(
    predictions_csv: str,
    output_path: str,
    title: str = "AAV Serotype Tropism Predictions",
):
    """Grouped bar chart showing predicted scores for each serotype."""
    df = pd.read_csv(predictions_csv)
    LABEL_COLS = ["cns", "peripheral", "broad", "bbb"]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df))
    width = 0.2

    for i, label in enumerate(LABEL_COLS):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df[label].astype(float),
                      width, label=label,
                      color=LABEL_COLORS.get(label, "gray"),
                      alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Predicted Probability", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.legend(title="Tropism Label", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Serotype predictions chart → {output_path}")
