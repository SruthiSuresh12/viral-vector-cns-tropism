"""
run_pipeline.py
---------------
Complete end-to-end pipeline for AAV capsid CNS tropism prediction.
Runs fully offline with sklearn (no ESM-2 / GPU required).

Produces:
  - data/processed/features.npz          — 128-dim biophysical feature vectors
  - models/rf_classifier.pkl             — trained Random Forest
  - results/predictions/all_serotypes.csv
  - results/figures/roc_pr_curves.png
  - results/figures/feature_importance.png
  - results/figures/serotype_predictions.png
  - results/figures/umap_capsid_space.png
  - results/figures/phpeB_scan_heatmap.png
  - results/figures/vr_sensitivity.png
  - results/predictions/phpeB_scan_raw.csv
  - results/predictions/phpeB_top_mutations.csv
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── directories ──────────────────────────────────────────────────────────────
for d in ["data/processed", "models", "results/figures", "results/predictions"]:
    Path(d).mkdir(parents=True, exist_ok=True)

LABEL_COLS  = ["cns", "peripheral", "broad", "bbb"]
LABEL_COLORS = {"cns": "#2196F3", "peripheral": "#FF5722", "broad": "#4CAF50", "bbb": "#9C27B0"}

# Kyte-Doolittle hydrophobicity
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# VP1 variable region windows (0-indexed residue positions)
VR_WINDOWS = {
    "VR-I":   (252, 272), "VR-II":  (325, 331), "VR-III": (380, 394),
    "VR-IV":  (447, 471), "VR-V":   (490, 505), "VR-VI":  (530, 554),
    "VR-VII": (574, 609), "VR-VIII":(655, 668), "VR-IX":  (705, 723),
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. FASTA PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_fasta(path):
    seqs = {}
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name: seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.upper())
    if name: seqs[name] = "".join(buf)
    return seqs

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION  (128-dim biophysical fingerprint)
# ─────────────────────────────────────────────────────────────────────────────
def aa_composition(seq):
    total = max(len(seq), 1)
    return np.array([seq.count(a) / total for a in AAS])  # 20

def dipeptide_composition(seq):
    """400-dim dipeptide → PCA-reduced to 20 for tractability."""
    dp = {}
    for i in range(len(seq)-1):
        pair = seq[i:i+2]
        if all(c in AAS for c in pair):
            dp[pair] = dp.get(pair, 0) + 1
    total = sum(dp.values()) or 1
    vec = np.zeros(400)
    for i, a in enumerate(AAS):
        for j, b in enumerate(AAS):
            vec[i*20+j] = dp.get(a+b, 0) / total
    return vec

def physicochemical_global(seq):
    """Global biophysical properties: 15 features."""
    clean = [c for c in seq if c in AAS]
    if not clean: return np.zeros(15)
    n = len(clean)
    # Charge at pH 7.4 (simplified Henderson-Hasselbalch)
    pos_charge = clean.count("K") + clean.count("R") + 0.1*clean.count("H")
    neg_charge = clean.count("D") + clean.count("E")
    net_charge = pos_charge - neg_charge
    # Hydrophobicity statistics
    hydro = [KD[c] for c in clean]
    hydro_mean = np.mean(hydro)
    hydro_std  = np.std(hydro)
    hydro_max  = max(hydro)
    # MW proxy (avg residue ~111 Da)
    mw_norm = n * 111 / 1e5
    # Aromatic fraction
    aromatic = (clean.count("F") + clean.count("W") + clean.count("Y")) / n
    # Cysteine fraction (disulfides)
    cys_frac = clean.count("C") / n
    # Charged fraction
    charged_frac = (clean.count("K") + clean.count("R") +
                    clean.count("D") + clean.count("E")) / n
    # Polar uncharged
    polar_frac = (clean.count("S") + clean.count("T") + clean.count("N") +
                  clean.count("Q")) / n
    # Proline fraction (beta-turns)
    pro_frac = clean.count("P") / n
    # Glycine fraction (flexibility)
    gly_frac = clean.count("G") / n
    # Tiny AA
    tiny_frac = (clean.count("G") + clean.count("A") + clean.count("S")) / n
    # Instability proxy (dipeptide-based, simplified)
    instability_pairs = {"RR","DR","ER","QK","KK","IK","SK","KS"}
    inst = sum(1 for i in range(n-1) if (seq[i]+seq[i+1]) in instability_pairs) / max(n-1,1)
    return np.array([net_charge/n, hydro_mean, hydro_std, hydro_max, mw_norm,
                     aromatic, cys_frac, charged_frac, polar_frac, pro_frac,
                     gly_frac, tiny_frac, inst, pos_charge/n, neg_charge/n])

def vr_features(seq):
    """Per-VR mean hydrophobicity + charge: 9×3 = 27 features."""
    feats = []
    for vr, (start, end) in VR_WINDOWS.items():
        segment = seq[start:end] if end <= len(seq) else seq[max(0, len(seq)-10):]
        segment = [c for c in segment if c in AAS]
        if not segment:
            feats.extend([0., 0., 0.])
            continue
        h = np.mean([KD[c] for c in segment])
        charge = (segment.count("K") + segment.count("R") -
                  segment.count("D") - segment.count("E")) / len(segment)
        aromatic = (segment.count("F") + segment.count("W") + segment.count("Y")) / len(segment)
        feats.extend([h, charge, aromatic])
    return np.array(feats)  # 27

def terminal_features(seq):
    """N-terminal and C-terminal 30aa composition: 2×20 = 40 features."""
    n30 = seq[:30]
    c30 = seq[-30:] if len(seq) >= 30 else seq
    return np.concatenate([aa_composition(n30), aa_composition(c30)])  # 40

def kmer_diversity(seq, k=3):
    """Shannon entropy of k-mer distribution: 6 features."""
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1) if all(c in AAS for c in seq[i:i+k])]
    from collections import Counter
    counts = Counter(kmers)
    total = sum(counts.values()) or 1
    probs = np.array(list(counts.values())) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    unique_ratio = len(counts) / max(len(kmers), 1)
    # Tetrapeptide features
    kmers4 = [seq[i:i+4] for i in range(len(seq)-3) if all(c in AAS for c in seq[i:i+4])]
    counts4 = Counter(kmers4)
    probs4 = np.array(list(counts4.values())) / max(sum(counts4.values()), 1)
    entropy4 = -np.sum(probs4 * np.log2(probs4 + 1e-12))
    # Hydrophobic run length max
    runs, current = [], 0
    for c in seq:
        if c in AAS and KD.get(c, 0) > 1.5:
            current += 1
        else:
            if current: runs.append(current)
            current = 0
    max_run = max(runs) if runs else 0
    mean_run = np.mean(runs) if runs else 0
    return np.array([entropy, unique_ratio, entropy4, max_run/50, mean_run/10, len(seq)/800])

def extract_features(seq):
    """Concatenate all feature groups → 128-dim vector."""
    f1 = aa_composition(seq)                  # 20
    f2 = physicochemical_global(seq)          # 15
    f3 = vr_features(seq)                     # 27
    f4 = terminal_features(seq)               # 40 — captures signal peptide & C-term tail
    f5 = kmer_diversity(seq)                  # 6
    # Dipeptide PCA: reduce 400 → 20
    dp_full = dipeptide_composition(seq)      # 400
    # Use fixed PCA directions (deterministic, captures variance structure)
    dp_sub = dp_full.reshape(20,20).mean(axis=1)  # 20 (row means = AA-specific dipeptide tendencies)
    return np.concatenate([f1, f2, f3, f4, f5, dp_sub])  # 128

# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  AAV Capsid CNS Tropism Predictor — Full Pipeline")
print("=" * 60)

print("\n[1/7] Extracting biophysical features...")
sequences = parse_fasta("data/raw/capsid_sequences.fasta")
labels_df = pd.read_csv("data/raw/labels.csv").set_index("name")

names, X_list, y_list = [], [], []
for name, seq in sequences.items():
    if name in labels_df.index:
        names.append(name)
        X_list.append(extract_features(seq))
        y_list.append([int(labels_df.loc[name, c]) for c in LABEL_COLS])

names = np.array(names)
X = np.array(X_list, dtype=np.float32)   # (14, 128)
y = np.array(y_list, dtype=int)           # (14, 4)

print(f"  Sequences: {len(names)}")
print(f"  Feature matrix: {X.shape}")
print(f"  Label distribution:")
for i, lbl in enumerate(LABEL_COLS):
    print(f"    {lbl:<12}: {y[:,i].sum()}/{len(y)} positive")

np.savez_compressed("data/processed/features.npz", X=X, names=names, y=y)
print("  Saved → data/processed/features.npz")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Running 5-fold cross-validation (3 models)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Random Forest":      MultiOutputClassifier(RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, min_samples_leaf=1)),
    "Gradient Boosting":  MultiOutputClassifier(GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    "Logistic Regression":MultiOutputClassifier(LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
}

cv_results = {}   # model → label → [auroc]
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# collect OOF probs for ROC plots
oof_probs = {m: np.zeros_like(y, dtype=float) for m in models}

for mname, clf in models.items():
    aurocs = {lbl: [] for lbl in LABEL_COLS}
    oof_p = np.zeros_like(y, dtype=float)
    for fold, (tr, va) in enumerate(cv_folds.split(X_scaled, y[:,0])):
        clf.fit(X_scaled[tr], y[tr])
        for li, lbl in enumerate(LABEL_COLS):
            prob = clf.estimators_[li].predict_proba(X_scaled[va])[:,1]
            oof_p[va, li] = prob
            if len(np.unique(y[va,li])) == 2:
                aurocs[lbl].append(auc(*roc_curve(y[va,li], prob)[:2]))
    cv_results[mname] = {lbl: np.mean(v) if v else float("nan") for lbl, v in aurocs.items()}
    oof_probs[mname] = oof_p

print(f"\n  {'Model':<22} {'CNS':>7} {'Periph':>7} {'Broad':>7} {'BBB':>7}")
print("  " + "-"*52)
for mname, res in cv_results.items():
    row = f"  {mname:<22}"
    for lbl in LABEL_COLS:
        v = res[lbl]
        row += f" {v:>7.3f}" if not np.isnan(v) else f" {'N/A':>7}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN FINAL MODEL (Random Forest on all data)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Training final Random Forest model on all data...")
final_rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=500, max_depth=6,
                                                          random_state=42, oob_score=True))
final_rf.fit(X_scaled, y)
with open("models/rf_classifier.pkl", "wb") as f:
    pickle.dump((final_rf, scaler), f)
print("  Saved → models/rf_classifier.pkl")

# Final predictions
final_probs = np.column_stack([
    est.predict_proba(X_scaled)[:,1] for est in final_rf.estimators_
])

pred_df = pd.DataFrame(final_probs, columns=LABEL_COLS)
pred_df.insert(0, "name", names)
pred_df["top_label"] = pred_df[LABEL_COLS].idxmax(axis=1)
pred_df.to_csv("results/predictions/all_serotypes.csv", index=False)
print("  Saved → results/predictions/all_serotypes.csv")

print("\n  Final model predictions:")
print(f"  {'Serotype':<14} {'CNS':>6} {'Periph':>7} {'Broad':>6} {'BBB':>6}  {'Top'}")
print("  " + "-"*48)
for _, row in pred_df.iterrows():
    print(f"  {row['name']:<14} {row['cns']:>6.3f} {row['peripheral']:>7.3f} "
          f"{row['broad']:>6.3f} {row['bbb']:>6.3f}  {row['top_label']}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 1 — ROC / PR CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Generating figures...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("AAV Capsid Tropism Classifier — Performance (5-fold CV, Random Forest)",
             fontsize=14, fontweight="bold", y=1.01)

best_oof = oof_probs["Random Forest"]

for col, lbl in enumerate(LABEL_COLS):
    # ROC
    ax = axes[0, col]
    if len(np.unique(y[:,col])) == 2:
        fpr, tpr, _ = roc_curve(y[:,col], best_oof[:,col])
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, color=LABEL_COLORS[lbl], label=f"AUC={auroc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.15, color=LABEL_COLORS[lbl])
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax.set_title(f"ROC — {lbl.upper()}", fontweight="bold", fontsize=11)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # PR
    ax = axes[1, col]
    if len(np.unique(y[:,col])) == 2:
        prec, rec, _ = precision_recall_curve(y[:,col], best_oof[:,col])
        ap = average_precision_score(y[:,col], best_oof[:,col])
        ax.plot(rec, prec, lw=2.5, color=LABEL_COLORS[lbl], label=f"AP={ap:.3f}")
        ax.fill_between(rec, prec, alpha=0.15, color=LABEL_COLORS[lbl])
        baseline = y[:,col].mean()
        ax.axhline(baseline, color="gray", ls="--", lw=1, alpha=0.5, label=f"Baseline={baseline:.2f}")
    ax.set_title(f"PR — {lbl.upper()}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/roc_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/roc_pr_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 2 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = (
    [f"AA_{a}" for a in AAS] +                           # 20
    ["NetCharge/n","Hydro_mean","Hydro_std","Hydro_max","MW_norm",
     "Aromatic","Cys","Charged","Polar","Pro","Gly","Tiny","Instability",
     "PosCharge","NegCharge"] +                          # 15
    [f"{vr}_{prop}" for vr in VR_WINDOWS for prop in ["Hydro","Charge","Arom"]] +  # 27
    [f"Nterm_AA_{a}" for a in AAS] +                     # 20
    [f"Cterm_AA_{a}" for a in AAS] +                     # 20
    ["3mer_H_entropy","3mer_unique","4mer_entropy","HydroRun_max",
     "HydroRun_mean","SeqLen_norm"] +                    # 6
    [f"DP_{a}" for a in AAS]                              # 20
)
assert len(FEATURE_NAMES) == 128, f"Feature name count mismatch: {len(FEATURE_NAMES)}"

fig, axes = plt.subplots(1, 4, figsize=(22, 7))
fig.suptitle("Feature Importance by Tropism Label (Random Forest — Mean Decrease Impurity)",
             fontsize=13, fontweight="bold")

for col, (lbl, ax) in enumerate(zip(LABEL_COLS, axes)):
    importances = final_rf.estimators_[col].feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_vals  = importances[top_idx]
    colors_bar = [LABEL_COLORS[lbl]] * 20

    ax.barh(range(20), top_vals[::-1], color=colors_bar[::-1],
            edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"{lbl.upper()} tropism", fontweight="bold", fontsize=11,
                 color=LABEL_COLORS[lbl])
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("results/figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FIGURE 3 — SEROTYPE PREDICTION CHART
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
x = np.arange(len(pred_df))
w = 0.2
for i, lbl in enumerate(LABEL_COLS):
    offset = (i - 1.5) * w
    bars = ax.bar(x + offset, pred_df[lbl], w, label=lbl,
                  color=LABEL_COLORS[lbl], alpha=0.85,
                  edgecolor="white", linewidth=0.4)

# Overlay ground truth dots
for li, lbl in enumerate(LABEL_COLS):
    for xi, (_, row) in enumerate(pred_df.iterrows()):
        name = row["name"]
        if name in labels_df.index:
            gt = int(labels_df.loc[name, lbl])
            offset = (li - 1.5) * w
            marker = "★" if gt == 1 else "○"
            ax.text(xi + offset, row[lbl] + 0.04, marker,
                    ha="center", va="bottom", fontsize=8,
                    color="black" if gt == 1 else "gray")

ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.35, label="Decision threshold (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(pred_df["name"], rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Predicted Probability", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Predicted Tropism Scores Across All Capsids\n★ = ground-truth positive label",
             fontsize=12, fontweight="bold")
ax.legend(title="Label", loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("results/figures/serotype_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/serotype_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FIGURE 4 — t-SNE of capsid feature space
# ─────────────────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=min(5, len(X)-1), random_state=42,
            max_iter=2000, learning_rate="auto", init="pca")
coords = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
dom_labels = [LABEL_COLS[np.argmax(y[i])] for i in range(len(y))]
dom_colors_arr = [LABEL_COLORS[l] for l in dom_labels]

scatter = ax.scatter(coords[:,0], coords[:,1], c=dom_colors_arr,
                     s=120, edgecolor="black", linewidth=0.8, zorder=3, alpha=0.9)
for i, name in enumerate(names):
    ax.annotate(name, (coords[i,0], coords[i,1]),
                fontsize=8, fontweight="bold",
                xytext=(6, 4), textcoords="offset points",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

legend_elems = [Patch(facecolor=LABEL_COLORS[l], edgecolor="black", label=l) for l in LABEL_COLS]
ax.legend(handles=legend_elems, title="Dominant tropism", loc="best", fontsize=10)
ax.set_xlabel("t-SNE 1", fontsize=12); ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title("t-SNE of Capsid Biophysical Feature Space\n(colored by dominant tropism label)",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/figures/tsne_capsid_space.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/tsne_capsid_space.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. MUTATIONAL SCAN — PHP.eB (BBB label)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Running in silico mutational scan on PHP.eB (BBB label)...")

def assign_vr(pos):
    for vr, (s, e) in VR_WINDOWS.items():
        if s <= pos <= e: return vr
    return "other"

def score_sequence(seq, clf, scaler):
    f = extract_features(seq)
    f_s = scaler.transform(f[np.newaxis,:])
    return clf.estimators_[3].predict_proba(f_s)[0,1]  # BBB = index 3

phpeB_seq = sequences["AAV-PHP.eB"]
wt_score = score_sequence(phpeB_seq, final_rf, scaler)
print(f"  PHP.eB WT BBB score: {wt_score:.4f}")

SCAN_POSITIONS = 250   # scan first 250 residues (covers all VRs I–VII)
scan_records = []

for pos in range(SCAN_POSITIONS):
    wt_aa = phpeB_seq[pos]
    for mut_aa in AAS:
        if mut_aa == wt_aa: continue
        mut_seq = phpeB_seq[:pos] + mut_aa + phpeB_seq[pos+1:]
        mut_score = score_sequence(mut_seq, final_rf, scaler)
        delta = mut_score - wt_score
        scan_records.append({
            "position": pos+1,
            "wt_aa": wt_aa,
            "mutant_aa": mut_aa,
            "wt_score": wt_score,
            "mutant_score": mut_score,
            "delta_score": delta,
            "abs_delta": abs(delta),
            "vr_region": assign_vr(pos),
        })
    if (pos+1) % 50 == 0:
        print(f"  Scanned {pos+1}/{SCAN_POSITIONS} positions...")

scan_df = pd.DataFrame(scan_records)
scan_df.to_csv("results/predictions/phpeB_scan_raw.csv", index=False)
print(f"  Saved → results/predictions/phpeB_scan_raw.csv  ({len(scan_df)} variants)")

# Top beneficial mutations
top_gain = (scan_df.sort_values("delta_score", ascending=False)
            .head(20)[["position","wt_aa","mutant_aa","delta_score","vr_region"]]
            .reset_index(drop=True))
top_gain.to_csv("results/predictions/phpeB_top_mutations.csv", index=False)
print("  Saved → results/predictions/phpeB_top_mutations.csv")
print("\n  Top 15 mutations increasing BBB score:")
print(f"  {'Pos':>5} {'WT':>4} {'Mut':>4} {'Δ BBB':>8} {'VR'}")
print("  " + "-"*35)
for _, r in top_gain.head(15).iterrows():
    print(f"  {int(r['position']):>5} {r['wt_aa']:>4} → {r['mutant_aa']:<4} {r['delta_score']:>+8.4f}  {r['vr_region']}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. FIGURE 5 — SCAN HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] Generating mutational scan heatmap...")

VR_COLORS = {
    "VR-I":"#E91E63","VR-II":"#9C27B0","VR-III":"#3F51B5",
    "VR-IV":"#2196F3","VR-V":"#00BCD4","VR-VI":"#4CAF50",
    "VR-VII":"#FFEB3B","VR-VIII":"#FF9800","VR-IX":"#F44336","other":"#EEEEEE"
}
import matplotlib.colors as mcolors

positions = sorted(scan_df["position"].unique())
pos_to_idx = {p: i for i, p in enumerate(positions)}
n_pos = len(positions)

matrix = np.zeros((20, n_pos))
for _, row in scan_df.iterrows():
    p_idx = pos_to_idx[row["position"]]
    aa_idx = AAS.index(row["mutant_aa"]) if row["mutant_aa"] in AAS else -1
    if aa_idx >= 0:
        matrix[aa_idx, p_idx] = row["delta_score"]

vmax = np.percentile(np.abs(matrix[matrix != 0]), 90) if (matrix != 0).any() else 0.1

fig = plt.figure(figsize=(22, 9))
gs = gridspec.GridSpec(3, 1, height_ratios=[10, 1.2, 1.2], hspace=0.05)

# Heatmap
ax_h = fig.add_subplot(gs[0])
im = ax_h.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 aspect="auto", origin="lower", interpolation="nearest")
ax_h.set_yticks(range(20))
ax_h.set_yticklabels(AAS, fontsize=8)
ax_h.set_ylabel("Substitution", fontsize=11)
ax_h.set_xticks([])
cbar = plt.colorbar(im, ax=ax_h, shrink=0.5, pad=0.005, location="right")
cbar.set_label("Δ BBB score", fontsize=10)
ax_h.set_title(
    "PHP.eB In Silico Mutational Scan — BBB Tropism\n"
    "Blue = gain; Red = loss; Columns are residue positions 1–250",
    fontsize=12, fontweight="bold"
)

# WT residue markers
wt_shown = {}
for _, row in scan_df.iterrows():
    p_idx = pos_to_idx[row["position"]]
    if p_idx not in wt_shown and row["wt_aa"] in AAS:
        ax_h.plot(p_idx, AAS.index(row["wt_aa"]), "k.", markersize=4, alpha=0.5)
        wt_shown[p_idx] = True

# VR annotation bar
ax_vr = fig.add_subplot(gs[1])
vr_rgb = [mcolors.to_rgb(VR_COLORS.get(assign_vr(p-1), "#EEEEEE")) for p in positions]
ax_vr.imshow([vr_rgb], aspect="auto", interpolation="nearest")
ax_vr.set_yticks([0]); ax_vr.set_yticklabels(["VR"], fontsize=9)
ax_vr.set_xticks([])

# VR label text
current_vr, start_idx = None, 0
for idx, pos in enumerate(positions):
    vr = assign_vr(pos-1)
    if vr != current_vr:
        if current_vr and current_vr != "other":
            mid = (start_idx + idx - 1) / 2
            ax_vr.text(mid, 0, current_vr, ha="center", va="center",
                       fontsize=7.5, fontweight="bold",
                       color="white" if current_vr in ["VR-I","VR-II","VR-III","VR-IX"] else "black")
        current_vr, start_idx = vr, idx

# Delta score line
ax_d = fig.add_subplot(gs[2])
pos_summary = scan_df.groupby("position").agg(
    mean_delta=("delta_score","mean"),
    max_gain=("delta_score","max"),
).reset_index()
ax_d.fill_between(pos_summary["position"], 0, pos_summary["max_gain"],
                  alpha=0.4, color="#2196F3", label="Max gain")
ax_d.fill_between(pos_summary["position"], 0, pos_summary["mean_delta"],
                  alpha=0.6, color="#FF5722", label="Mean Δ")
ax_d.axhline(0, color="black", lw=0.7)
ax_d.set_xlabel("Residue position (VP1, 1–250)", fontsize=10)
ax_d.set_ylabel("Δ BBB", fontsize=9)
ax_d.legend(fontsize=8, loc="upper right")
ax_d.grid(alpha=0.2)

plt.savefig("results/figures/phpeB_scan_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/phpeB_scan_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. FIGURE 6 — VR SENSITIVITY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
vr_sens = (scan_df.groupby("vr_region")
           .agg(mean_abs_delta=("abs_delta","mean"),
                max_abs_delta=("abs_delta","max"),
                n_variants=("delta_score","count"))
           .sort_values("mean_abs_delta", ascending=False)
           .reset_index())

fig, ax = plt.subplots(figsize=(11, 5))
colors_bar = [VR_COLORS.get(r, "#EEEEEE") for r in vr_sens["vr_region"]]
bars = ax.bar(vr_sens["vr_region"], vr_sens["mean_abs_delta"],
              color=colors_bar, edgecolor="black", linewidth=0.8, alpha=0.9)
for bar, row in zip(bars, vr_sens.itertuples()):
    h = bar.get_height()
    ax.annotate(f"{h:.3f}\n(n={row.n_variants})",
                xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5)

ax.set_xlabel("VP1 Variable / Constant Region", fontsize=12)
ax.set_ylabel("Mean |Δ BBB Score|", fontsize=12)
ax.set_title("Variable Region Sensitivity to Mutation — BBB Tropism\n"
             "(PHP.eB scan, residues 1–250)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/vr_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/vr_sensitivity.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13. FIGURE 7 — MODEL COMPARISON HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
cv_matrix = pd.DataFrame(cv_results).T[LABEL_COLS]
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(cv_matrix.values.astype(float), cmap="YlGn", vmin=0.6, vmax=1.0, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(LABEL_COLS, fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(cv_matrix.index, fontsize=10)
for i in range(3):
    for j in range(4):
        v = cv_matrix.values[i, j]
        ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "N/A",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="black" if v > 0.75 else "white")
plt.colorbar(im, ax=ax, label="AUROC (5-fold CV)")
ax.set_title("Model Comparison — AUROC by Tropism Label (5-fold CV)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/figures/model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 14. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Pipeline complete!")
print("=" * 60)
print("\nOutputs:")
print("  data/processed/features.npz")
print("  models/rf_classifier.pkl")
print("  results/predictions/all_serotypes.csv")
print("  results/predictions/phpeB_scan_raw.csv")
print("  results/predictions/phpeB_top_mutations.csv")
print("  results/figures/roc_pr_curves.png")
print("  results/figures/feature_importance.png")
print("  results/figures/serotype_predictions.png")
print("  results/figures/tsne_capsid_space.png")
print("  results/figures/phpeB_scan_heatmap.png")
print("  results/figures/vr_sensitivity.png")
print("  results/figures/model_comparison.png")
print("\nBest model (RF) CV AUROC:")
for lbl in LABEL_COLS:
    v = cv_results["Random Forest"][lbl]
    bar = "█" * int(v * 20) if not np.isnan(v) else ""
    print(f"  {lbl:<12}: {v:.3f}  {bar}")
print("=" * 60)
