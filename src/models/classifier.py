"""
classifier.py
-------------
Multi-label tropism classifier trained on ESM-2 capsid embeddings.
Predicts 4 labels: cns, peripheral, broad, bbb_crossing

Architecture: MLP with 3 hidden layers, dropout regularization
Training: Binary cross-entropy loss, 5-fold stratified cross-validation
Benchmarks against: Logistic Regression, Random Forest, SVM (physicochemical features)

Usage:
    # Train
    python src/models/classifier.py --train \
        --embeddings data/processed/embeddings.npz \
        --labels data/raw/labels.csv \
        --output models/

    # Predict
    python src/models/classifier.py --predict \
        --embeddings data/processed/my_capsid_embeddings.npz \
        --model models/mlp_classifier.pkl \
        --output results/predictions/my_capsid.csv
"""

import argparse
import csv
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Only sklearn baselines will run.")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.pipeline import Pipeline

LABEL_COLS = ["cns", "peripheral", "broad", "bbb"]


# ─── MLP Model ────────────────────────────────────────────────────────────────

class TropismMLP(nn.Module):
    """
    Multi-label MLP classifier for AAV tropism prediction.

    Input: ESM-2 mean-pooled embedding (1280-dim)
    Output: 4 sigmoid probabilities [cns, peripheral, broad, bbb]
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dims: List[int] = [512, 256, 128],
        n_labels: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 1280,
    hidden_dims: List[int] = [512, 256, 128],
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 16,
    weight_decay: float = 1e-4,
) -> Tuple["TropismMLP", "StandardScaler"]:
    """Train MLP with BCE loss."""
    # Normalize embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TropismMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_labels=y_train.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{epochs} — Loss: {epoch_loss/len(loader):.4f}")

    return model, scaler


def predict_mlp(
    model: "TropismMLP",
    scaler: "StandardScaler",
    X: np.ndarray,
) -> np.ndarray:
    """Return sigmoid probabilities for each label."""
    model.eval()
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_t).numpy()
    return probs


# ─── Sklearn Baselines ────────────────────────────────────────────────────────

def get_baseline_models() -> Dict:
    return {
        "LogReg": MultiOutputClassifier(
            LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        ),
        "RandomForest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        ),
        "GBM": MultiOutputClassifier(
            GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        ),
    }


# ─── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    names: List[str],
    n_folds: int = 5,
    use_mlp: bool = True,
) -> Dict:
    """
    5-fold cross-validation.
    Returns per-fold metrics for each label and each model.
    """
    # Stratify on first label (CNS) for consistency
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {label: {"auroc": [], "auprc": [], "f1": []} for label in LABEL_COLS}
    results["model"] = "MLP" if use_mlp else "baseline"

    all_results = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y[:, 0])):
        print(f"\n  Fold {fold_idx + 1}/{n_folds}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if use_mlp and TORCH_AVAILABLE:
            model, scaler = train_mlp(X_train, y_train)
            probs = predict_mlp(model, scaler, X_val)
        else:
            # Fallback to Random Forest
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            clf = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=200, random_state=42)
            )
            clf.fit(X_train_s, y_train)
            probs = np.column_stack([
                clf.estimators_[i].predict_proba(X_val_s)[:, 1]
                for i in range(y_train.shape[1])
            ])

        for i, label in enumerate(LABEL_COLS):
            y_true = y_val[:, i]
            y_prob = probs[:, i]

            if len(np.unique(y_true)) < 2:
                print(f"    [WARN] Only one class in {label} for this fold — skipping")
                continue

            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)

            # F1 with threshold = 0.5
            y_pred = (y_prob >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            results[label]["auroc"].append(auroc)
            results[label]["auprc"].append(auprc)
            results[label]["f1"].append(f1)

    # Aggregate
    print("\n=== Cross-Validation Results ===")
    print(f"{'Label':<15} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    print("-" * 42)
    for label in LABEL_COLS:
        auroc = np.mean(results[label]["auroc"]) if results[label]["auroc"] else float("nan")
        auprc = np.mean(results[label]["auprc"]) if results[label]["auprc"] else float("nan")
        f1 = np.mean(results[label]["f1"]) if results[label]["f1"] else float("nan")
        print(f"{label:<15} {auroc:>8.3f} {auprc:>8.3f} {f1:>8.3f}")

    return results


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_labels(labels_csv: str, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Load tropism labels, aligned to embedding order.
    Returns: label matrix (n_sequences, 4), ordered names
    """
    df = pd.read_csv(labels_csv).set_index("name")
    y = []
    valid_names = []

    for name in names:
        if name in df.index:
            row = df.loc[name]
            y.append([int(row[col]) for col in LABEL_COLS])
            valid_names.append(name)
        else:
            print(f"  [WARN] No label found for '{name}' — skipping")

    return np.array(y, dtype=float), valid_names


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_model(model, scaler, output_dir: str, name: str = "mlp"):
    """Save model and scaler to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if TORCH_AVAILABLE and isinstance(model, TropismMLP):
        torch.save(model.state_dict(), f"{output_dir}/{name}_weights.pt")
    with open(f"{output_dir}/{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {output_dir}/")


def load_model(output_dir: str, input_dim: int = 1280) -> Tuple:
    """Load saved model and scaler."""
    model = TropismMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(f"{output_dir}/mlp_weights.pt", map_location="cpu"))
    model.eval()
    with open(f"{output_dir}/mlp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AAV tropism classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--embeddings", default="data/processed/embeddings.npz")
    parser.add_argument("--labels", default="data/raw/labels.csv")
    parser.add_argument("--model", default="models/")
    parser.add_argument("--output", default="results/predictions/predictions.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    # Load embeddings
    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"]
    names = list(data["names"])
    print(f"Loaded {len(names)} embeddings, dim={embeddings.shape[1]}")

    if args.train:
        # Load labels
        y, valid_names = load_labels(args.labels, names)
        valid_idx = [names.index(n) for n in valid_names]
        X = embeddings[valid_idx]

        print(f"\nTraining on {len(valid_names)} labeled sequences...")
        print(f"Label distribution:")
        for i, label in enumerate(LABEL_COLS):
            pos = int(y[:, i].sum())
            print(f"  {label}: {pos}/{len(y)} positive ({100*pos/len(y):.0f}%)")

        # Cross-validation
        print(f"\nRunning {args.folds}-fold cross-validation...")
        cv_results = cross_validate(X, y, valid_names, n_folds=args.folds)

        # Train final model on all data
        print("\nTraining final model on all data...")
        if TORCH_AVAILABLE:
            final_model, scaler = train_mlp(X, y, input_dim=embeddings.shape[1])
            save_model(final_model, scaler, args.model)
        else:
            print("[WARN] PyTorch not available — saving RF baseline instead")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            final_model = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=200, random_state=42)
            )
            final_model.fit(X_scaled, y)
            with open(f"{args.model}/rf_classifier.pkl", "wb") as f:
                pickle.dump((final_model, scaler), f)

    if args.predict:
        print("\nRunning prediction...")
        model, scaler = load_model(args.model, input_dim=embeddings.shape[1])
        probs = predict_mlp(model, scaler, embeddings)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name"] + LABEL_COLS + ["top_label"])
            for name, prob_row in zip(names, probs):
                top = LABEL_COLS[np.argmax(prob_row)]
                writer.writerow([name] + [f"{p:.4f}" for p in prob_row] + [top])

        print(f"Predictions saved to {args.output}")

        # Print table
        print(f"\n{'Name':<20} {'CNS':>6} {'Periph':>6} {'Broad':>6} {'BBB':>6}")
        print("-" * 46)
        for name, prob_row in zip(names, probs):
            print(f"{name:<20} {prob_row[0]:>6.3f} {prob_row[1]:>6.3f} "
                  f"{prob_row[2]:>6.3f} {prob_row[3]:>6.3f}")


if __name__ == "__main__":
    main()
