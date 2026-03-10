"""
mutational_scan.py
------------------
In silico saturation mutagenesis of AAV capsid sequences.
Systematically substitutes each residue position with all 19 alternatives,
re-embeds with ESM-2, and scores with the trained classifier.

Delta-score = mutant_score - wildtype_score

High-magnitude positions are candidates for rational BBB-engineering experiments.
These results are cross-referenced with known VP1 variable regions (VR-I to VR-IX).

Usage:
    python src/models/mutational_scan.py \
        --fasta data/raw/my_capsid.fasta \
        --sequence PHP.eB \
        --model models/ \
        --label bbb \
        --output results/predictions/phpeB_scan.csv
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Standard amino acids
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Known VP1 variable region windows (residue positions, 0-indexed)
# Based on Govindasamy et al., 2006 (AAV1 numbering)
VP1_VARIABLE_REGIONS = {
    "VR-I":   (252, 272),
    "VR-II":  (325, 331),
    "VR-III": (380, 394),
    "VR-IV":  (447, 471),
    "VR-V":   (490, 505),
    "VR-VI":  (530, 554),
    "VR-VII": (574, 609),
    "VR-VIII":(655, 668),
    "VR-IX":  (705, 723),
}


def assign_vr(position: int) -> str:
    """Return variable region name for a given residue position, or 'other'."""
    for vr_name, (start, end) in VP1_VARIABLE_REGIONS.items():
        if start <= position <= end:
            return vr_name
    return "other"


def run_single_scan(
    sequence: str,
    embedder_fn,
    classifier_fn,
    label_idx: int = 3,  # default: bbb
    max_positions: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan all positions in `sequence`.

    Args:
        sequence: wild-type amino acid sequence
        embedder_fn: callable(seq: str) → np.ndarray(1280,)
        classifier_fn: callable(embedding: np.ndarray) → np.ndarray(4,) probabilities
        label_idx: which label to track (0=cns, 1=peripheral, 2=broad, 3=bbb)
        max_positions: limit to first N positions (for testing)

    Returns:
        DataFrame with columns: position, wt_aa, mutant_aa, delta_score, vr_region
    """
    seq_len = len(sequence)
    if max_positions:
        seq_len = min(seq_len, max_positions)

    # Wild-type score
    wt_embedding = embedder_fn(sequence)
    wt_probs = classifier_fn(wt_embedding[np.newaxis, :])[0]
    wt_score = wt_probs[label_idx]

    if verbose:
        print(f"Wild-type score (label {label_idx}): {wt_score:.4f}")
        print(f"Scanning {seq_len} positions × 19 mutations = {seq_len * 19} variants...")

    records = []
    for pos in range(seq_len):
        wt_aa = sequence[pos]

        for mut_aa in AAS:
            if mut_aa == wt_aa:
                continue  # Skip synonymous

            # Construct mutant sequence
            mutant_seq = sequence[:pos] + mut_aa + sequence[pos + 1:]

            # Embed and score
            mut_embedding = embedder_fn(mutant_seq)
            mut_probs = classifier_fn(mut_embedding[np.newaxis, :])[0]
            mut_score = mut_probs[label_idx]

            delta = float(mut_score) - float(wt_score)

            records.append({
                "position": pos + 1,  # 1-indexed for biology
                "wt_aa": wt_aa,
                "mutant_aa": mut_aa,
                "wt_score": float(wt_score),
                "mutant_score": float(mut_score),
                "delta_score": delta,
                "abs_delta": abs(delta),
                "vr_region": assign_vr(pos),
            })

        if verbose and (pos + 1) % 50 == 0:
            print(f"  Position {pos + 1}/{seq_len} done...")

    df = pd.DataFrame(records)
    return df


def summarize_scan(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Summarize mutational scan results.

    Returns per-position summary with:
    - mean delta across all 19 mutations
    - max gain mutation (most beneficial for label)
    - max loss mutation (most damaging for label)
    - position sensitivity (std of deltas)
    """
    summary = (
        df.groupby(["position", "wt_aa", "vr_region"])
        .agg(
            mean_delta=("delta_score", "mean"),
            std_delta=("delta_score", "std"),
            max_gain=("delta_score", "max"),
            max_loss=("delta_score", "min"),
        )
        .reset_index()
    )
    summary["sensitivity"] = summary["std_delta"]
    summary = summary.sort_values("sensitivity", ascending=False)
    return summary


def get_top_beneficial_mutations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top N single mutations predicted to most increase the target label."""
    return (
        df.sort_values("delta_score", ascending=False)
        .head(top_n)[["position", "wt_aa", "mutant_aa", "delta_score", "vr_region"]]
        .reset_index(drop=True)
    )


def get_top_damaging_mutations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return top N single mutations predicted to most decrease the target label."""
    return (
        df.sort_values("delta_score", ascending=True)
        .head(top_n)[["position", "wt_aa", "mutant_aa", "delta_score", "vr_region"]]
        .reset_index(drop=True)
    )


def vr_region_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean sensitivity by variable region."""
    return (
        df.groupby("vr_region")
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            max_abs_delta=("abs_delta", "max"),
            n_variants=("delta_score", "count"),
        )
        .sort_values("mean_abs_delta", ascending=False)
        .reset_index()
    )


# ─── Mock embedder/classifier for testing without GPU ────────────────────────

def make_mock_embedder(embed_dim: int = 1280):
    """
    Deterministic mock embedder for testing pipeline without ESM-2.
    Uses sequence composition as a proxy for the embedding.
    """
    AAS_IDX = {aa: i for i, aa in enumerate(AAS)}

    def mock_embedder(seq: str) -> np.ndarray:
        # Composition-based embedding (not a real LM, for testing only)
        comp = np.zeros(20)
        for aa in seq:
            if aa in AAS_IDX:
                comp[AAS_IDX[aa]] += 1
        comp /= len(seq) + 1e-8
        # Expand to embed_dim with positional encoding proxy
        np.random.seed(hash(seq[:50]) % 2**31)
        noise = np.random.normal(0, 0.01, embed_dim)
        embedding = np.tile(comp, embed_dim // 20 + 1)[:embed_dim]
        return embedding + noise

    return mock_embedder


def make_mock_classifier(n_labels: int = 4):
    """Mock classifier for testing. Uses embedding norm as proxy."""
    def mock_classifier(X: np.ndarray) -> np.ndarray:
        # Simple linear probe on first few dimensions
        weights = np.array([
            [0.1, -0.1,  0.05, -0.05],  # label 0: cns
            [-0.1, 0.1, -0.05,  0.05],  # label 1: peripheral
            [0.05, 0.05, 0.1,  0.05],   # label 2: broad
            [0.15, -0.15, 0.02, 0.1],   # label 3: bbb
        ]).T
        logits = X[:, :4] @ weights
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        return np.clip(probs, 0.05, 0.95)
    return mock_classifier


# ─── CLI ──────────────────────────────────────────────────────────────────────

LABEL_MAP = {"cns": 0, "peripheral": 1, "broad": 2, "bbb": 3}


def main():
    parser = argparse.ArgumentParser(description="Mutational scan for AAV tropism")
    parser.add_argument("--fasta", required=True, help="FASTA file with query sequence")
    parser.add_argument("--sequence", default=None, help="Sequence name in FASTA (default: first)")
    parser.add_argument("--model", default="models/", help="Model directory")
    parser.add_argument("--label", default="bbb", choices=list(LABEL_MAP.keys()))
    parser.add_argument("--output", default="results/predictions/scan_results.csv")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=None, help="Limit scan length (testing)")
    parser.add_argument("--mock", action="store_true", help="Use mock embedder (no ESM-2 needed)")
    args = parser.parse_args()

    # Load sequence
    sequences = {}
    current_name = None
    current_seq = []
    with open(args.fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name:
        sequences[current_name] = "".join(current_seq)

    query_name = args.sequence or list(sequences.keys())[0]
    sequence = sequences[query_name]
    print(f"Query: {query_name} ({len(sequence)} aa)")
    print(f"Label: {args.label}")

    # Set up embedder and classifier
    if args.mock:
        print("\n[MOCK MODE] Using deterministic mock embedder (for testing)")
        embedder_fn = make_mock_embedder()
        classifier_fn = make_mock_classifier()
    else:
        try:
            import torch
            from src.features.esm_embeddings import extract_embeddings_esm2
            from src.models.classifier import load_model, predict_mlp

            print("\nLoading ESM-2 and trained classifier...")
            model, alphabet = None, None  # lazy load inside scan

            # Build callable embedder
            esm_model, esm_alphabet = __import__("esm").pretrained.esm2_t33_650M_UR50D()
            esm_model.eval()
            batch_converter = esm_alphabet.get_batch_converter()

            def esm_embedder(seq: str) -> np.ndarray:
                data = [("query", seq[:1022])]
                _, _, tokens = batch_converter(data)
                with torch.no_grad():
                    results = esm_model(tokens, repr_layers=[33])
                rep = results["representations"][33][0, 1:min(len(seq), 1022) + 1]
                return rep.mean(dim=0).numpy()

            clf_model, scaler = load_model(args.model)

            def classifier_fn(X: np.ndarray) -> np.ndarray:
                return predict_mlp(clf_model, scaler, X)

            embedder_fn = esm_embedder

        except Exception as e:
            print(f"[ERROR] Could not load ESM-2 or classifier: {e}")
            print("[INFO] Falling back to mock mode. Use --mock flag to suppress this.")
            embedder_fn = make_mock_embedder()
            classifier_fn = make_mock_classifier()

    label_idx = LABEL_MAP[args.label]

    # Run scan
    print("\nRunning mutational scan...")
    t0 = time.time()
    df = run_single_scan(
        sequence,
        embedder_fn,
        classifier_fn,
        label_idx=label_idx,
        max_positions=args.max_positions,
    )
    elapsed = time.time() - t0
    print(f"Scan complete in {elapsed:.1f}s ({len(df)} variants tested)")

    # Save raw results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nRaw scan results → {args.output}")

    # Summaries
    summary_path = args.output.replace(".csv", "_position_summary.csv")
    summary = summarize_scan(df)
    summary.to_csv(summary_path, index=False)
    print(f"Position summary → {summary_path}")

    vr_path = args.output.replace(".csv", "_vr_sensitivity.csv")
    vr_sens = vr_region_sensitivity(df)
    vr_sens.to_csv(vr_path, index=False)
    print(f"VR region sensitivity → {vr_path}")

    # Print top beneficial mutations
    print(f"\n=== Top {args.top_n} beneficial mutations for {args.label.upper()} ===")
    top_gain = get_top_beneficial_mutations(df, top_n=args.top_n)
    print(top_gain.to_string(index=False))

    print(f"\n=== Variable Region Sensitivity ===")
    print(vr_sens.to_string(index=False))

    print(f"\n=== Most Sensitive Positions (top 10) ===")
    print(summary.head(10)[["position", "wt_aa", "vr_region", "sensitivity", "max_gain", "max_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
