#!/usr/bin/env python3
"""
predict.py
----------
CLI entry point for the AAV capsid tropism predictor.

Usage examples:
    # Predict tropism for one or more capsids
    python predict.py --fasta my_capsid.fasta --model models/ --output results/predictions/

    # Run full mutational scan for BBB engineering
    python predict.py --fasta my_capsid.fasta --scan --label bbb --output results/scan/

    # Use mock mode (no ESM-2 required, for testing)
    python predict.py --fasta data/raw/capsid_sequences.fasta --mock --output results/test/
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path


def parse_fasta(fasta_path: str) -> dict:
    sequences = {}
    current_name = None
    current_seq = []
    with open(fasta_path) as f:
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
    return sequences


def print_banner():
    print("=" * 60)
    print("  AAV Capsid CNS Tropism Predictor")
    print("  ESM-2 + Multi-label MLP Classifier")
    print("=" * 60)
    print()


def run_prediction(sequences: dict, embedder_fn, classifier_fn, label_cols, output_dir: str):
    """Embed all sequences and predict tropism scores."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Predicting tropism for {len(sequences)} sequences...")
    for name, seq in sequences.items():
        print(f"  [{name}] {len(seq)} aa...")
        emb = embedder_fn(seq)
        probs = classifier_fn(emb[np.newaxis, :])[0]
        top_label = label_cols[int(np.argmax(probs))]
        results.append({
            "name": name,
            **{label: float(probs[i]) for i, label in enumerate(label_cols)},
            "top_label": top_label,
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(out_path, index=False)

    print(f"\nResults saved → {out_path}")
    print("\n" + "=" * 60)
    print(f"{'Name':<20} {'CNS':>6} {'Periph':>7} {'Broad':>6} {'BBB':>6}  {'Top'}") 
    print("-" * 60)
    for _, row in df.iterrows():
        print(
            f"{row['name']:<20} {row['cns']:>6.3f} {row['peripheral']:>7.3f} "
            f"{row['broad']:>6.3f} {row['bbb']:>6.3f}  {row['top_label']}"
        )
    print("=" * 60)
    return df


def run_scan(sequences: dict, query_name: str, embedder_fn, classifier_fn,
             label: str, output_dir: str, max_positions: int = None):
    """Run mutational scan on a single sequence."""
    from src.models.mutational_scan import (
        run_single_scan, summarize_scan, get_top_beneficial_mutations,
        vr_region_sensitivity, LABEL_MAP
    )

    label_idx = LABEL_MAP[label]
    sequence = sequences[query_name]

    print(f"\nRunning mutational scan...")
    print(f"  Query: {query_name} ({len(sequence)} aa)")
    print(f"  Target label: {label.upper()}")
    if max_positions:
        print(f"  Scanning first {max_positions} positions only")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = run_single_scan(
        sequence, embedder_fn, classifier_fn,
        label_idx=label_idx,
        max_positions=max_positions,
    )

    raw_path = os.path.join(output_dir, f"{query_name}_scan_raw.csv")
    df.to_csv(raw_path, index=False)

    summary = summarize_scan(df)
    summary_path = os.path.join(output_dir, f"{query_name}_position_summary.csv")
    summary.to_csv(summary_path, index=False)

    vr_sens = vr_region_sensitivity(df)
    vr_path = os.path.join(output_dir, f"{query_name}_vr_sensitivity.csv")
    vr_sens.to_csv(vr_path, index=False)

    # Plots
    try:
        from src.visualization.plots import plot_mutational_scan_heatmap, plot_vr_sensitivity
        heatmap_path = os.path.join(output_dir, f"{query_name}_heatmap.png")
        plot_mutational_scan_heatmap(df, sequence, heatmap_path, label=label,
                                      title=f"{query_name} — Mutational Scan ({label.upper()})")
        vr_fig_path = os.path.join(output_dir, f"{query_name}_vr_sensitivity.png")
        plot_vr_sensitivity(vr_sens, vr_fig_path, label=label)
    except Exception as e:
        print(f"[WARN] Could not generate plots: {e}")

    print(f"\nOutputs saved to {output_dir}/")
    print(f"\n=== Top 10 Mutations Increasing {label.upper()} Score ===")
    top = get_top_beneficial_mutations(df, top_n=10)
    print(top.to_string(index=False))

    print(f"\n=== Variable Region Sensitivity ===")
    print(vr_sens.to_string(index=False))


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Predict AAV capsid CNS tropism from sequence using ESM-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--model", default="models/", help="Trained model directory")
    parser.add_argument("--scan", action="store_true", help="Run mutational scan")
    parser.add_argument("--sequence", default=None, help="Sequence name for scan (default: first)")
    parser.add_argument("--label", default="bbb",
                        choices=["cns", "peripheral", "broad", "bbb"],
                        help="Target label for mutational scan")
    parser.add_argument("--max-positions", type=int, default=None,
                        help="Limit scan to first N positions (faster testing)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock embedder/classifier (no ESM-2 install needed)")
    args = parser.parse_args()

    LABEL_COLS = ["cns", "peripheral", "broad", "bbb"]

    # Load sequences
    if not os.path.exists(args.fasta):
        print(f"[ERROR] FASTA file not found: {args.fasta}")
        sys.exit(1)

    sequences = parse_fasta(args.fasta)
    if not sequences:
        print(f"[ERROR] No sequences found in {args.fasta}")
        sys.exit(1)

    print(f"Loaded {len(sequences)} sequence(s): {', '.join(list(sequences.keys())[:5])}")

    # Set up embedder and classifier
    if args.mock:
        print("\n[MOCK MODE] Using deterministic mock embedder (for testing pipeline)")
        print("  To use real ESM-2, install: pip install fair-esm torch")
        print("  Then remove the --mock flag\n")
        from src.models.mutational_scan import make_mock_embedder, make_mock_classifier
        embedder_fn = make_mock_embedder()
        classifier_fn = make_mock_classifier()

    else:
        try:
            import torch
            import esm
            from src.models.classifier import load_model, predict_mlp

            print("Loading ESM-2 (esm2_t33_650M_UR50D)...")
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            esm_model.eval()
            batch_converter = alphabet.get_batch_converter()

            def embedder_fn(seq: str) -> np.ndarray:
                data = [("q", seq[:1022])]
                _, _, tokens = batch_converter(data)
                with torch.no_grad():
                    results = esm_model(tokens, repr_layers=[33])
                rep = results["representations"][33][0, 1:min(len(seq), 1022) + 1]
                return rep.mean(dim=0).numpy()

            print("Loading trained classifier...")
            clf_model, scaler = load_model(args.model, input_dim=1280)

            def classifier_fn(X: np.ndarray) -> np.ndarray:
                return predict_mlp(clf_model, scaler, X)

        except FileNotFoundError:
            print("[WARN] Trained model not found. Run training first:")
            print("  python src/models/classifier.py --train \\")
            print("    --embeddings data/processed/embeddings.npz \\")
            print("    --labels data/raw/labels.csv")
            print("\nFalling back to mock mode for demonstration...")
            from src.models.mutational_scan import make_mock_embedder, make_mock_classifier
            embedder_fn = make_mock_embedder()
            classifier_fn = make_mock_classifier()

        except ImportError as e:
            print(f"[WARN] Missing dependency: {e}")
            print("Install with: pip install fair-esm torch")
            print("Falling back to mock mode...\n")
            from src.models.mutational_scan import make_mock_embedder, make_mock_classifier
            embedder_fn = make_mock_embedder()
            classifier_fn = make_mock_classifier()

    # Run prediction or scan
    if args.scan:
        query_name = args.sequence or list(sequences.keys())[0]
        if query_name not in sequences:
            print(f"[ERROR] Sequence '{query_name}' not found in FASTA")
            sys.exit(1)
        run_scan(
            sequences, query_name, embedder_fn, classifier_fn,
            label=args.label,
            output_dir=args.output,
            max_positions=args.max_positions,
        )
    else:
        run_prediction(sequences, embedder_fn, classifier_fn, LABEL_COLS, args.output)


if __name__ == "__main__":
    main()
