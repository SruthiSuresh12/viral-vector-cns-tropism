"""
esm_embeddings.py
-----------------
Extracts per-sequence embeddings from Meta's ESM-2 protein language model.
Uses mean-pooling across residue positions → 1280-dimensional vector per capsid.

Model: esm2_t33_650M_UR50D (650M parameters)
  - 33 transformer layers
  - Trained on 250M UniRef50 sequences
  - No GPU required for capsid-scale datasets (~100 sequences)

Usage:
    python src/features/esm_embeddings.py \
        --fasta data/raw/capsid_sequences.fasta \
        --output data/processed/embeddings.npz
"""

import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import esm
    import torch
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("[WARN] fair-esm not installed. Install with: pip install fair-esm")


def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse FASTA file → {name: sequence}."""
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


def extract_embeddings_esm2(
    sequences: Dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 4,
    repr_layer: int = 33,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract mean-pooled residue embeddings from ESM-2.

    Returns:
        embeddings: np.ndarray of shape (n_sequences, 1280)
        names: list of sequence names in same order
    """
    if not ESM_AVAILABLE:
        raise ImportError("Install fair-esm: pip install fair-esm")

    print(f"Loading ESM-2 model: {model_name}")
    model, alphabet = esm.pretrained.__dict__[model_name]()
    model.eval()

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Running on: {device}")

    batch_converter = alphabet.get_batch_converter()
    names = list(sequences.keys())
    seqs = list(sequences.values())

    all_embeddings = []

    # Process in batches to manage memory
    for batch_start in range(0, len(names), batch_size):
        batch_names = names[batch_start:batch_start + batch_size]
        batch_seqs = seqs[batch_start:batch_start + batch_size]

        print(f"  Processing batch {batch_start // batch_size + 1} "
              f"({batch_start + 1}–{min(batch_start + batch_size, len(names))} "
              f"of {len(names)})...")

        # Truncate sequences >1022 residues (ESM-2 context limit)
        batch_data = [(n, s[:1022]) for n, s in zip(batch_names, batch_seqs)]

        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)

        # representations shape: (batch, seq_len, embed_dim)
        representations = results["representations"][repr_layer]

        # Mean-pool over sequence positions (excluding BOS/EOS tokens)
        for i, (name, seq) in enumerate(zip(batch_names, batch_seqs)):
            seq_len = min(len(seq), 1022)
            # Tokens: [BOS, residue_1, ..., residue_n, EOS] → index 1 to seq_len
            mean_repr = representations[i, 1:seq_len + 1].mean(dim=0).cpu().numpy()
            all_embeddings.append(mean_repr)

        time.sleep(0.1)  # Small pause between batches

    embeddings = np.vstack(all_embeddings)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    return embeddings, names


def extract_embeddings_physicochemical(sequences: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    """
    Fallback: extract traditional physicochemical features.
    Used for baseline comparison vs ESM-2.

    Features per sequence (41 total):
    - Amino acid composition (20)
    - Dipeptide composition summary (5)
    - Physical properties: MW, pI, GRAVY, aromaticity, instability (5)
    - Secondary structure fractions (3)
    - VP1-specific: VR-I through VR-IX mean hydrophobicity (9) [approximate windows]
    """
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        raise ImportError("Install biopython: pip install biopython")

    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

    # Kyte-Doolittle hydrophobicity scale
    KD = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }

    names = list(sequences.keys())
    feature_matrix = []

    for name in names:
        seq = sequences[name]
        # Remove non-standard AAs
        clean_seq = "".join(c for c in seq.upper() if c in AA_ORDER)
        if len(clean_seq) < 10:
            print(f"  [WARN] Short sequence for {name}: {len(clean_seq)} AAs")
            feature_matrix.append(np.zeros(41))
            continue

        analysis = ProteinAnalysis(clean_seq)

        # 1. AA composition (20 features)
        aa_comp = analysis.get_amino_acids_percent()
        aa_feats = [aa_comp.get(aa, 0.0) for aa in AA_ORDER]

        # 2. Global physicochemical (5 features)
        mw = analysis.molecular_weight() / 100_000  # Normalize
        gravy = analysis.gravy()
        aromaticity = analysis.aromaticity()
        instability = analysis.instability_index() / 100
        try:
            pi = analysis.isoelectric_point() / 14  # Normalize to [0,1]
        except Exception:
            pi = 0.5

        # 3. Secondary structure fractions (3 features)
        helix, turn, sheet = analysis.secondary_structure_fraction()

        # 4. VR region hydrophobicity (approximate windows for VP1 ~ 730 aa)
        # Canonical VR positions from Govindasamy et al., 2006
        vr_windows = [
            (253, 273), (326, 332), (381, 395), (448, 472),
            (491, 506), (531, 555), (575, 610), (656, 669), (706, 724)
        ]
        vr_hydrophob = []
        for start, end in vr_windows:
            if end <= len(clean_seq):
                segment = clean_seq[start:end]
                h = np.mean([KD.get(aa, 0.0) for aa in segment])
            else:
                h = 0.0
            vr_hydrophob.append(h)

        features = aa_feats + [mw, gravy, aromaticity, instability, pi, helix, turn, sheet] + vr_hydrophob
        feature_matrix.append(features)

    return np.array(feature_matrix), names


def save_embeddings(embeddings: np.ndarray, names: List[str], output_path: str):
    """Save embeddings and metadata to .npz file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        names=np.array(names, dtype=str)
    )
    print(f"Saved embeddings → {output_path}")
    print(f"  Shape: {embeddings.shape}")


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load saved embeddings."""
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], list(data["names"])


def main(fasta_path: str, output_path: str, method: str, batch_size: int):
    print(f"Loading sequences from {fasta_path}...")
    sequences = parse_fasta(fasta_path)
    print(f"  Found {len(sequences)} sequences")

    if method == "esm2":
        print("\nExtracting ESM-2 embeddings (this takes ~10 min on CPU)...")
        embeddings, names = extract_embeddings_esm2(
            sequences, batch_size=batch_size
        )
    elif method == "physicochemical":
        print("\nExtracting physicochemical features (baseline)...")
        embeddings, names = extract_embeddings_physicochemical(sequences)
    else:
        raise ValueError(f"Unknown method: {method}")

    save_embeddings(embeddings, names, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/raw/capsid_sequences.fasta")
    parser.add_argument("--output", default="data/processed/embeddings.npz")
    parser.add_argument("--method", choices=["esm2", "physicochemical"], default="esm2")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    main(args.fasta, args.output, args.method, args.batch_size)
