"""
test_pipeline.py
----------------
Integration tests for the AAV tropism predictor pipeline.
Tests run without ESM-2 or trained models using mock components.

Run with:
    pytest tests/test_pipeline.py -v
    # or
    python -m pytest tests/ -v
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sequences():
    """Small set of test capsid sequences."""
    return {
        "AAV9_fragment": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNE" * 3,
        "AAV8_fragment": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNF" * 3,
        "PHP_eB_fragment": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNG" * 3,
    }


@pytest.fixture
def sample_fasta(sample_sequences, tmp_path):
    """Write test FASTA file."""
    fasta_path = tmp_path / "test_capsids.fasta"
    with open(fasta_path, "w") as f:
        for name, seq in sample_sequences.items():
            f.write(f">{name}\n{seq}\n")
    return str(fasta_path)


@pytest.fixture
def sample_labels_csv(sample_sequences, tmp_path):
    """Write test labels CSV."""
    csv_path = tmp_path / "labels.csv"
    data = {
        "name": list(sample_sequences.keys()),
        "cns":        [1, 0, 1],
        "peripheral": [1, 1, 0],
        "broad":      [1, 1, 1],
        "bbb":        [1, 0, 1],
        "doi":        ["10.test/1", "10.test/2", "10.test/3"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_embedder():
    from src.models.mutational_scan import make_mock_embedder
    return make_mock_embedder(embed_dim=1280)


@pytest.fixture
def mock_classifier():
    from src.models.mutational_scan import make_mock_classifier
    return make_mock_classifier(n_labels=4)


@pytest.fixture
def sample_embeddings(sample_sequences, tmp_path):
    """Create fake embedding NPZ file."""
    names = list(sample_sequences.keys())
    embeddings = np.random.randn(len(names), 1280).astype(np.float32)
    path = tmp_path / "embeddings.npz"
    np.savez_compressed(str(path), embeddings=embeddings, names=np.array(names))
    return str(path)


# ─── Tests: FASTA Parsing ─────────────────────────────────────────────────────

def test_parse_fasta(sample_fasta, sample_sequences):
    from predict import parse_fasta
    result = parse_fasta(sample_fasta)
    assert set(result.keys()) == set(sample_sequences.keys())
    for name, seq in sample_sequences.items():
        assert result[name] == seq


def test_parse_fasta_empty_file(tmp_path):
    from predict import parse_fasta
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    result = parse_fasta(str(empty))
    assert result == {}


# ─── Tests: Mock Embedder ─────────────────────────────────────────────────────

def test_mock_embedder_shape(mock_embedder):
    seq = "MAADHYLPDWLED" * 5
    emb = mock_embedder(seq)
    assert emb.shape == (1280,)
    assert not np.any(np.isnan(emb))


def test_mock_embedder_deterministic(mock_embedder):
    seq = "ACDEFGHIKLMNPQRSTVWY" * 4
    emb1 = mock_embedder(seq)
    emb2 = mock_embedder(seq)
    np.testing.assert_array_almost_equal(emb1, emb2)


def test_mock_embedder_different_sequences(mock_embedder):
    emb1 = mock_embedder("AAAAAAAAAA" * 5)
    emb2 = mock_embedder("KKKKKKKKKK" * 5)
    # Different sequences should produce different embeddings
    assert not np.allclose(emb1, emb2)


# ─── Tests: Mock Classifier ───────────────────────────────────────────────────

def test_mock_classifier_shape(mock_classifier):
    X = np.random.randn(5, 1280).astype(np.float32)
    probs = mock_classifier(X)
    assert probs.shape == (5, 4)


def test_mock_classifier_probabilities_in_range(mock_classifier):
    X = np.random.randn(10, 1280).astype(np.float32)
    probs = mock_classifier(X)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


# ─── Tests: Physicochemical Features ─────────────────────────────────────────

def test_physicochemical_features(sample_sequences):
    from src.features.esm_embeddings import extract_embeddings_physicochemical
    embeddings, names = extract_embeddings_physicochemical(sample_sequences)
    assert embeddings.shape[0] == len(sample_sequences)
    assert embeddings.shape[1] == 41
    assert not np.any(np.isnan(embeddings))


# ─── Tests: Label Loading ─────────────────────────────────────────────────────

def test_load_labels(sample_labels_csv, sample_sequences):
    from src.models.classifier import load_labels, LABEL_COLS
    names = list(sample_sequences.keys())
    y, valid_names = load_labels(sample_labels_csv, names)
    assert len(valid_names) == len(names)
    assert y.shape == (len(names), 4)
    assert set(valid_names) == set(names)
    assert np.all((y == 0) | (y == 1))


def test_load_labels_missing_sequence(sample_labels_csv):
    from src.models.classifier import load_labels
    names = ["AAV9_fragment", "NONEXISTENT_CAPSID"]
    y, valid_names = load_labels(sample_labels_csv, names)
    assert "NONEXISTENT_CAPSID" not in valid_names
    assert "AAV9_fragment" in valid_names


# ─── Tests: Mutational Scan ───────────────────────────────────────────────────

def test_mutational_scan_basic(mock_embedder, mock_classifier):
    from src.models.mutational_scan import run_single_scan

    seq = "ACDEFGHIKLMNPQRSTVWYACDEFG"  # 25 AA
    df = run_single_scan(seq, mock_embedder, mock_classifier,
                         label_idx=3, max_positions=5, verbose=False)

    # 5 positions × 19 mutations (excluding WT)
    assert len(df) == 5 * 19
    assert "delta_score" in df.columns
    assert "position" in df.columns
    assert "vr_region" in df.columns


def test_mutational_scan_delta_wt_is_zero(mock_embedder, mock_classifier):
    """Wild-type mutations should not appear in output."""
    from src.models.mutational_scan import run_single_scan
    seq = "ACDEFGHIKLMN"
    df = run_single_scan(seq, mock_embedder, mock_classifier,
                         label_idx=0, max_positions=3, verbose=False)
    # Check no WT→WT rows
    for _, row in df.iterrows():
        assert row["wt_aa"] != row["mutant_aa"]


def test_summarize_scan(mock_embedder, mock_classifier):
    from src.models.mutational_scan import run_single_scan, summarize_scan
    seq = "ACDEFGHIKLMNPQRS"
    df = run_single_scan(seq, mock_embedder, mock_classifier,
                         label_idx=3, max_positions=5, verbose=False)
    summary = summarize_scan(df)
    assert len(summary) == 5  # One row per position
    assert "sensitivity" in summary.columns
    assert "max_gain" in summary.columns


def test_vr_assignment():
    from src.models.mutational_scan import assign_vr, VP1_VARIABLE_REGIONS
    # Position within VR-IV should be assigned VR-IV
    vr4_start = VP1_VARIABLE_REGIONS["VR-IV"][0]
    assert assign_vr(vr4_start) == "VR-IV"
    # Position outside all VRs
    assert assign_vr(10) == "other"


# ─── Tests: Prediction Pipeline ───────────────────────────────────────────────

def test_run_prediction(sample_sequences, mock_embedder, mock_classifier, tmp_path):
    from predict import run_prediction
    LABEL_COLS = ["cns", "peripheral", "broad", "bbb"]
    output_dir = str(tmp_path / "predictions")
    df = run_prediction(sample_sequences, mock_embedder, mock_classifier,
                        LABEL_COLS, output_dir)
    assert len(df) == len(sample_sequences)
    for col in LABEL_COLS:
        assert col in df.columns
        assert df[col].between(0, 1).all()
    assert os.path.exists(os.path.join(output_dir, "predictions.csv"))


# ─── Tests: Visualization (just check no errors) ──────────────────────────────

def test_plot_vr_sensitivity(tmp_path):
    from src.visualization.plots import plot_vr_sensitivity
    vr_df = pd.DataFrame({
        "vr_region": ["VR-I", "VR-IV", "VR-VIII", "other"],
        "mean_abs_delta": [0.05, 0.12, 0.09, 0.03],
        "max_abs_delta": [0.15, 0.25, 0.18, 0.08],
        "n_variants": [100, 200, 150, 500],
    })
    output_path = str(tmp_path / "vr_test.png")
    plot_vr_sensitivity(vr_df, output_path)
    assert os.path.exists(output_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
