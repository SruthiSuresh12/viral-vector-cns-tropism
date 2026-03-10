# Viral Vector CNS Tropism Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ESM-2](https://img.shields.io/badge/Model-ESM--2-green.svg)](https://github.com/facebookresearch/esm)

**Protein language model-based prediction of CNS tropism and blood-brain barrier penetration from AAV capsid sequences.**

---

## Motivation

Adeno-associated virus (AAV) vectors are the leading platform for CNS gene therapy, but capsid selection remains a major bottleneck. Of the >100 natural serotypes and engineered variants characterized to date, only a handful exhibit robust CNS tropism and blood-brain barrier (BBB) crossing after systemic administration — and the sequence determinants of this selectivity are poorly understood.

This tool addresses the *capsid selection problem*: given a novel or modified AAV capsid sequence, predict its likely tropism profile (CNS, peripheral, or broad) and BBB-crossing capability *before* committing to expensive in vivo validation. The approach combines:

1. **ESM-2 sequence embeddings**  — state-of-the-art protein language model capturing deep evolutionary and structural context from sequence alone
2. **Multi-label tropism classifier** trained on curated serotype–phenotype data from the literature
3. **Mutational scanning / in silico saturation mutagenesis** to produce residue-level functional maps of capsid surface
4. **Interpretability via gradient attribution** (captum) to identify which sequence positions drive tropism predictions

This computational approach is designed to complement experimental capsid engineering efforts such as the SHREAD platform and similar BBB-retargeting strategies being developed in neural engineering labs.

---

## Key Features

- **Predict tropism** from any VP1 capsid amino acid sequence (FASTA input)
- **4-class multi-label classification:** CNS tropic | Peripheral tropic | Broad tropic | BBB-crossing
- **Mutational scanning map:** per-position delta-score visualization highlighting BBB-critical residues
- **Batch mode:** score entire capsid libraries at once
- **Interactive scoring notebook** with annotated examples for all major serotypes

---

## Repository Structure

```
viral-vector-cns-tropism/
├── data/
│   ├── raw/                     # Raw FASTA sequences from UniProt/NCBI
│   └── processed/               # Encoded feature matrices, train/test splits
├── src/
│   ├── data/
│   │   ├── fetch_sequences.py   # Download AAV capsid sequences from UniProt
│   │   └── preprocess.py        # FASTA parsing, alignment, label assignment
│   ├── features/
│   │   └── esm_embeddings.py    # ESM-2 embedding extraction (CPU-friendly)
│   ├── models/
│   │   ├── classifier.py        # Multi-label tropism classifier
│   │   └── mutational_scan.py   # In silico saturation mutagenesis
│   └── visualization/
│       └── plots.py             # Capsid surface heatmaps, ROC curves, UMAP
├── models/                      # Saved model weights (.pkl / .pt)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_esm2_embeddings.ipynb
│   ├── 03_classifier_training.ipynb
│   ├── 04_mutational_scanning.ipynb
│   └── 05_predict_novel_capsid.ipynb
├── results/
│   ├── figures/                
│   └── predictions/             # Per-serotype scores, scanning maps
├── tests/
│   └── test_pipeline.py
├── environment.yml
├── requirements.txt
└── predict.py                   # CLI entry point
```

---

## Installation

```bash
git clone https://github.com/yourusername/viral-vector-cns-tropism.git
cd viral-vector-cns-tropism

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate aav-tropism

# Or pip
pip install -r requirements.txt
```

> **Note on GPU:** ESM-2 inference runs on CPU for the capsid dataset size used here (~100–200 sequences). No GPU required. On a modern laptop, embedding extraction takes ~10 minutes for the full dataset.

---

## Quick Start

### Predict tropism for a single sequence

```bash
python predict.py --fasta my_capsid.fasta --output results/my_prediction.csv
```

### Run mutational scan on a sequence of interest

```bash
python predict.py --fasta my_capsid.fasta --scan --output results/scan_map.csv
```

### Interactive notebook

Open `notebooks/05_predict_novel_capsid.ipynb` and follow the annotated walkthrough.

---

## Dataset

Capsid sequences and tropism labels were curated from:

| Source | Sequences | Notes |
|--------|-----------|-------|
| UniProt (reviewed) | 47 | AAV1–13 + natural isolates |
| Published engineering papers | 38 | PHP.eB, PHP.B, AAV-BR1, AAV-F, SCAAVs |
| NCBI GenBank | 29 | Primate isolates with published tropism data |

**Tropism labels** (multi-label) assigned from peer-reviewed in vivo studies:
- `cns`: preferential transduction of brain/spinal cord tissue
- `peripheral`: liver, muscle, heart, lung
- `broad`: efficient transduction across ≥3 tissue types
- `bbb`: demonstrated systemic IV → CNS crossing in rodents or primates

Full label sources and DOI references are in `data/raw/labels.csv`.

---

## Methods

### 1. Sequence Embedding with ESM-2

We use Meta's `esm2_t33_650M_UR50D` model — 650M parameters, trained on 250M UniRef50 sequences — to extract per-residue embeddings. Sequences are mean-pooled across residue positions to yield a fixed 1280-dimensional representation per capsid.

```python
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
```

This captures evolutionary context and structural priors that physicochemical features (MW, isoelectric point, hydrophobicity) cannot.

### 2. Multi-Label Classifier

A lightweight MLP (3 hidden layers, dropout, sigmoid output) trained on ESM-2 embeddings with:
- Binary cross-entropy loss per label
- 5-fold cross-validation
- Threshold optimization per label (F1-maximizing threshold)

Also benchmarked against: logistic regression, random forest, SVM — ESM-2 + MLP outperforms physicochemical baselines by ~12 AUROC points on CNS classification.

### 3. Mutational Scanning

For a query sequence, we systematically substitute each residue with all 19 alternatives and re-score with the trained classifier. The **delta-score** (mutant score − wildtype score) is mapped onto the capsid sequence, producing a residue-level sensitivity profile. High-magnitude positions are candidates for rational engineering experiments.

### 4. Interpretability

Gradient × input attribution (via `captum`) identifies which embedding dimensions — and by proxy which input residues — most influence the BBB-crossing prediction. Results are cross-referenced with known VP1 variable regions (VR-I through VR-IX) and published structure–function studies.

---

## Results

### Classifier Performance (5-fold CV)

| Label | AUROC | AUPRC | F1 (opt. threshold) |
|-------|-------|-------|---------------------|
| CNS tropic | 0.91 | 0.87 | 0.84 |
| Peripheral tropic | 0.88 | 0.83 | 0.81 |
| Broad tropic | 0.79 | 0.71 | 0.73 |
| BBB-crossing | 0.93 | 0.89 | 0.86 |

### Known Serotype Predictions

| Serotype | CNS | Peripheral | Broad | BBB | Known phenotype |
|----------|-----|------------|-------|-----|-----------------|
| AAV9 | 0.81 | 0.72 | 0.68 | 0.77 | ✓ CNS/BBB |
| PHP.eB | 0.95 | 0.11 | 0.22 | 0.94 | ✓ CNS/BBB (engineered) |
| AAV8 | 0.18 | 0.93 | 0.61 | 0.09 | ✓ Liver |
| AAV-BR1 | 0.87 | 0.24 | 0.31 | 0.91 | ✓ Brain endothelium |
| AAV2 | 0.61 | 0.55 | 0.82 | 0.38 | ✓ Broad |

*Figures in `results/figures/` include ROC curves, UMAP of capsid embedding space, and mutational scan heatmaps for PHP.eB and AAV9.*

---

## Biological Interpretation

Mutational scanning of PHP.eB reveals that positions in **VR-IV (aa 452–460)** and **VR-VIII (aa 585–600)** have the highest impact on BBB-crossing prediction — consistent with published mutagenesis data showing these loops interact with LY6A, the primary receptor mediating PHP.eB CNS transduction in mice.

The model also successfully recovers known engineering logic: mutations that increase positive surface charge in VR-VIII (e.g., N587K) shift predictions toward higher BBB scores, matching experimental observations in rational design studies.

---

## Limitations

- Training set is small (~114 sequences with curated labels). Predictions should be treated as hypotheses, not ground truth.
- Mouse PHP.eB tropism does not translate to primates; the model is trained primarily on rodent phenotype data and may not generalize to human/NHP contexts.
- Structural features (pore geometry, receptor binding pocket) are not yet integrated; this is a sequence-only model.


---
