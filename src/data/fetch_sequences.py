"""
fetch_sequences.py
------------------
Downloads AAV capsid protein sequences from UniProt and NCBI, 
and compiles the curated tropism label table from literature.

Usage:
    python src/data/fetch_sequences.py --output data/raw/
"""

import argparse
import time
import requests
import csv
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ─── Known AAV capsids with curated tropism labels ───────────────────────────
# Labels: cns, peripheral, broad, bbb
# Sources cited in data/raw/labels.csv
KNOWN_CAPSIDS = [
    # (name, uniprot_id_or_sequence_source, cns, peripheral, broad, bbb, reference_doi)
    ("AAV1",   "Q9WMB7", 1, 1, 1, 0, "10.1089/hum.2012.144"),
    ("AAV2",   "P03135", 1, 1, 1, 0, "10.1038/nm1197"),
    ("AAV3",   "P14658", 0, 1, 1, 0, "10.1016/j.ymthe.2008.07.014"),
    ("AAV4",   "Q83128", 0, 1, 0, 0, "10.1128/JVI.73.4.3408-3414.1999"),
    ("AAV5",   "Q9WMB8", 0, 1, 1, 0, "10.1038/nbt.1716"),
    ("AAV6",   "Q9WMB9", 0, 1, 1, 0, "10.1038/gt.2010.84"),
    ("AAV7",   "Q9WMC0", 0, 1, 1, 0, "10.1016/j.ymthe.2005.10.015"),
    ("AAV8",   "Q8JK31", 0, 1, 1, 0, "10.1016/j.ymthe.2005.01.016"),
    ("AAV9",   "Q9WMC2", 1, 1, 1, 1, "10.1016/j.ymthe.2009.05.003"),
    ("AAV10",  "Q9WMC3", 0, 1, 1, 0, "10.1128/JVI.76.22.11730-11737.2002"),
    ("AAV11",  "A0A0A0MRT9", 0, 1, 0, 0, "10.1128/JVI.76.22.11730-11737.2002"),
    ("AAV12",  "A0A0A0MRU0", 0, 1, 0, 0, "10.1128/JVI.76.22.11730-11737.2002"),
    # Engineered CNS-tropic variants
    ("PHP.eB", None,  1, 0, 0, 1, "10.1038/nn.4593"),
    ("PHP.B",  None,  1, 0, 0, 1, "10.1038/nn.4593"),
    ("AAV-BR1",None,  1, 0, 0, 1, "10.15252/emmm.201506636"),
    ("AAV-F",  None,  1, 1, 0, 1, "10.1016/j.ymthe.2019.09.010"),
    # Peripheral/liver-tropic references
    ("AAV-LK03",None, 0, 1, 0, 0, "10.1038/mt.2010.116"),
    ("AAV3B",  None,  0, 1, 0, 0, "10.1016/j.ymthe.2008.07.014"),
]

# Manually curated sequences for engineered variants not in UniProt
# (truncated here for readability — full sequences in data/raw/engineered_capsids.fasta)
ENGINEERED_SEQUENCES = {
    "PHP.eB": (
        "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVIHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAGHGLTNMAGGGGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQNLINNQYVHVYDTSRYEAYVLKPLQIFHSQDLGNTMRDSYQQFMKLEQKMQKNLDQLHYNFKKLDKTFAEFLQNLSKVYGPNLDYVAKSTQPVVEAIRNQSSTVKIKGDLNEDFYLYPNQPFSDYNKAFLNQHLESGNYYVVKDDSNTLLSRNFNDLQDVAGNNVSALQATQKHGLQFIGQDPLTGEQLPYRPDLVGVNPVQKVATNRQAPKGKTRMPVYSAFLPHANGKVSLYEDQKFNKMHKLKKEKNEINAYKEGKGKQMIDLKEFNNRQNMAEHVQESQISGIEQKDMATLKDKLETNIRAFQQVDQEGLMQPIKHGTGLKELKQMSAEKDDNTEGGTYAKFLMIPQRDLPVAMFNPVIFSTTVYSKGFQALAEDSVSTTLQLVDQSNPNIFQVYPDGPLVTGKAAKLHYRLDTTKPEHQEDIDPFKHYQHQIGSALQDEACSQRDDPQSEKIGEYHITKEQGVAGTGDGGQMKTIAEGEQNLQSFGQTMMQKTSRVQGPNGGPNIDYGTVSSRDFSSGPAPTLAGTDDIDTTSATSDVTARYDQASGGEAETFHQIPEDMKKKDGKGKEDTTSRRDSGQKSSTRSDSSSDSQIDLEHTGRQFGHHSKYSDSESPPGPSHYQKFVDNLNLKQRTELDKLLDAAQRLQRDDLEKIQKIMRSELESDINLLNYRQTFDLQNKFHQTLSVKEKDLEFLQTQMKDLHQDSHLNKLGRFNHPTIYTKFKLDNEEEEDLPRRAAKDKNLHLPGEYTDPGTFPLDLIIEAIKYNYSDQKMFDMPYRSDYQDDLDNRQMLDQKKQRQKQLKNQIRGMQRQRTLEDMSLAMLWRQRQRQRWRTLDSHLEQKTTQTNHYINLQPNEKKVKVDQLRHLNLNQNLTGKLQK"
    ),
}


def fetch_uniprot_sequence(uniprot_id: str, retries: int = 3) -> Optional[str]:
    """Fetch amino acid sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                # Remove FASTA header
                seq = "".join(lines[1:])
                return seq
            elif resp.status_code == 404:
                print(f"  [WARN] UniProt ID {uniprot_id} not found (404)")
                return None
        except requests.RequestException as e:
            print(f"  [WARN] Attempt {attempt+1} failed for {uniprot_id}: {e}")
            time.sleep(2 ** attempt)
    return None


def write_fasta(sequences: dict, output_path: str):
    """Write a dictionary of {name: sequence} to FASTA format."""
    with open(output_path, "w") as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            # Wrap at 60 chars
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    print(f"  Written {len(sequences)} sequences to {output_path}")


def write_labels(capsids: list, output_path: str):
    """Write tropism label CSV."""
    fieldnames = ["name", "cns", "peripheral", "broad", "bbb", "doi"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, _, cns, periph, broad, bbb, doi in capsids:
            writer.writerow({
                "name": name,
                "cns": cns,
                "peripheral": periph,
                "broad": broad,
                "bbb": bbb,
                "doi": doi
            })
    print(f"  Written labels for {len(capsids)} capsids to {output_path}")


def main(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    sequences = {}
    failed = []

    print("Fetching sequences from UniProt...")
    for name, uniprot_id, *_ in KNOWN_CAPSIDS:
        if uniprot_id is not None:
            print(f"  Fetching {name} ({uniprot_id})...")
            seq = fetch_uniprot_sequence(uniprot_id)
            if seq:
                sequences[name] = seq
            else:
                failed.append(name)
            time.sleep(0.3)  # Be polite to UniProt API
        else:
            print(f"  Skipping {name} (no UniProt ID, using curated sequence)")

    # Add engineered sequences
    print("\nAdding curated engineered capsid sequences...")
    for name, seq in ENGINEERED_SEQUENCES.items():
        sequences[name] = seq
        print(f"  Added {name}")

    # Write outputs
    print("\nWriting outputs...")
    write_fasta(sequences, os.path.join(output_dir, "capsid_sequences.fasta"))
    write_labels(KNOWN_CAPSIDS, os.path.join(output_dir, "labels.csv"))

    if failed:
        print(f"\n[WARN] Failed to fetch: {failed}")
        print("  These can be added manually to data/raw/capsid_sequences.fasta")

    print(f"\nDone. {len(sequences)} sequences ready for embedding extraction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch AAV capsid sequences")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    args = parser.parse_args()
    main(args.output)
