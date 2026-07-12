#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
read_fasta_labels.py

Non-core utility script for the released PSMF-MTP dataset.

This script reads FASTA files whose header contains a 21-dimensional binary
label vector and exports sequence-label CSV files.

Current expected project layout:

    PSMF-MTP/
      dataset/
        train_new.fasta
        test_new.fasta
      scripts/
        read_fasta_labels.py

Expected FASTA example:

    >000000000000000010000
    ARRRRCSDRFRNCPADEALCGRRRR

The 21 label order is:
    AAP, ABP, ACP, ACVP, ADP, AEP, AFP, AHIVP, AHP, AIP, AMRSAP,
    APP, ATP, AVP, BBP, BIP, CPP, DPPIP, QSP, SBP, THP
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


LABELS: List[str] = [
    "AAP", "ABP", "ACP", "ACVP", "ADP", "AEP", "AFP", "AHIVP", "AHP",
    "AIP", "AMRSAP", "APP", "ATP", "AVP", "BBP", "BIP", "CPP",
    "DPPIP", "QSP", "SBP", "THP"
]


def project_root_from_script() -> Path:
    """
    If this file is placed in PSMF-MTP/scripts/, the project root is its parent.
    This allows the script to run correctly from PyCharm/Anaconda even if the
    working directory is not the project root.
    """
    return Path(__file__).resolve().parents[1]


def read_fasta_with_labels(fasta_path: Path) -> pd.DataFrame:
    """
    Read a FASTA file whose header contains a 21-bit binary label vector.

    Parameters
    ----------
    fasta_path : Path
        Path to a FASTA file.

    Returns
    -------
    pd.DataFrame
        Columns:
        sample_id, sequence, length, label_count, AAP, ABP, ..., THP
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records: List[Dict[str, object]] = []
    header = None
    seq_lines: List[str] = []
    sample_index = 0

    def flush_record(h: str | None, lines: List[str]) -> None:
        nonlocal sample_index

        if h is None:
            return

        sequence = "".join(lines).strip().upper().replace(" ", "")
        if not sequence:
            return

        match = re.search(r"[01]{21}", h)
        if match is None:
            raise ValueError(
                f"Cannot find a 21-bit label vector in FASTA header: >{h}"
            )

        label_values = [int(x) for x in match.group(0)]

        row: Dict[str, object] = {
            "sample_id": f"sample_{sample_index:06d}",
            "sequence": sequence,
            "length": len(sequence),
            "label_count": int(sum(label_values)),
        }
        row.update({label: value for label, value in zip(LABELS, label_values)})
        records.append(row)
        sample_index += 1

    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                flush_record(header, seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

        flush_record(header, seq_lines)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid records parsed from: {fasta_path}")

    return df


def summarize(df: pd.DataFrame, name: str) -> None:
    """Print a basic dataset summary."""
    print(f"\n[{name}]")
    print(f"Samples: {len(df)}")
    print(
        f"Sequence length: min={df['length'].min()}, "
        f"max={df['length'].max()}, mean={df['length'].mean():.2f}"
    )
    print(f"Single-label samples: {(df['label_count'] == 1).sum()}")
    print(f"Multi-label samples: {(df['label_count'] > 1).sum()}")
    print(f"Max labels per sample: {df['label_count'].max()}")
    print("\nLabel counts:")
    print(df[LABELS].sum(axis=0).astype(int).to_string())


def main() -> None:
    root = project_root_from_script()
    default_dataset_dir = root / "dataset"
    default_out_dir = root / "parsed_labels"

    parser = argparse.ArgumentParser(
        description="Parse released FASTA files with 21-bit labels."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset_dir,
        help="Directory containing train_new.fasta and test_new.fasta. "
             "Default: <project_root>/dataset",
    )
    parser.add_argument(
        "--train_fasta",
        type=str,
        default="train_new.fasta",
        help="Training FASTA filename.",
    )
    parser.add_argument(
        "--test_fasta",
        type=str,
        default="test_new.fasta",
        help="Test FASTA filename.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for parsed CSV files. "
             "Default: <project_root>/parsed_labels",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.dataset_dir / args.train_fasta
    test_path = args.dataset_dir / args.test_fasta

    print(f"Project root detected as: {root}")
    print(f"Training FASTA: {train_path}")
    print(f"Test FASTA: {test_path}")

    train_df = read_fasta_with_labels(train_path)
    test_df = read_fasta_with_labels(test_path)

    summarize(train_df, "Training set")
    summarize(test_df, "Test set")

    train_out = args.out_dir / "train_sequences_labels.csv"
    test_out = args.out_dir / "test_sequences_labels.csv"

    train_df.to_csv(train_out, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_out, index=False, encoding="utf-8-sig")

    print("\nSaved parsed CSV files:")
    print(f"  {train_out}")
    print(f"  {test_out}")


if __name__ == "__main__":
    main()
