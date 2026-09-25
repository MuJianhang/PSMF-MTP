#!/usr/bin/env python
"""Validate the released PSMF-MTP files, labels, and feature matrices."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from load_features_demo import load_all_features  # noqa: E402


EXPECTED_COUNTS = {"train": 7872, "test": 1969}
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "dataset/train_new.fasta",
    "dataset/test_new.fasta",
    "dataset/LABELS.md",
    "pre_trained_model/config.json",
    "pre_trained_model/model.safetensors",
    "pre_trained_model/vocab.json",
    "pre_trained_model/pretraining_metadata.json",
    "scripts/read_fasta_labels.py",
    "scripts/load_features_demo.py",
    "scripts/extract_pretrained_features_demo.py",
    "scripts/evaluate_predictions.py",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def inspect_fasta(path: Path, expected_count: int) -> None:
    count = 0
    min_len = None
    max_len = 0
    current_header = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal count, min_len, max_len, current_header, sequence_parts
        if current_header is None:
            return
        if re.search(r"[01]{21}", current_header) is None:
            fail(f"No 21-bit label vector in {path}: >{current_header}")
        sequence = "".join(sequence_parts).strip().upper()
        if not sequence:
            fail(f"Empty sequence in {path}: >{current_header}")
        length = len(sequence)
        min_len = length if min_len is None else min(min_len, length)
        max_len = max(max_len, length)
        count += 1

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        flush()

    if count != expected_count:
        fail(f"{path.name}: expected {expected_count} sequences, found {count}")
    if min_len is None or min_len < 1 or max_len > 50:
        fail(f"{path.name}: unexpected sequence-length range {min_len}-{max_len}")
    print(f"PASS FASTA {path.name}: {count} sequences, length {min_len}-{max_len}")


def main() -> None:
    missing = [str(ROOT / rel) for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    if missing:
        fail("Missing required files:\n  " + "\n  ".join(missing))
    print(f"PASS required files: {len(REQUIRED_FILES)}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "<<<<<<<" in readme or ">>>>>>>" in readme or "=======" in readme:
        fail("README.md contains unresolved merge-conflict markers")
    print("PASS README conflict-marker check")

    for split, count in EXPECTED_COUNTS.items():
        inspect_fasta(ROOT / "dataset" / f"{split}_new.fasta", count)

    train, _ = load_all_features(ROOT / "features", "train")
    test, _ = load_all_features(ROOT / "features", "test")
    if train.shape != (7872, 3387):
        fail(f"Unexpected training-feature shape: {train.shape}")
    if test.shape != (1969, 3387):
        fail(f"Unexpected test-feature shape: {test.shape}")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        fail("Released features contain NaN or infinite values")
    print("PASS released feature matrices are finite and have expected shapes")

    merged_train = ROOT / "features" / "merged" / "train_features_3387.npy"
    merged_test = ROOT / "features" / "merged" / "test_features_3387.npy"
    if merged_train.exists() and np.load(merged_train, mmap_mode="r").shape != train.shape:
        fail("Existing merged training matrix has an unexpected shape")
    if merged_test.exists() and np.load(merged_test, mmap_mode="r").shape != test.shape:
        fail("Existing merged test matrix has an unexpected shape")
    print("PASS merged-feature shape check")

    print("\nPUBLIC RELEASE CHECK: PASS")


if __name__ == "__main__":
    main()
