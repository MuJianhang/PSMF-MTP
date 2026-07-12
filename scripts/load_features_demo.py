#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
load_features_demo.py

Non-core utility script for the released PSMF-MTP features.

This script is adapted to the current released feature filenames:

    PSMF-MTP/
      features/
        AAC/
          train_aac.csv
          test_AAC.csv   or test_aac.csv
        DDE/
          train_dde.csv
          test_dde.csv
        CKSAAGP/
          train_cksaagp.csv
          test_cksaagp.csv
        PAAC/
          train_paac.csv
          test_paac.csv
        APAAC/
          train_apaac.csv
          test_apaac.csv
        AAindex1/
          aaindex1.my.csv
          train_aaindex1.npy
          test_aaindex1.npy
        CTD/
          train_ctd.npy
          test_ctd.npy
        MorganFP/
          train_morganfp_400.npy
          test_morganfp_400.npy

It automatically locates the project root from the script location, so it can
be run directly from PyCharm/Anaconda even if the working directory is different.

Final concatenated feature order:
    AAC:       0-400
    DDE:       400-800
    CKSAAGP:   800-1200
    PAAC:      1200-1596
    APAAC:     1596-1980
    AAindex1:  1980-2546
    CTD:       2546-2987
    MorganFP:  2987-3387
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


RAW_DIMS = {
    "AAC": 20,
    "DDE": 400,
    "CKSAAGP": 100,
    "PAAC": 22,
    "APAAC": 24,
    "AAindex1": 566,
    "CTD": 441,
    "MorganFP": 400,
}

EXPANSION = {
    "AAC": 20,
    "DDE": 1,
    "CKSAAGP": 4,
    "PAAC": 18,
    "APAAC": 16,
    "AAindex1": 1,
    "CTD": 1,
    "MorganFP": 1,
}

FINAL_DIMS = {name: RAW_DIMS[name] * EXPANSION[name] for name in RAW_DIMS}

FEATURE_ORDER = [
    "AAC",
    "DDE",
    "CKSAAGP",
    "PAAC",
    "APAAC",
    "AAindex1",
    "CTD",
    "MorganFP",
]


def project_root_from_script() -> Path:
    """
    If this file is placed in PSMF-MTP/scripts/, the project root is its parent.
    """
    return Path(__file__).resolve().parents[1]


def with_optional_csv(path_without_suffix: Path) -> list[Path]:
    """
    Return candidates for files that may be shown without extension on Windows.
    """
    return [path_without_suffix.with_suffix(".csv"), path_without_suffix]


def resolve_existing(candidates: Iterable[Path], feature_name: str) -> Path:
    """
    Return the first existing path from candidates.
    """
    candidates = list(candidates)
    for p in candidates:
        if p.exists():
            return p

    msg = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"{feature_name} feature file not found. Tried:\n{msg}\n\n"
        f"Please check whether the file name and folder name are correct."
    )


def _drop_leading_index_column(arr: np.ndarray, feature_name: str, expected_dim: int) -> np.ndarray:
    """
    The CSV files for AAC/DDE/CKSAAGP/PAAC/APAAC usually contain one leading
    index column. If the matrix has expected_dim + 1 columns, drop the first column.
    """
    if arr.shape[1] == expected_dim + 1:
        return arr[:, 1:]

    if arr.shape[1] == expected_dim:
        return arr

    raise ValueError(
        f"{feature_name} dimension mismatch. Expected {expected_dim} feature columns "
        f"or {expected_dim + 1} columns with a leading index column, got {arr.shape[1]} columns."
    )


def load_csv_raw_feature(path: Path, feature_name: str) -> np.ndarray:
    """Load a raw CSV feature matrix and remove the leading index column if needed."""
    expected_dim = RAW_DIMS[feature_name]

    # Most released CSV files have no header, so header=None is used.
    arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    arr = _drop_leading_index_column(arr, feature_name, expected_dim)
    return arr


def expand_feature(arr: np.ndarray, feature_name: str) -> np.ndarray:
    """
    Expand raw features by repeating each original column according to the
    feature-specific expansion factor.
    """
    repeat = EXPANSION[feature_name]
    out = arr if repeat == 1 else np.repeat(arr, repeats=repeat, axis=1)

    expected_final_dim = FINAL_DIMS[feature_name]
    if out.shape[1] != expected_final_dim:
        raise ValueError(
            f"{feature_name} expanded dimension mismatch. "
            f"Expected {expected_final_dim}, got {out.shape[1]}."
        )

    return out.astype(np.float32)


def load_npy_feature(path: Path, feature_name: str) -> np.ndarray:
    """Load an NPY feature matrix and validate its dimension."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{feature_name} should be 2D, got shape {arr.shape}.")

    expected_dim = FINAL_DIMS[feature_name]
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f"{feature_name} dimension mismatch. Expected {expected_dim}, got {arr.shape[1]}."
        )

    return arr


def feature_files(feature_root: Path, split: str) -> Dict[str, Path]:
    """
    Return existing feature file paths.

    The current preferred filenames are:
        train_cksaagp.csv / test_cksaagp.csv
        train_paac.csv    / test_paac.csv
        train_apaac.csv   / test_apaac.csv

    Some older names are also supported for compatibility.
    """
    feature_root = Path(feature_root)

    if split == "train":
        return {
            "AAC": resolve_existing([
                feature_root / "AAC" / "train_aac.csv",
                feature_root / "AAC" / "etec_train_aac.csv",
            ], "AAC"),
            "DDE": resolve_existing([
                feature_root / "DDE" / "train_dde.csv",
                feature_root / "DDE" / "etec_train_dde.csv",
            ], "DDE"),
            "CKSAAGP": resolve_existing([
                *with_optional_csv(feature_root / "CKSAAGP" / "train_cksaagp"),
                feature_root / "CKSAAGP" / "train_cksaagp3.csv",
                feature_root / "CKSAAGP" / "etec_train_cksaagp3.csv",
            ], "CKSAAGP"),
            "PAAC": resolve_existing([
                *with_optional_csv(feature_root / "PAAC" / "train_paac"),
                feature_root / "PAAC" / "train_paac2.csv",
                feature_root / "PAAC" / "etec_train_paac2.csv",
            ], "PAAC"),
            "APAAC": resolve_existing([
                *with_optional_csv(feature_root / "APAAC" / "train_apaac"),
                feature_root / "APAAC" / "train_apaac2.csv",
                feature_root / "APAAC" / "etec_train_apaac2.csv",
            ], "APAAC"),
            "AAindex1": resolve_existing([
                feature_root / "AAindex1" / "train_aaindex1.npy",
                feature_root / "AAindex1" / "train_aaindex1_566.npy",
            ], "AAindex1"),
            "CTD": resolve_existing([
                feature_root / "CTD" / "train_ctd.npy",
                feature_root / "CTD" / "train_ctd_pure441.npy",
            ], "CTD"),
            "MorganFP": resolve_existing([
                feature_root / "MorganFP" / "train_morganfp_400.npy",
            ], "MorganFP"),
        }

    if split == "test":
        return {
            "AAC": resolve_existing([
                feature_root / "AAC" / "test_AAC.csv",
                feature_root / "AAC" / "test_aac.csv",
                feature_root / "AAC" / "etec_test_AAC.csv",
                feature_root / "AAC" / "etec_test_aac.csv",
            ], "AAC"),
            "DDE": resolve_existing([
                feature_root / "DDE" / "test_dde.csv",
                feature_root / "DDE" / "etec_test_dde.csv",
            ], "DDE"),
            "CKSAAGP": resolve_existing([
                *with_optional_csv(feature_root / "CKSAAGP" / "test_cksaagp"),
                feature_root / "CKSAAGP" / "test_cksaagp3.csv",
                feature_root / "CKSAAGP" / "etec_test_cksaagp3.csv",
            ], "CKSAAGP"),
            "PAAC": resolve_existing([
                *with_optional_csv(feature_root / "PAAC" / "test_paac"),
                feature_root / "PAAC" / "test_paac2.csv",
                feature_root / "PAAC" / "etec_test_paac2.csv",
            ], "PAAC"),
            "APAAC": resolve_existing([
                *with_optional_csv(feature_root / "APAAC" / "test_apaac"),
                feature_root / "APAAC" / "test_apaac2.csv",
                feature_root / "APAAC" / "etec_test_apaac2.csv",
            ], "APAAC"),
            "AAindex1": resolve_existing([
                feature_root / "AAindex1" / "test_aaindex1.npy",
                feature_root / "AAindex1" / "test_aaindex1_566.npy",
            ], "AAindex1"),
            "CTD": resolve_existing([
                feature_root / "CTD" / "test_ctd.npy",
                feature_root / "CTD" / "test_ctd_pure441.npy",
            ], "CTD"),
            "MorganFP": resolve_existing([
                feature_root / "MorganFP" / "test_morganfp_400.npy",
            ], "MorganFP"),
        }

    raise ValueError("split must be 'train' or 'test'.")


def load_all_features(feature_root: Path, split: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load and concatenate all eight feature groups.

    Returns
    -------
    X : np.ndarray
        Concatenated 3387-dimensional feature matrix.
    groups : dict
        Individual feature matrices after expansion.
    """
    files = feature_files(feature_root, split)
    groups: Dict[str, np.ndarray] = {}

    print(f"\nUsing feature root: {Path(feature_root).resolve()}")

    for name in FEATURE_ORDER:
        path = files[name]

        if path.suffix.lower() == ".csv" or path.suffix == "":
            raw = load_csv_raw_feature(path, name)
            groups[name] = expand_feature(raw, name)
        elif path.suffix.lower() == ".npy":
            groups[name] = load_npy_feature(path, name)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

        print(f"{split:5s} {name:9s}: {groups[name].shape} <- {path}")

    sample_counts = {name: groups[name].shape[0] for name in FEATURE_ORDER}
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"Sample count mismatch across feature groups: {sample_counts}")

    X = np.concatenate([groups[name] for name in FEATURE_ORDER], axis=1)
    if X.shape[1] != 3387:
        raise ValueError(f"Concatenated feature dimension should be 3387, got {X.shape[1]}.")

    print(f"{split:5s} concatenated features: {X.shape}")
    return X, groups


def main() -> None:
    root = project_root_from_script()
    default_feature_root = root / "features"
    default_out_dir = root / "features" / "merged"

    parser = argparse.ArgumentParser(
        description="Load and concatenate the eight released feature groups."
    )
    parser.add_argument(
        "--feature_root",
        type=Path,
        default=default_feature_root,
        help="Path to the released features directory. Default: <project_root>/features",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for merged 3387-dimensional feature arrays. Default: <project_root>/features/merged",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root detected as: {root}")
    print("Loading training features...")
    X_train, _ = load_all_features(args.feature_root, "train")

    print("\nLoading test features...")
    X_test, _ = load_all_features(args.feature_root, "test")

    train_out = args.out_dir / "train_features_3387.npy"
    test_out = args.out_dir / "test_features_3387.npy"

    np.save(train_out, X_train)
    np.save(test_out, X_test)

    print("\nSaved merged feature files:")
    print(f"  {train_out}")
    print(f"  {test_out}")

    print("\nFeature dimension order:")
    start = 0
    for name in FEATURE_ORDER:
        end = start + FINAL_DIMS[name]
        print(f"  {name:9s}: {start:4d}-{end:4d}")
        start = end


if __name__ == "__main__":
    main()
