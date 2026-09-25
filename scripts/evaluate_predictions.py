#!/usr/bin/env python
"""Calculate the five multi-label metrics reported for PSMF-MTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".csv":
        matrix = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported matrix format: {path}. Use .npy or headerless .csv")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional matrix, got {matrix.shape} from {path}")
    return np.asarray(matrix)


def multilabel_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction and truth shapes differ: {y_pred.shape} vs {y_true.shape}")

    y_pred = (y_pred > 0).astype(np.int8)
    y_true = (y_true > 0).astype(np.int8)

    intersection = np.logical_and(y_pred, y_true).sum(axis=1).astype(np.float64)
    union = np.logical_or(y_pred, y_true).sum(axis=1).astype(np.float64)
    predicted_count = y_pred.sum(axis=1).astype(np.float64)
    true_count = y_true.sum(axis=1).astype(np.float64)

    precision_per_sample = np.divide(
        intersection,
        predicted_count,
        out=np.zeros_like(intersection),
        where=(intersection > 0) & (predicted_count > 0),
    )
    coverage_per_sample = np.divide(
        intersection,
        true_count,
        out=np.zeros_like(intersection),
        where=(intersection > 0) & (true_count > 0),
    )
    accuracy_per_sample = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=(intersection > 0) & (union > 0),
    )

    exact_match = np.all(y_pred == y_true, axis=1)
    hamming_loss = np.not_equal(y_pred, y_true).mean(axis=1)

    return {
        "accuracy": float(accuracy_per_sample.mean()),
        "precision": float(precision_per_sample.mean()),
        "coverage": float(coverage_per_sample.mean()),
        "absolute_true": float(exact_match.mean()),
        "absolute_false": float(hamming_loss.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true", type=Path, required=True, help="Binary ground-truth matrix")
    parser.add_argument("--pred", type=Path, required=True, help="Probability or binary prediction matrix")
    parser.add_argument("--threshold", type=float, default=0.57, help="Threshold for probability predictions")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    y_true = load_matrix(args.true)
    scores_or_labels = load_matrix(args.pred)
    if scores_or_labels.shape != y_true.shape:
        raise ValueError(
            f"Prediction and truth shapes differ: {scores_or_labels.shape} vs {y_true.shape}"
        )

    is_binary = np.all(np.isin(scores_or_labels, [0, 1]))
    y_pred = scores_or_labels.astype(np.int8) if is_binary else (scores_or_labels >= args.threshold)
    result = {
        "threshold": None if is_binary else args.threshold,
        "shape": list(y_true.shape),
        "metrics": multilabel_metrics(y_pred, y_true),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
