#!/usr/bin/env python
"""Run training, prediction, and evaluation for the reference pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from read_fasta_labels import LABELS, read_fasta_with_labels  # noqa: E402


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "demo_config.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "reference_demo")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    threshold = float(config["threshold"] if args.threshold is None else args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_demo.py"),
        "--config",
        str(args.config),
        "--output_dir",
        str(args.output_dir),
        "--device",
        args.device,
    ]
    for option, value in (
        ("--epochs", args.epochs),
        ("--batch_size", args.batch_size),
        ("--hidden_dim", args.hidden_dim),
        ("--max_train_samples", args.max_train_samples),
        ("--threshold", args.threshold),
    ):
        if value is not None:
            train_command.extend([option, str(value)])
    run(train_command)

    checkpoint_path = args.output_dir / "reference_model.pt"
    predict_command = [
        sys.executable,
        str(ROOT / "scripts" / "predict_demo.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "test",
        "--output_dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--threshold",
        str(threshold),
    ]
    if args.max_test_samples is not None:
        predict_command.extend(["--max_samples", str(args.max_test_samples)])
    run(predict_command)

    label_frame = read_fasta_with_labels(ROOT / "dataset" / "test_new.fasta")
    true_labels = label_frame[LABELS].to_numpy(dtype=np.int8)
    if args.max_test_samples is not None:
        true_labels = true_labels[: min(args.max_test_samples, true_labels.shape[0])]
    true_label_path = args.output_dir / "test_labels.npy"
    np.save(true_label_path, true_labels)

    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_predictions.py"),
            "--true",
            str(true_label_path),
            "--pred",
            str(args.output_dir / "test_probabilities.npy"),
            "--threshold",
            str(threshold),
            "--output",
            str(args.output_dir / "metrics.json"),
        ]
    )

    print("\nReference pipeline completed successfully.")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
