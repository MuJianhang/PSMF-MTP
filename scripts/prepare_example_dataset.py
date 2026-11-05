#!/usr/bin/env python
"""Prepare a small, aligned example from the released benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from read_fasta_labels import LABELS, read_fasta_with_labels


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=ROOT)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "examples" / "small_demo")
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--test_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_samples < 4 or args.test_samples < 1:
        raise ValueError("Use at least four training samples and one test sample")

    target_files = [
        args.output_dir / "dataset" / f"{split}_new.fasta" for split in ("train", "test")
    ] + [
        args.output_dir / "features" / "merged" / f"{split}_features_3387.npy"
        for split in ("train", "test")
    ] + [args.output_dir / "provenance.json"]
    if not args.overwrite and any(path.exists() for path in target_files):
        raise FileExistsError(
            "Example files already exist. Choose another --output_dir or explicitly use --overwrite."
        )

    rng = np.random.default_rng(args.seed)
    prepared = {}
    for split, count in (("train", args.train_samples), ("test", args.test_samples)):
        frame = read_fasta_with_labels(args.source_root / "dataset" / f"{split}_new.fasta")
        matrix = np.load(
            args.source_root / "features" / "merged" / f"{split}_features_3387.npy",
            mmap_mode="r",
        )
        if matrix.shape != (len(frame), 3387):
            raise ValueError(f"{split}: sequence and feature shapes do not agree")
        if count > len(frame):
            raise ValueError(f"{split}: requested {count} samples, but only {len(frame)} are available")
        indices = np.sort(rng.choice(len(frame), size=count, replace=False))
        selected_features = np.asarray(matrix[indices], dtype=np.float32)
        if not np.isfinite(selected_features).all():
            raise ValueError(f"{split}: selected features contain NaN or infinite values")
        selected_frame = frame.iloc[indices]
        records = []
        for source_index, (_, row) in zip(indices, selected_frame.iterrows()):
            bits = "".join(str(int(row[label])) for label in LABELS)
            records.append(f">{split}_{source_index:06d} labels={bits}\n{row['sequence']}\n")
        prepared[split] = (indices, selected_features, "".join(records))

    provenance = {
        "purpose": "Small aligned input example from the released benchmark.",
        "source": "Released train_new.fasta/test_new.fasta and their aligned feature matrices.",
        "seed": args.seed,
        "feature_dimension": 3387,
        "label_names": LABELS,
        "splits": {},
    }
    for split, (indices, features, fasta_text) in prepared.items():
        fasta_out = args.output_dir / "dataset" / f"{split}_new.fasta"
        feature_out = args.output_dir / "features" / "merged" / f"{split}_features_3387.npy"
        fasta_out.parent.mkdir(parents=True, exist_ok=True)
        feature_out.parent.mkdir(parents=True, exist_ok=True)
        fasta_out.write_text(fasta_text, encoding="utf-8")
        np.save(feature_out, features)
        provenance["splits"][split] = {
            "sample_count": len(indices),
            "source_row_indices_zero_based": indices.tolist(),
        }
        print(f"{split}: {len(indices)} sequences, features {features.shape}")
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared example: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
