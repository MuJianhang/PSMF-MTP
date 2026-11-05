#!/usr/bin/env python
"""Extract the flattened frozen-encoder inputs used by the released checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel

ROOT = Path(__file__).resolve().parents[1]
TOKEN_LENGTH = 50


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header = None
    parts = []

    def flush():
        if header is None:
            return
        sequence = "".join("".join(parts).split()).upper()
        if not 1 <= len(sequence) <= 50:
            raise ValueError(f"Expected 1-50 residues for >{header}; got {len(sequence)}")
        records.append((header, sequence))

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, parts = line[1:].strip(), []
                if not header:
                    raise ValueError("FASTA headers must not be empty")
            else:
                if header is None:
                    raise ValueError("Sequence encountered before a FASTA header")
                parts.append(line)
    flush()
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder_dir", type=Path, default=ROOT / "pre_trained_model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_samples is not None and args.max_samples < 1):
        raise ValueError("Batch size and sample limit must be positive")
    if args.output.suffix.lower() != ".npy":
        raise ValueError("--output must name a .npy file")
    records = read_fasta(args.fasta)
    if args.max_samples is not None:
        records = records[:args.max_samples]
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    vocab = json.loads((args.encoder_dir / "vocab.json").read_text(encoding="utf-8"))
    if int(vocab["[PAD]"]) != 0:
        raise ValueError("The released checkpoint expects padding token id 0")
    encoder = BertModel.from_pretrained(args.encoder_dir, add_pooling_layer=False).to(device)
    encoder.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            ids = np.zeros((len(batch), TOKEN_LENGTH), dtype=np.int64)
            masks = np.zeros_like(ids)
            for row, (_, sequence) in enumerate(batch):
                residues = sequence[:TOKEN_LENGTH - 2]
                tokens = [vocab["[CLS]"]] + [vocab.get(aa, vocab["[UNK]"]) for aa in residues] + [vocab["[SEP]"]]
                ids[row, :len(tokens)] = tokens
                masks[row, :len(tokens)] = 1
            hidden = encoder(
                input_ids=torch.from_numpy(ids).to(device),
                attention_mask=torch.from_numpy(masks).to(device),
            ).last_hidden_state
            # Preserve the selected training run's float16 cache -> float32 input convention.
            flat = hidden.reshape(len(batch), -1).cpu().numpy().astype(np.float16).astype(np.float32)
            pieces.append(flat)
    features = np.concatenate(pieces, axis=0)
    if features.shape != (len(records), 6400) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid encoder output: {features.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    with args.output.with_suffix(".ids.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "fasta_header", "sequence", "encoded_residue_count"])
        for index, (header, sequence) in enumerate(records):
            writer.writerow([index, header, sequence, min(len(sequence), TOKEN_LENGTH - 2)])
    metadata = {
        "input_fasta": str(args.fasta),
        "input_sha256": hashlib.sha256(args.fasta.read_bytes()).hexdigest(),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "token_length_including_special_tokens": TOKEN_LENGTH,
        "maximum_encoded_residues": TOKEN_LENGTH - 2,
        "representation": "flattened last_hidden_state",
        "precision_convention": "float16 rounding followed by float32, matching the selected training run",
        "device": str(device),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Sequences: {len(records)}")
    print(f"Sequence feature shape: {features.shape}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
