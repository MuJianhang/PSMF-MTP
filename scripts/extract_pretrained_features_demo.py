#!/usr/bin/env python
"""Extract demonstration embeddings with the released non-final pretrained encoder."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM


ROOT = Path(__file__).resolve().parents[1]


def read_fasta(path: Path, limit: int | None) -> list[str]:
    sequences: list[str] = []
    parts: list[str] = []

    def flush() -> None:
        if parts:
            sequence = "".join(parts).strip().upper()
            sequence = re.sub(r"[^ARNDCQEGHILKMFPSTWYVX]", "X", sequence)
            sequences.append(sequence)

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                if limit is not None and len(sequences) >= limit:
                    break
                parts = []
            else:
                parts.append(line)
        if limit is None or len(sequences) < limit:
            flush()

    return sequences[:limit] if limit is not None else sequences


def encode_batch(sequences: list[str], vocab: dict[str, int], max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad = vocab["[PAD]"]
    unk = vocab["[UNK]"]
    cls = vocab["[CLS]"]
    sep = vocab["[SEP]"]

    input_ids = []
    attention_masks = []
    residue_masks = []
    for sequence in sequences:
        residues = [vocab.get(residue, unk) for residue in sequence[: max_length - 2]]
        ids = [cls, *residues, sep]
        attention = [1] * len(ids)
        residue_mask = [0, *([1] * len(residues)), 0]
        pad_count = max_length - len(ids)
        ids.extend([pad] * pad_count)
        attention.extend([0] * pad_count)
        residue_mask.extend([0] * pad_count)
        input_ids.append(ids)
        attention_masks.append(attention)
        residue_masks.append(residue_mask)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(residue_masks, dtype=torch.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_dir = ROOT / "pre_trained_model"
    fasta_path = ROOT / "dataset" / f"{args.split}_new.fasta"
    output = args.output or ROOT / "examples" / f"{args.split}_pretrained_features.npy"

    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    model = BertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    max_length = int(model.config.max_position_embeddings)

    sequences = read_fasta(fasta_path, args.limit)
    if not sequences:
        raise RuntimeError(f"No sequences were read from {fasta_path}")

    all_features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            input_ids, attention_mask, residue_mask = encode_batch(batch, vocab, max_length)
            hidden = model.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            weights = residue_mask.unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            all_features.append(pooled.cpu().numpy().astype(np.float32))

    features = np.concatenate(all_features, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    print(f"Sequences: {len(sequences)}")
    print(f"Feature shape: {features.shape}")
    print(f"Saved: {output}")
    print("Note: this demo uses only the released pretrained encoder, not the final PSMF-MTP predictor.")


if __name__ == "__main__":
    main()
