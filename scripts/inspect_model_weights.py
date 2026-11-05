#!/usr/bin/env python
"""Verify released PSMF-MTP weights without importing the retained core implementation."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "weights/model_config.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    path = args.checkpoint or args.config.parent / config["checkpoint"]["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match model_config.json")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a tensor state_dict")
    if tuple(state["output_layer.weight"].shape) != (21, 80):
        raise ValueError("Output layer differs from the documented 21-label configuration")
    if tuple(state["global_pos"].shape) != (1, 41, 80):
        raise ValueError("Fusion token configuration differs from model_config.json")
    groups = {int(key.split(".")[1]) for key in state if key.startswith("group_tokenizers.")}
    if groups != set(range(8)):
        raise ValueError("Expected eight descriptor-group mappings")
    for prefix in ["seq_tokenizer.", "mamba_fusion_blocks.", "transformer_classifier."]:
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f"Missing model component: {prefix}")
    parameter_values = sum(value.numel() for value in state.values())
    if parameter_values != config["checkpoint"]["parameter_values"]:
        raise ValueError("Parameter count differs from model_config.json")
    print(json.dumps({
        "checkpoint": str(path),
        "epochs": config["training"]["epochs"],
        "seed": config["training"]["seed"],
        "descriptor_dimension": config["input"]["descriptor_dimension"],
        "sequence_dimension": config["input"]["sequence_dimension"],
        "output_labels": len(config["labels"]),
        "parameter_values": parameter_values,
        "sha256": digest,
    }, indent=2))
    print("PSMF-MTP WEIGHTS CHECK: PASS")


if __name__ == "__main__":
    main()
