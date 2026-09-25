#!/usr/bin/env python
"""Generate multi-label predictions with the trained reference model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.demo_model import ReferenceMLP  # noqa: E402


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "reference_demo")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    feature_path = ROOT / "features" / "merged" / f"{args.split}_features_3387.npy"
    features = np.load(feature_path).astype(np.float32)
    if args.max_samples is not None:
        features = features[: min(args.max_samples, features.shape[0])]

    feature_mean = checkpoint["feature_mean"].cpu().numpy()
    feature_std = checkpoint["feature_std"].cpu().numpy()
    features = (features - feature_mean) / feature_std

    model = ReferenceMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))

    probability_matrix = np.concatenate(probabilities, axis=0)
    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)
    prediction_matrix = (probability_matrix >= threshold).astype(np.int8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = args.output_dir / f"{args.split}_probabilities.npy"
    prediction_path = args.output_dir / f"{args.split}_predictions.npy"
    np.save(probability_path, probability_matrix)
    np.save(prediction_path, prediction_matrix)
    print(f"Device: {device}")
    print(f"Prediction shape: {probability_matrix.shape}")
    print(f"Threshold: {threshold}")
    print(f"Saved probabilities: {probability_path}")
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
