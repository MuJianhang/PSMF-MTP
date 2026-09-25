#!/usr/bin/env python
"""Train the runnable multi-label reference model on the released features."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from model.demo_model import ReferenceMLP  # noqa: E402
from read_fasta_labels import LABELS, read_fasta_with_labels  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_config(path: Path) -> dict[str, float | int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    for name in (
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "hidden_dim",
        "dropout",
        "validation_fraction",
        "threshold",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "demo_config.json")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "reference_demo")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--validation_fraction", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    seed = int(config["seed"])
    set_seed(seed)
    device = choose_device(args.device)

    feature_path = ROOT / "features" / "merged" / "train_features_3387.npy"
    label_path = ROOT / "dataset" / "train_new.fasta"
    features = np.load(feature_path).astype(np.float32)
    label_frame = read_fasta_with_labels(label_path)
    labels = label_frame[LABELS].to_numpy(dtype=np.float32)

    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"Feature/label sample mismatch: {features.shape[0]} vs {labels.shape[0]}")
    if features.shape[1] != 3387 or labels.shape[1] != 21:
        raise ValueError(f"Unexpected input shapes: features={features.shape}, labels={labels.shape}")

    rng = np.random.default_rng(seed)
    selected = rng.permutation(features.shape[0])
    if args.max_train_samples is not None:
        if args.max_train_samples < 4:
            raise ValueError("--max_train_samples must be at least 4")
        selected = selected[: min(args.max_train_samples, features.shape[0])]
    features = features[selected]
    labels = labels[selected]

    validation_fraction = float(config["validation_fraction"])
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    validation_size = max(1, int(round(features.shape[0] * validation_fraction)))
    train_size = features.shape[0] - validation_size
    if train_size < 2:
        raise ValueError("Not enough samples remain for training")

    train_features = features[:train_size]
    validation_features = features[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]

    feature_mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    train_features = (train_features - feature_mean) / feature_std
    validation_features = (validation_features - feature_mean) / feature_std

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation_features),
        torch.from_numpy(validation_labels),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    model = ReferenceMLP(
        input_dim=features.shape[1],
        hidden_dim=int(config["hidden_dim"]),
        output_dim=labels.shape[1],
        dropout=float(config["dropout"]),
    ).to(device)

    positive = torch.from_numpy(train_labels.sum(axis=0)).to(device)
    negative = train_labels.shape[0] - positive
    positive_weight = torch.where(positive > 0, negative / positive.clamp_min(1.0), torch.ones_like(positive))
    positive_weight = positive_weight.clamp(min=1.0, max=50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_state = None
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * batch_features.shape[0]

        model.eval()
        validation_loss_sum = 0.0
        with torch.inference_mode():
            for batch_features, batch_labels in validation_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                loss = criterion(model(batch_features), batch_labels)
                validation_loss_sum += float(loss.item()) * batch_features.shape[0]

        train_loss = train_loss_sum / len(train_dataset)
        validation_loss = validation_loss_sum / len(validation_dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:03d}/{int(config['epochs']):03d} "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "reference_model.pt"
    history_path = args.output_dir / "training_history.json"
    checkpoint = {
        "model_state": best_state,
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(config["hidden_dim"]),
        "output_dim": int(labels.shape[1]),
        "dropout": float(config["dropout"]),
        "threshold": float(config["threshold"]),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "label_names": LABELS,
        "reference_pipeline": True,
    }
    torch.save(checkpoint, checkpoint_path)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
