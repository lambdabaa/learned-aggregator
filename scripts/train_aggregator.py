"""Train a TrajectoryMLP aggregator on scored trajectory data.

Loads JSONL files produced by generate_trajectories.py, extracts fixed-length
features from each trajectory's step_scores, and trains a 2-layer MLP with
binary cross-entropy loss and early stopping on validation loss.

Saves the trained checkpoint to its_hub/aggregators/checkpoints/mlp_agg.pt
in a format compatible with LearnedMLPAggregator.

Usage:
    python scripts/train_aggregator.py \\
        --data-dir data/trajectories \\
        --checkpoint-out ../../its_hub/its_hub/aggregators/checkpoints/mlp_agg.pt \\
        --hidden-width 16 \\
        --lr 1e-3 \\
        --epochs 200 \\
        --patience 10 \\
        --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.learned_aggregator.features import extract_features
from src.learned_aggregator.model import TrajectoryMLP

FEATURE_NAMES = [
    "mean", "min", "max", "last", "length", "variance",
    "pos_min_norm", "pos_max_norm", "last_minus_first", "gap_at_min",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert problem records → (features, labels) arrays."""
    features_list: list[np.ndarray] = []
    labels_list: list[float] = []
    for rec in records:
        for traj in rec.get("trajectories", []):
            step_scores = traj.get("step_scores", [])
            is_correct = traj.get("is_correct", False)
            feat = extract_features(step_scores)
            features_list.append(feat)
            labels_list.append(1.0 if is_correct else 0.0)
    X = np.stack(features_list).astype(np.float32)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_width: int = 16,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    checkpoint_out: str = "mlp_agg.pt",
) -> TrajectoryMLP:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = TrajectoryMLP(input_dim=X_train.shape[1], hidden_width=hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(torch.tensor(X_train))
            train_loss = criterion(train_preds, torch.tensor(y_train)).item()
            train_acc = _accuracy(train_preds.numpy(), y_train)

            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = _accuracy(val_preds.numpy(), y_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        train_acc = _accuracy(model(torch.tensor(X_train)).numpy(), y_train)
        val_acc = _accuracy(model(X_val_t).numpy(), y_val)
    print(f"\nFinal: train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | gap={train_acc - val_acc:.3f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_width": hidden_width}, checkpoint_out)
    print(f"Checkpoint saved to {checkpoint_out}")

    return model


def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    predicted_labels = (preds >= 0.5).astype(int)
    true_labels = labels.astype(int)
    return float((predicted_labels == true_labels).mean())


# ---------------------------------------------------------------------------
# Weight profile inspection
# ---------------------------------------------------------------------------

def print_weight_profile(model: TrajectoryMLP, feature_names: list[str]) -> None:
    """Print input-layer weight magnitudes per feature (interpretability)."""
    with torch.no_grad():
        w = model.net[0].weight.abs().mean(dim=0).numpy()  # (input_dim,)
    print("\nPer-step weight profile (input-layer mean absolute weight per feature):")
    ranked = sorted(zip(feature_names, w.tolist()), key=lambda x: -x[1])
    for name, magnitude in ranked:
        bar = "█" * int(magnitude * 50)
        print(f"  {name:20s} {magnitude:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/trajectories",
                        help="Directory containing train.jsonl, val.jsonl")
    parser.add_argument(
        "--checkpoint-out",
        default=str(Path(__file__).parent.parent.parent /
                    "its_hub/its_hub/aggregators/checkpoints/mlp_agg.pt"),
    )
    parser.add_argument("--hidden-width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    print(f"Loading train data from {train_path}")
    X_train, y_train = build_dataset(load_jsonl(train_path))
    print(f"  {len(X_train)} trajectories, {y_train.mean():.2%} correct")

    print(f"Loading val data from {val_path}")
    X_val, y_val = build_dataset(load_jsonl(val_path))
    print(f"  {len(X_val)} trajectories, {y_val.mean():.2%} correct")

    print(f"\nTraining MLP (hidden_width={args.hidden_width}, lr={args.lr}, "
          f"epochs={args.epochs}, patience={args.patience})")

    model = train(
        X_train, y_train, X_val, y_val,
        hidden_width=args.hidden_width,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_out=args.checkpoint_out,
    )

    print_weight_profile(model, FEATURE_NAMES)


if __name__ == "__main__":
    main()
