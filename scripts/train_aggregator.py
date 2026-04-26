"""Train a trajectory aggregator on scored trajectory data.

Supports two model types (--model):

  mlp   2-layer MLP trained with BCE + early stopping (default).
        Saves a .pt checkpoint loadable by LearnedMLPAggregator.

  gbdt  Gradient-boosted decision trees via sklearn.
        Saves a .pkl checkpoint loadable by _make_gbdt_agg in evaluate.py.

Usage:
    python scripts/train_aggregator.py \\
        --data-dir data/trajectories_combined \\
        --model mlp --hidden-width 8 \\
        --checkpoint-out ../../its_hub/its_hub/aggregators/checkpoints/mlp_agg_h8.pt

    python scripts/train_aggregator.py \\
        --data-dir data/trajectories_combined \\
        --model gbdt \\
        --checkpoint-out ../../its_hub/its_hub/aggregators/checkpoints/gbdt_agg.pkl
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
    """Convert problem records → (feature_vectors, labels) arrays for MLP/GBDT."""
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


def build_sequence_dataset(records: list[dict]) -> tuple[list[list[float]], np.ndarray]:
    """Convert problem records → (raw_sequences, labels) for LSTM."""
    sequences: list[list[float]] = []
    labels: list[float] = []
    for rec in records:
        for traj in rec.get("trajectories", []):
            step_scores = traj.get("step_scores", [])
            sequences.append(step_scores if step_scores else [0.5])
            labels.append(1.0 if traj.get("is_correct", False) else 0.0)
    return sequences, np.array(labels, dtype=np.float32)


def _pad_sequences(sequences: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences → (padded_tensor, lengths)."""
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.int64)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(sequences), max_len, 1)
    for i, seq in enumerate(sequences):
        t = torch.tensor(seq, dtype=torch.float32)
        padded[i, : len(seq), 0] = t
    return padded, lengths


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

def train_lstm(
    train_seqs: list[list[float]],
    y_train: np.ndarray,
    val_seqs: list[list[float]],
    y_val: np.ndarray,
    hidden_size: int = 8,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    checkpoint_out: str = "lstm_agg.pt",
) -> None:
    from src.learned_aggregator.model import TrajectoryLSTM

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = TrajectoryLSTM(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_pad, L_train = _pad_sequences(train_seqs)
    X_val_pad, L_val = _pad_sequences(val_seqs)
    y_train_t = torch.tensor(y_train)
    y_val_t = torch.tensor(y_val)

    n = len(train_seqs)
    best_val_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start : start + batch_size]
            xb, lb = X_train_pad[b], L_train[b]
            yb = y_train_t[b]
            optimizer.zero_grad()
            criterion(model(xb, lb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_pad, L_train).numpy()
            train_loss = float(criterion(torch.tensor(train_preds), y_train_t))
            train_acc = _accuracy(train_preds, y_train)
            val_preds = model(X_val_pad, L_val)
            val_loss = float(criterion(val_preds, y_val_t))
            val_acc = _accuracy(val_preds.numpy(), y_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_acc = _accuracy(model(X_train_pad, L_train).numpy(), y_train)
        val_acc = _accuracy(model(X_val_pad, L_val).numpy(), y_val)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nFinal: train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | "
          f"gap={train_acc - val_acc:.3f} | params={n_params}")

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": hidden_size}, checkpoint_out)
    print(f"Checkpoint saved to {checkpoint_out}")


def train_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    checkpoint_out: str = "gbdt_agg.pkl",
) -> None:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    clf.fit(X_train, y_train.astype(int))

    train_acc = clf.score(X_train, y_train.astype(int))
    val_acc = clf.score(X_val, y_val.astype(int))
    print(f"\nFinal: train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | gap={train_acc - val_acc:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_out)), exist_ok=True)
    joblib.dump(clf, checkpoint_out)
    print(f"Checkpoint saved to {checkpoint_out}")

    print("\nFeature importances (Gini):")
    ranked = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in ranked:
        bar = "█" * int(imp * 200)
        print(f"  {name:20s} {imp:.4f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/trajectories",
                        help="Directory containing train.jsonl, val.jsonl")
    parser.add_argument("--model", choices=["mlp", "gbdt", "lstm"], default="mlp")
    parser.add_argument(
        "--checkpoint-out",
        default=None,
        help="Output path (.pt for mlp, .pkl for gbdt). Auto-detected if omitted.",
    )
    parser.add_argument("--hidden-width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent.parent.parent / "its_hub/its_hub/aggregators/checkpoints"
    if args.checkpoint_out is None:
        suffix = {"mlp": "mlp_agg.pt", "gbdt": "gbdt_agg.pkl", "lstm": "lstm_agg.pt"}
        args.checkpoint_out = str(ckpt_dir / suffix[args.model])

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")

    print(f"Loading train data from {train_path}")
    X_train, y_train = build_dataset(load_jsonl(train_path))
    print(f"  {len(X_train)} trajectories, {y_train.mean():.2%} correct")

    print(f"Loading val data from {val_path}")
    X_val, y_val = build_dataset(load_jsonl(val_path))
    print(f"  {len(X_val)} trajectories, {y_val.mean():.2%} correct")

    if args.model == "lstm":
        print(f"\nTraining LSTM (hidden_size={args.hidden_width}, lr={args.lr}, "
              f"epochs={args.epochs}, patience={args.patience})")
        train_seqs, y_tr = build_sequence_dataset(load_jsonl(train_path))
        val_seqs, y_v = build_sequence_dataset(load_jsonl(val_path))
        train_lstm(train_seqs, y_tr, val_seqs, y_v,
                   hidden_size=args.hidden_width, lr=args.lr,
                   epochs=args.epochs, patience=args.patience,
                   batch_size=args.batch_size, seed=args.seed,
                   checkpoint_out=args.checkpoint_out)
    elif args.model == "gbdt":
        print("\nTraining GBDT (n_estimators=200, max_depth=3, lr=0.05, subsample=0.8)")
        train_gbdt(X_train, y_train, X_val, y_val,
                   seed=args.seed, checkpoint_out=args.checkpoint_out)
    else:
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
