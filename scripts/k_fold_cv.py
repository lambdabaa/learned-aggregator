"""Nested 5-fold cross-validation for trajectory aggregators.

Outer loop (k=5): performance estimation.
Inner split (20% of outer train): HP selection.

HP search spaces
  MLP:  hidden_width  in {4, 6, 8, 16, 32}
  LSTM: hidden_size   in {4, 8, 16}
  GBDT: n_estimators x max_depth in {50,100,200,500} x {2,3,4,5}

Fixed HPs across all runs: MLP/LSTM lr=1e-3, epochs=200, patience=10,
batch_size=64; GBDT lr=0.05, subsample=0.8, random_state=42.

Reports mean ± std selection accuracy across outer folds, plus the HP
chosen by each fold's inner selection.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.learned_aggregator.features import extract_features
from src.learned_aggregator.model import TrajectoryMLP, TrajectoryLSTM

SEED = 42
MLP_WIDTHS  = [4, 6, 8, 16, 32]
LSTM_WIDTHS = [4, 8, 16]
GBDT_GRID   = [(n, d) for n in [50, 100, 200, 500] for d in [2, 3, 4, 5]]
K_OUTER     = 5
INNER_VAL_FRAC = 0.20

FEATURE_NAMES = [
    "mean", "min", "max", "last", "length", "variance",
    "pos_min_norm", "pos_max_norm", "last_minus_first", "gap_at_min",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_all_problems(*paths: str) -> list[dict]:
    problems = []
    for p in paths:
        with open(p) as f:
            for line in f:
                problems.append(json.loads(line))
    return problems


def build_features(problems: list[dict]):
    """Return (X, y, seqs) for MLP/GBDT features and raw sequences for LSTM."""
    X, y, seqs = [], [], []
    for prob in problems:
        for t in prob["trajectories"]:
            X.append(extract_features(t["step_scores"]))
            y.append(float(t["is_correct"]))
            seqs.append(t["step_scores"])
    return np.array(X), np.array(y), seqs


def select_best(trajectories: list[dict], score_fn) -> bool:
    scores = [score_fn(t["step_scores"]) for t in trajectories]
    return bool(trajectories[int(np.argmax(scores))]["is_correct"])


def selection_accuracy(problems: list[dict], score_fn) -> float:
    return float(np.mean([select_best(p["trajectories"], score_fn) for p in problems]))


# ---------------------------------------------------------------------------
# Baseline score functions
# ---------------------------------------------------------------------------

def _prod(s):    return math.prod(s) if s else 0.0
def _min(s):     return min(s) if s else 0.0
def _mean(s):    return sum(s) / len(s) if s else 0.0
def _geomean(s): return math.exp(sum(math.log(max(v, 1e-9)) for v in s) / len(s)) if s else 0.0
def _random(s):  return random.random()


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------

def train_mlp(
    X_tr, y_tr, X_v, y_v,
    hidden_width: int,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 64,
    seed: int = SEED,
) -> TrajectoryMLP:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = TrajectoryMLP(hidden_width=hidden_width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr, dtype=torch.float32)
    X_v_t   = torch.tensor(X_v,  dtype=torch.float32)
    y_v_t   = torch.tensor(y_v,  dtype=torch.float32)

    best_val, best_state, no_improve = float("inf"), None, 0
    n = len(X_tr_t)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            opt.zero_grad()
            crit(model(X_tr_t[b]), y_tr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(X_v_t), y_v_t))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# LSTM training
# ---------------------------------------------------------------------------

def _pad(seqs):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    X = torch.zeros(len(seqs), max_len, 1)
    for i, s in enumerate(seqs):
        X[i, :len(s), 0] = torch.tensor(s, dtype=torch.float32)
    return X, torch.tensor(lengths, dtype=torch.int64)


def train_lstm(
    seqs_tr, y_tr, seqs_v, y_v,
    hidden_size: int,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 10,
    batch_size: int = 64,
    seed: int = SEED,
) -> TrajectoryLSTM:
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = TrajectoryLSTM(hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    X_tr_pad, L_tr = _pad(seqs_tr)
    X_v_pad,  L_v  = _pad(seqs_v)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_v_t  = torch.tensor(y_v,  dtype=torch.float32)

    best_val, best_state, no_improve = float("inf"), None, 0
    n = len(seqs_tr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            opt.zero_grad()
            crit(model(X_tr_pad[b], L_tr[b]), y_tr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(X_v_pad, L_v), y_v_t))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# GBDT training
# ---------------------------------------------------------------------------

def train_gbdt(X_tr, y_tr, n_estimators: int, max_depth: int, seed: int = SEED):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.05, subsample=0.8, random_state=seed,
    )
    clf.fit(X_tr, y_tr.astype(int))
    return clf


# ---------------------------------------------------------------------------
# Score function builders
# ---------------------------------------------------------------------------

def mlp_score_fn(model: TrajectoryMLP):
    model.eval()
    def fn(step_scores):
        with torch.no_grad():
            x = torch.tensor(extract_features(step_scores), dtype=torch.float32).unsqueeze(0)
            return float(model(x).squeeze())
    return fn


def lstm_score_fn(model: TrajectoryLSTM):
    model.eval()
    def fn(step_scores):
        x = torch.tensor([[v] for v in step_scores], dtype=torch.float32).unsqueeze(0)
        l = torch.tensor([len(step_scores)])
        with torch.no_grad():
            return float(model(x, l))
    return fn


def gbdt_score_fn(clf):
    def fn(step_scores):
        return float(clf.predict_proba([extract_features(step_scores)])[0][1])
    return fn


# ---------------------------------------------------------------------------
# K-fold split
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int, seed: int = SEED):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx  = idx[start:end]
        train_idx = idx[:start] + idx[end:]
        folds.append((train_idx, test_idx))
    return folds


def inner_split(train_idx: list[int], val_frac: float = INNER_VAL_FRAC, seed: int = SEED):
    rng = random.Random(seed)
    idx = list(train_idx)
    rng.shuffle(idx)
    split = int(len(idx) * (1 - val_frac))
    return idx[:split], idx[split:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dirs", nargs="+",
                        default=["data/trajectories_combined"],
                        help="Directories containing train/val/test JSONL files")
    parser.add_argument("--output", default="results/kfold_cv.json")
    args = parser.parse_args()

    # Pool all problems
    problems = []
    for d in args.data_dirs:
        for split in ["train", "val", "test"]:
            p = Path(d) / f"{split}.jsonl"
            if p.exists():
                with open(p) as f:
                    for line in f:
                        problems.append(json.loads(line))
    print(f"Pooled {len(problems)} problems")

    folds = kfold_indices(len(problems), K_OUTER)

    # accumulators: model_name -> list of per-fold accuracy
    fold_accs   = {}  # model -> [fold_acc, ...]
    fold_hps    = {}  # model -> [chosen_hp, ...]

    def record(name, acc, hp=None):
        fold_accs.setdefault(name, []).append(acc)
        if hp is not None:
            fold_hps.setdefault(name, []).append(hp)

    for fold_i, (outer_train_idx, outer_test_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Outer fold {fold_i+1}/{K_OUTER}  "
              f"(train={len(outer_train_idx)}, test={len(outer_test_idx)})")

        outer_train_probs = [problems[i] for i in outer_train_idx]
        outer_test_probs  = [problems[i] for i in outer_test_idx]

        inner_tr_idx, inner_v_idx = inner_split(outer_train_idx)
        inner_tr_probs = [problems[i] for i in inner_tr_idx]
        inner_v_probs  = [problems[i] for i in inner_v_idx]

        print(f"  Inner split: train={len(inner_tr_probs)}, val={len(inner_v_probs)}")

        X_itr, y_itr, seqs_itr = build_features(inner_tr_probs)
        X_iv,  y_iv,  seqs_iv  = build_features(inner_v_probs)
        X_otr, y_otr, seqs_otr = build_features(outer_train_probs)

        # ---- Baselines (no training) ----
        for name, fn in [("random", _random), ("prod", _prod),
                         ("geomean", _geomean), ("min", _min), ("mean", _mean)]:
            acc = selection_accuracy(outer_test_probs, fn)
            record(name, acc)

        # ---- MLP: inner HP selection then outer retrain ----
        print("  MLP HP selection...")
        best_mlp_w, best_mlp_acc = None, -1
        for w in MLP_WIDTHS:
            m = train_mlp(X_itr, y_itr, X_iv, y_iv, hidden_width=w)
            acc = selection_accuracy(inner_v_probs, mlp_score_fn(m))
            if acc > best_mlp_acc:
                best_mlp_acc, best_mlp_w = acc, w
        print(f"  MLP best inner width={best_mlp_w} (val_sel={best_mlp_acc:.3f})")
        m_mlp = train_mlp(X_otr, y_otr, X_iv, y_iv, hidden_width=best_mlp_w)
        acc = selection_accuracy(outer_test_probs, mlp_score_fn(m_mlp))
        record("mlp", acc, best_mlp_w)
        print(f"  MLP outer test acc={acc:.3f}")

        # ---- LSTM: inner HP selection then outer retrain ----
        print("  LSTM HP selection...")
        best_lstm_h, best_lstm_acc = None, -1
        for h in LSTM_WIDTHS:
            m = train_lstm(seqs_itr, y_itr, seqs_iv, y_iv, hidden_size=h)
            acc = selection_accuracy(inner_v_probs, lstm_score_fn(m))
            if acc > best_lstm_acc:
                best_lstm_acc, best_lstm_h = acc, h
        print(f"  LSTM best inner h={best_lstm_h} (val_sel={best_lstm_acc:.3f})")
        m_lstm = train_lstm(seqs_otr, y_otr, seqs_iv, y_iv, hidden_size=best_lstm_h)
        acc = selection_accuracy(outer_test_probs, lstm_score_fn(m_lstm))
        record("lstm", acc, best_lstm_h)
        print(f"  LSTM outer test acc={acc:.3f}")

        # ---- GBDT: inner HP selection then outer retrain ----
        print("  GBDT HP selection...")
        best_gbdt_hp, best_gbdt_acc = None, -1
        for n_est, depth in GBDT_GRID:
            clf = train_gbdt(X_itr, y_itr, n_est, depth)
            acc = selection_accuracy(inner_v_probs, gbdt_score_fn(clf))
            if acc > best_gbdt_acc:
                best_gbdt_acc, best_gbdt_hp = acc, (n_est, depth)
        print(f"  GBDT best inner (n={best_gbdt_hp[0]}, d={best_gbdt_hp[1]}) "
              f"(val_sel={best_gbdt_acc:.3f})")
        clf_gbdt = train_gbdt(X_otr, y_otr, *best_gbdt_hp)
        acc = selection_accuracy(outer_test_probs, gbdt_score_fn(clf_gbdt))
        record("gbdt", acc, best_gbdt_hp)
        print(f"  GBDT outer test acc={acc:.3f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"{'Model':12s}  {'Mean':>6}  {'Std':>6}  {'95% CI':>16}  HP choices")
    print("-" * 70)

    results = {}
    for name in ["random", "prod", "geomean", "min", "mean", "mlp", "lstm", "gbdt"]:
        accs = fold_accs[name]
        mean = np.mean(accs)
        std  = np.std(accs, ddof=1)
        # t-distribution 95% CI with k-1 degrees of freedom
        from scipy import stats
        t_crit = stats.t.ppf(0.975, df=K_OUTER - 1)
        se  = std / math.sqrt(K_OUTER)
        ci_lo, ci_hi = mean - t_crit * se, mean + t_crit * se
        hps = fold_hps.get(name, [])
        print(f"{name:12s}  {mean:.3f}   {std:.3f}   "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]  {hps}")
        results[name] = {
            "fold_accs": accs,
            "mean": mean, "std": std,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "hp_per_fold": hps,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
