"""Evaluate trajectory aggregators on selection accuracy.

Loads the test-split JSONL, scores all N candidate trajectories with each
aggregator, picks the highest-scoring one, and reports selection accuracy
overall and stratified by trajectory length, problem difficulty, and N.

Bootstrap 95% CIs are computed with 1000 resamples.

Also writes calibration data for reliability diagrams (learned vs prod).

Usage:
    python scripts/evaluate.py \\
        --test-jsonl data/trajectories/test.jsonl \\
        --checkpoint ../../its_hub/its_hub/aggregators/checkpoints/mlp_agg.pt \\
        --output results/eval.json
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.learned_aggregator.features import extract_features

# ---------------------------------------------------------------------------
# Aggregator factories
# ---------------------------------------------------------------------------


def _prod(step_scores: list[float]) -> float:
    if not step_scores:
        return 0.0
    result = 1.0
    for s in step_scores:
        result *= s
    return result


def _min_agg(step_scores: list[float]) -> float:
    return min(step_scores) if step_scores else 0.0


def _mean_agg(step_scores: list[float]) -> float:
    return sum(step_scores) / len(step_scores) if step_scores else 0.0


def _random_agg(step_scores: list[float]) -> float:
    return random.random()


def _make_mlp_agg(checkpoint_path: str) -> Callable[[list[float]], float]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Inline model to avoid import path issues
    import torch.nn as nn
    hidden_width = checkpoint.get("hidden_width", 16)
    net = nn.Sequential(
        nn.Linear(10, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, 1),
        nn.Sigmoid(),
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    def score(step_scores: list[float]) -> float:
        feat = extract_features(step_scores)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(net(x).squeeze(-1).item())

    return score


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_data(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------


def select_best(trajectories: list[dict], agg_fn: Callable, n: int) -> bool:
    """Select best trajectory among first n using agg_fn; return is_correct."""
    candidates = trajectories[:n]
    if not candidates:
        return False
    scores = [agg_fn(t.get("step_scores", [])) for t in candidates]
    best_idx = int(np.argmax(scores))
    return bool(candidates[best_idx].get("is_correct", False))


def _trajectory_length_bucket(traj: dict) -> str:
    n = len(traj.get("steps", traj.get("step_scores", [])))
    if n <= 4:
        return "short(<=4)"
    elif n <= 9:
        return "medium(5-9)"
    else:
        return "long(>=10)"


def bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    means = [arr[rng.choice(len(arr), size=len(arr), replace=True)].mean()
             for _ in range(n_resamples)]
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def evaluate_aggregator(
    records: list[dict],
    agg_fn: Callable,
    n_values: list[int],
) -> dict:
    """Evaluate one aggregator across stratifications."""
    results = {"overall": {}, "by_length": {}, "by_n": {}}

    for n in n_values:
        outcomes = [select_best(r.get("trajectories", []), agg_fn, n) for r in records
                    if len(r.get("trajectories", [])) >= 1]
        if not outcomes:
            continue
        acc = float(np.mean(outcomes))
        lo, hi = bootstrap_ci([float(x) for x in outcomes])
        results["by_n"][n] = {"acc": acc, "ci_lo": lo, "ci_hi": hi, "n_problems": len(outcomes)}

    # Overall: use all trajectories available per problem
    outcomes = [select_best(r.get("trajectories", []), agg_fn, len(r.get("trajectories", [])))
                for r in records if r.get("trajectories")]
    if outcomes:
        acc = float(np.mean(outcomes))
        lo, hi = bootstrap_ci([float(x) for x in outcomes])
        results["overall"] = {"acc": acc, "ci_lo": lo, "ci_hi": hi, "n_problems": len(outcomes)}

    # By trajectory length (using first trajectory as representative)
    by_len: dict[str, list[float]] = {}
    for r in records:
        trajs = r.get("trajectories", [])
        if not trajs:
            continue
        bucket = _trajectory_length_bucket(trajs[0])
        correct = select_best(trajs, agg_fn, len(trajs))
        by_len.setdefault(bucket, []).append(float(correct))
    for bucket, outcomes in by_len.items():
        lo, hi = bootstrap_ci(outcomes)
        results["by_length"][bucket] = {
            "acc": float(np.mean(outcomes)), "ci_lo": lo, "ci_hi": hi,
            "n_problems": len(outcomes),
        }

    return results


def calibration_data(
    records: list[dict],
    agg_fn: Callable,
    n_bins: int = 10,
) -> dict:
    """Collect (predicted_score, is_correct) pairs for reliability diagram."""
    scores, labels = [], []
    for r in records:
        for traj in r.get("trajectories", []):
            scores.append(agg_fn(traj.get("step_scores", [])))
            labels.append(float(traj.get("is_correct", False)))
    # Bin into n_bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= s < hi) for s in scores]
        idxs = [j for j, m in enumerate(mask) if m]
        if idxs:
            bin_accs.append(float(np.mean([labels[j] for j in idxs])))
            bin_confs.append(float(np.mean([scores[j] for j in idxs])))
            bin_counts.append(len(idxs))
        else:
            bin_accs.append(None)
            bin_confs.append(float((lo + hi) / 2))
            bin_counts.append(0)
    return {"bin_accs": bin_accs, "bin_confs": bin_confs, "bin_counts": bin_counts}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", default="data/trajectories/test.jsonl")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).parent.parent.parent /
                    "its_hub/its_hub/aggregators/checkpoints/mlp_agg.pt"),
    )
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--n-values", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading test data from {args.test_jsonl}")
    records = load_test_data(args.test_jsonl)
    print(f"  {len(records)} problems")

    aggregators: dict[str, Callable] = {
        "prod": _prod,
        "min": _min_agg,
        "mean": _mean_agg,
        "random": _random_agg,
    }
    if os.path.exists(args.checkpoint):
        aggregators["learned_mlp"] = _make_mlp_agg(args.checkpoint)
        print(f"Loaded LearnedMLP from {args.checkpoint}")
    else:
        print(f"Warning: checkpoint not found at {args.checkpoint}, skipping learned_mlp")

    all_results: dict = {}
    for name, agg_fn in aggregators.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_aggregator(records, agg_fn, args.n_values)
        all_results[name] = result
        acc = result.get("overall", {}).get("acc")
        ci = (result.get("overall", {}).get("ci_lo"),
              result.get("overall", {}).get("ci_hi"))
        if acc is not None:
            print(f"  Overall acc={acc:.3f} 95%CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

    # Calibration data for learned vs prod
    calib: dict = {}
    for name in ("prod", "learned_mlp"):
        if name in aggregators:
            calib[name] = calibration_data(records, aggregators[name])

    output = {"results": all_results, "calibration": calib}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")

    # Print comparison table
    print("\n=== Selection Accuracy Summary ===")
    header = f"{'Aggregator':15s}  {'Overall':8s}  " + "  ".join(f"N={n}" for n in args.n_values)
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        overall = res.get("overall", {}).get("acc")
        overall_str = f"{overall:.3f}" if overall is not None else "  N/A "
        n_strs = []
        for n in args.n_values:
            acc_n = res.get("by_n", {}).get(n, {}).get("acc")
            n_strs.append(f"{acc_n:.3f}" if acc_n is not None else " N/A ")
        print(f"{name:15s}  {overall_str:8s}  " + "  ".join(n_strs))


if __name__ == "__main__":
    main()
