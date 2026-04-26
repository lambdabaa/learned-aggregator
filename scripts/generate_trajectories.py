"""Thin runner for the trajectory corpus sdg_hub flow.

The canonical pipeline description lives in flows/trajectory_corpus.yaml.
This script:
  1. Imports custom blocks (triggering BlockRegistry registration).
  2. Loads MATH train-split problems into a pandas DataFrame.
  3. Applies a 70/15/15 problem-level split (seed-pinned).
  4. Runs flows/trajectory_corpus.yaml on each split.
  5. Writes per-split JSONL files to --output-dir.

Usage:
    python scripts/generate_trajectories.py \\
        --lm-endpoint http://localhost:8100/v1 \\
        --lm-model Qwen/Qwen2.5-1.5B-Instruct \\
        --prm-model Qwen/Qwen2.5-Math-PRM-7B \\
        --num-problems 200 \\
        --output-dir data/trajectories \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import pandas as pd

# Register custom blocks in sdg_hub's BlockRegistry before loading the flow.
import learned_aggregator.blocks  # noqa: F401

from sdg_hub import Flow

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
FLOW_YAML = os.path.join(os.path.dirname(__file__), "..", "flows", "trajectory_corpus.yaml")


def _load_math_problems(num_problems: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
    problems = list(ds)
    rng = random.Random(seed)
    rng.shuffle(problems)
    return problems[:num_problems]


def _split(problems: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = problems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _to_df(problems: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "problem_id": idx,
            "problem": p["problem"],
            "ground_truth": p["solution"],
            "level": str(p.get("level", "")),  # MATH difficulty level "Level 1"–"Level 5"
        }
        for idx, p in enumerate(problems)
    ])


def _write_jsonl(df: pd.DataFrame, path: str, split_name: str) -> None:
    """Group trajectory rows by problem and write one JSON record per problem.

    The RowMultiplierBlock fans out each problem to N trajectory rows.
    train_aggregator.py and evaluate.py expect grouped records:
      {"problem": ..., "trajectories": [{"step_scores": ..., "is_correct": ...}, ...]}
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    records = []
    for problem_id, group in df.groupby("problem_id", sort=True):
        row0 = group.iloc[0]
        trajectories = []
        for _, row in group.iterrows():
            step_scores = row.get("step_scores", [])
            if hasattr(step_scores, "tolist"):
                step_scores = step_scores.tolist()
            trajectories.append({
                "trajectory_text": str(row.get("trajectory_text", "")),
                "step_scores": step_scores if isinstance(step_scores, list) else [],
                "is_correct": bool(row.get("correct", False)),
                "extracted_answer": row.get("extracted_answer"),
            })
        records.append({
            "problem_id": int(problem_id),
            "problem": str(row0["problem"]),
            "ground_truth": str(row0["ground_truth"]),
            "level": str(row0.get("level", "")),
            "split": split_name,
            "trajectories": trajectories,
        })
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records)} problems ({len(df)} trajectory rows) to {path}")


def _check_lm_endpoint(endpoint: str) -> None:
    """Fail fast if the LM inference endpoint is unreachable."""
    import urllib.request
    import urllib.error

    models_url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=5) as resp:
            if resp.status != 200:
                raise SystemExit(
                    f"LM endpoint {models_url} returned HTTP {resp.status}.\n"
                    "Start the inference server before running this script.\n"
                    "Example: its-iaas --host 0.0.0.0 --port 8100 (or use vLLM)"
                )
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach LM endpoint at {models_url}: {exc.reason}\n"
            "Start the inference server before running this script.\n"
            "Example (its-iaas):\n"
            "  its-iaas --host 0.0.0.0 --port 8100 &\n"
            "  # then configure: POST http://localhost:8100/configure\n"
            "Example (vLLM on Linux/CUDA):\n"
            "  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8100"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-endpoint", default="http://localhost:8100/v1")
    parser.add_argument("--lm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Checking LM endpoint at {args.lm_endpoint} ...")
    _check_lm_endpoint(args.lm_endpoint)
    print("  OK\n")

    problems = _load_math_problems(args.num_problems, args.seed)
    train_probs, val_probs, test_probs = _split(problems, args.seed)
    print(f"Split: {len(train_probs)} train / {len(val_probs)} val / {len(test_probs)} test")

    flow = Flow.from_yaml(FLOW_YAML)
    flow.set_model_config(
        model=f"openai/{args.lm_model}",
        api_base=args.lm_endpoint,
        api_key="NO_API_KEY",
    )
    # Pass the PRM model name to the scoring block at runtime
    runtime_params = {"score_steps": {"model_name": args.prm_model}}

    for split_name, split_probs in [("train", train_probs), ("val", val_probs), ("test", test_probs)]:
        print(f"\nGenerating {split_name} split ({len(split_probs)} problems)...")
        df = _to_df(split_probs)
        result_df = flow.generate(df, runtime_params=runtime_params)
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        _write_jsonl(result_df, out_path, split_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
