"""Thin runner for the trajectory corpus sdg_hub flow.

The canonical pipeline description lives in flows/trajectory_corpus.yaml.
This script:
  1. Imports custom blocks (triggering BlockRegistry registration).
  2. Loads MATH problems filtered by level into a pandas DataFrame.
  3. Runs flows/trajectory_corpus.yaml one problem at a time, writing each
     result immediately to a staging JSONL (so crashes resume from where
     they stopped, not from zero).
  4. Optionally splits the staging file into train/val/test.

Usage — generate Level 5 only (original behaviour, writes split files):
    python scripts/generate_trajectories.py \\
        --lm-model Qwen/Qwen2.5-1.5B-Instruct \\
        --prm-model Qwen/Qwen2.5-Math-PRM-7B \\
        --num-problems 200 \\
        --levels 5 \\
        --output-dir data/trajectories

Usage — generate Level 1-4 into a staging file for later merging:
    python scripts/generate_trajectories.py \\
        --lm-model Qwen/Qwen2.5-1.5B-Instruct \\
        --prm-model Qwen/Qwen2.5-Math-PRM-7B \\
        --num-problems 300 \\
        --levels 1 2 3 4 \\
        --output-dir data/trajectories_level14 \\
        --no-split
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

# Dataset to use per level group.  Level 5 problems were historically sampled
# from lighteval/MATH-Hard; Level 1-4 come from the full lighteval/MATH train split.
_DATASET_FOR_LEVELS: dict[frozenset, str] = {
    frozenset({5}): "lighteval/MATH-Hard",
}
_DEFAULT_DATASET = "DigitalLearningGmbH/MATH-lighteval"


def _dataset_for_levels(levels: list[int]) -> str:
    key = frozenset(levels)
    return _DATASET_FOR_LEVELS.get(key, _DEFAULT_DATASET)


def _load_math_problems(num_problems: int, levels: list[int], seed: int, dataset: str | None) -> list[dict]:
    from datasets import load_dataset

    ds_name = dataset or _dataset_for_levels(levels)
    print(f"Loading problems from {ds_name} (levels {levels}) ...", flush=True)
    ds = load_dataset(ds_name, split="train")

    level_strs = {f"Level {l}" for l in levels}
    problems = [p for p in ds if str(p.get("level", "")).strip() in level_strs]
    print(f"  Found {len(problems)} problems matching levels {levels}", flush=True)

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


def _to_single_row_df(problem: dict, problem_id: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "problem_id": problem_id,
        "problem": problem["problem"],
        "ground_truth": problem["solution"],
        "level": str(problem.get("level", "")),
    }])


def _unwrap_trajectory_text(value) -> str:
    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(value)


def _df_to_record(result_df: pd.DataFrame, problem: dict, split_name: str, problem_id: int) -> dict:
    """Convert a flow result DataFrame (N trajectory rows) into one grouped record."""
    trajectories = []
    for _, row in result_df.iterrows():
        step_scores = row.get("step_scores", [])
        if hasattr(step_scores, "tolist"):
            step_scores = step_scores.tolist()
        trajectories.append({
            "trajectory_text": _unwrap_trajectory_text(row.get("trajectory_text", "")),
            "step_scores": step_scores if isinstance(step_scores, list) else [],
            "is_correct": bool(row.get("correct", False)),
            "extracted_answer": row.get("extracted_answer"),
        })
    return {
        "problem_id": problem_id,
        "problem": str(problem["problem"]),
        "ground_truth": str(problem["solution"]),
        "level": str(problem.get("level", "")),
        "split": split_name,
        "trajectories": trajectories,
    }


def _load_staging(path: str) -> tuple[list[dict], set[str]]:
    """Read existing staging file; return (records, set_of_problem_texts)."""
    records: list[dict] = []
    seen: set[str] = set()
    if not os.path.exists(path):
        return records, seen
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                seen.add(rec["problem"])
    return records, seen


def _write_split_jsonl(records: list[dict], path: str, split_name: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for i, rec in enumerate(records):
            rec = dict(rec)
            rec["problem_id"] = i
            rec["split"] = split_name
            f.write(json.dumps(rec) + "\n")
    print(f"  Wrote {len(records)} problems to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--levels", nargs="+", type=int, default=[5],
                        help="MATH difficulty levels to include (e.g. --levels 1 2 3 4)")
    parser.add_argument("--dataset", default=None,
                        help="HuggingFace dataset name (auto-detected from --levels if omitted)")
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--no-split", action="store_true",
                        help="Write all problems to staging.jsonl without train/val/test split")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    problems = _load_math_problems(args.num_problems, args.levels, args.seed, args.dataset)
    print(f"Loaded {len(problems)} problems (levels {args.levels})", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    staging_path = os.path.join(args.output_dir, "staging.jsonl")

    # Resume: skip problems already in the staging file.
    existing_records, seen_problems = _load_staging(staging_path)
    if seen_problems:
        print(f"Resuming: {len(seen_problems)} problems already staged, skipping them.", flush=True)

    flow = Flow.from_yaml(FLOW_YAML)
    runtime_params = {
        "generate_trajectory": {"model_name": args.lm_model, "global_seed": args.seed},
        "score_steps": {"model_name": args.prm_model},
    }

    to_generate = [p for p in problems if p["problem"] not in seen_problems]
    print(f"Generating {len(to_generate)} new problems ...", flush=True)

    total = len(to_generate)
    total_problems = len(seen_problems) + total
    with open(staging_path, "a") as f_out:
        for local_idx, prob in enumerate(to_generate):
            problem_id = len(seen_problems) + local_idx
            problem_num = problem_id + 1  # 1-indexed
            print(f"\n[{problem_num}/{total_problems}] level={prob.get('level', '?')}", flush=True)
            df = _to_single_row_df(prob, problem_id=0)  # always 0; fan-out groups by this
            result_df = flow.generate(df, runtime_params=runtime_params)
            rec = _df_to_record(result_df, prob, split_name="staged", problem_id=problem_id)
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            seen_problems.add(prob["problem"])

    # Re-read staging fully (includes both pre-existing and newly generated).
    all_records, _ = _load_staging(staging_path)
    print(f"\nStaging complete: {len(all_records)} problems in {staging_path}", flush=True)

    if args.no_split:
        print("--no-split: skipping train/val/test split. Run merge_splits.py to combine corpora.")
        return

    # Write train/val/test split files from staging
    train_recs, val_recs, test_recs = _split(all_records, args.seed)
    print(f"Split: {len(train_recs)} train / {len(val_recs)} val / {len(test_recs)} test")
    _write_split_jsonl(train_recs, os.path.join(args.output_dir, "train.jsonl"), "train")
    _write_split_jsonl(val_recs, os.path.join(args.output_dir, "val.jsonl"), "val")
    _write_split_jsonl(test_recs, os.path.join(args.output_dir, "test.jsonl"), "test")
    print("\nDone.")


if __name__ == "__main__":
    main()
