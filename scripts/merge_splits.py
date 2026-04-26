"""Merge two trajectory corpora and re-split 70/15/15.

Combines an existing split corpus (train/val/test JSONL files) with a new
staging JSONL file, then writes a fresh 70/15/15 split to an output directory.

Usage:
    python scripts/merge_splits.py \\
        --existing-dir data/trajectories \\
        --staging data/trajectories_level14/staging.jsonl \\
        --output-dir data/trajectories_combined \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_split_dir(directory: str) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        path = os.path.join(directory, f"{split}.jsonl")
        if os.path.exists(path):
            batch = _load_jsonl(path)
            records.extend(batch)
            print(f"  {split}: {len(batch)} problems from {path}")
        else:
            print(f"  {split}: not found at {path}, skipping")
    return records


def _dedup(records: list[dict]) -> list[dict]:
    """Drop duplicate problems (same problem text), keeping first occurrence."""
    seen: set[str] = set()
    out = []
    for rec in records:
        key = rec["problem"]
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out


def _split(records: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def _write_jsonl(records: list[dict], path: str, split_name: str) -> None:
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
    parser.add_argument("--existing-dir", default="data/trajectories",
                        help="Directory containing train/val/test.jsonl from the original corpus")
    parser.add_argument("--staging", default="data/trajectories_level14/staging.jsonl",
                        help="Staging JSONL file produced by generate_trajectories.py --no-split")
    parser.add_argument("--output-dir", default="data/trajectories_combined",
                        help="Output directory for the merged train/val/test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading existing corpus from {args.existing_dir} ...")
    existing = _load_split_dir(args.existing_dir)
    print(f"  Total: {len(existing)} problems")

    print(f"\nLoading staging corpus from {args.staging} ...")
    staging = _load_jsonl(args.staging)
    print(f"  Total: {len(staging)} problems")

    combined = existing + staging
    before_dedup = len(combined)
    combined = _dedup(combined)
    if len(combined) < before_dedup:
        print(f"\nDe-duplicated: {before_dedup - len(combined)} duplicate problems removed")
    print(f"\nCombined corpus: {len(combined)} problems total")

    # Report level distribution
    from collections import Counter
    level_counts = Counter(r.get("level", "unknown") for r in combined)
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")

    train, val, test = _split(combined, args.seed)
    print(f"\nSplit: {len(train)} train / {len(val)} val / {len(test)} test")

    _write_jsonl(train, os.path.join(args.output_dir, "train.jsonl"), "train")
    _write_jsonl(val, os.path.join(args.output_dir, "val.jsonl"), "val")
    _write_jsonl(test, os.path.join(args.output_dir, "test.jsonl"), "test")

    print(f"\nDone. Merged corpus written to {args.output_dir}/")


if __name__ == "__main__":
    main()
