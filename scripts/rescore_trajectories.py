"""Re-score existing trajectory JSONL files with the PRM.

The corpus generation run (generate_trajectories.py) produced degenerate
step scores (all ~0.5003) because the MLX 1.5B model occupied ~14 GB of
Metal memory during 187 min of generation, leaving insufficient headroom
for the 7B PRM when it loaded on the same 16 GB unified-memory machine.
The <extra_0> positions were found but logits were corrupted by MPS memory
pressure.

This script loads ONLY the PRM (no MLX model) and rescores in a fresh
process.  Trajectories are read from existing JSONL files; only step_scores
are updated.

Usage:
    .venv/bin/python scripts/rescore_trajectories.py \\
        --input-dir data/trajectories \\
        --prm-model Qwen/Qwen2.5-Math-PRM-7B \\
        --step-sep $'\\n\\n' \\
        [--splits train val test]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

STEP_SEP_DEFAULT = "\n\n"


def _load_prm(model_name: str):
    from its_hub.integration.transformers_prm import TransformersProcessRewardModel
    print(f"Loading PRM: {model_name} ...", flush=True)
    prm = TransformersProcessRewardModel(model_name=model_name)
    # Diagnostic: verify logit shape with a dummy call
    dummy_scores = prm.score("What is 1+1?", ["The answer is 2."])
    print(f"  PRM diagnostic — dummy score: {dummy_scores} (expect non-degenerate)", flush=True)
    return prm


def _rescore_file(path: str, prm, step_sep: str, dry_run: bool) -> dict:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    total_traj = sum(len(r["trajectories"]) for r in records)
    print(f"  {len(records)} problems, {total_traj} trajectories", flush=True)

    all_new_scores: list[float] = []
    for rec_idx, rec in enumerate(records):
        problem = rec["problem"]
        for traj in rec["trajectories"]:
            text = traj["trajectory_text"]
            steps = text.split(step_sep)
            steps = [s for s in steps if s.strip()]  # drop blank steps
            if not steps:
                steps = [text]
            new_scores = [float(s) for s in prm.score(problem, steps)]
            traj["step_scores"] = new_scores
            all_new_scores.extend(new_scores)

        if (rec_idx + 1) % 10 == 0:
            print(f"    scored {rec_idx + 1}/{len(records)} problems ...", flush=True)

    import statistics
    print(f"  Score stats — mean={statistics.mean(all_new_scores):.4f} "
          f"stdev={statistics.stdev(all_new_scores):.4f} "
          f"min={min(all_new_scores):.4f} max={max(all_new_scores):.4f}", flush=True)

    degenerate = sum(1 for s in all_new_scores if abs(s - 0.5) < 0.01)
    print(f"  Degenerate scores (|s-0.5|<0.01): {degenerate}/{len(all_new_scores)}", flush=True)

    if not dry_run:
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Written: {path}", flush=True)

    return {"total": len(all_new_scores), "degenerate": degenerate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/trajectories")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--step-sep", default=STEP_SEP_DEFAULT,
                        help="Step separator string (default: two newlines)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Score but do not overwrite files")
    args = parser.parse_args()

    prm = _load_prm(args.prm_model)

    for split in args.splits:
        path = os.path.join(args.input_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"Skipping {split} — {path} not found", flush=True)
            continue
        print(f"\nRescoring {split} split: {path}", flush=True)
        stats = _rescore_file(path, prm, args.step_sep, args.dry_run)
        print(f"  Done: {stats}", flush=True)

    print("\nRescore complete.", flush=True)


if __name__ == "__main__":
    main()
