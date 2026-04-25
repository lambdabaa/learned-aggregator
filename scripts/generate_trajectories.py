"""Generate trajectories for learned aggregator training data.

Samples N trajectories per problem from the MATH train split using
Qwen2.5-1.5B-Instruct and scores each step with MLXProcessRewardModel.

Output: JSONL files (train/val/test) in data/ directory.
Each line: {"problem": str, "split": str, "trajectories": [
    {"steps": [str, ...], "step_scores": [float, ...],
     "final_answer": str | null, "is_correct": bool}
]}

Usage:
    python scripts/generate_trajectories.py \\
        --lm-endpoint http://localhost:8100/v1 \\
        --lm-model Qwen/Qwen2.5-1.5B-Instruct \\
        --prm-model Qwen/Qwen2.5-Math-PRM-7B \\
        --num-problems 200 \\
        --n-per-problem 8 \\
        --output-dir data/trajectories \\
        --seed 42
"""

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)


def _extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text."""
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def _check_correct(predicted: str | None, ground_truth: str) -> bool:
    """Return True if predicted matches ground_truth."""
    if predicted is None:
        return False
    try:
        from math_verify import verify, parse

        return bool(verify(parse(predicted), parse(ground_truth)))
    except Exception:
        return predicted.strip() == ground_truth.strip()


def _load_math_problems(num_problems: int, seed: int) -> list[dict]:
    """Load problems from the MATH train split."""
    from datasets import load_dataset

    ds = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
    rng = __import__("random")
    rng.seed(seed)
    problems = list(ds)
    rng.shuffle(problems)
    return problems[:num_problems]


def _split_problems(
    problems: list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """70/15/15 problem-level split — all trajectories from a problem stay together."""
    import random

    rng = random.Random(seed)
    shuffled = problems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


async def _generate_and_score(
    problems: list[dict],
    lm_endpoint: str,
    lm_model: str,
    prm_model: str,
    n_per_problem: int,
    step_sep: str = "\n\n",
    max_steps: int = 15,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate trajectories and score them for a list of problems."""
    from tqdm import tqdm

    from its_hub.integration.mlx_prm import MLXProcessRewardModel
    from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
    from its_hub.utils import QWEN_SYSTEM_PROMPT

    lm = OpenAICompatibleLanguageModel(
        endpoint=lm_endpoint,
        model=lm_model,
        temperature=temperature,
    )
    sg = StepGeneration(step_token=step_sep, max_steps=max_steps)
    prm = MLXProcessRewardModel(model_name=prm_model, step_sep=step_sep)

    results: list[dict] = []
    for problem in tqdm(problems, desc="Generating trajectories"):
        question = problem["problem"]
        answer = problem["solution"]  # ground-truth solution text
        # Extract the final boxed answer from the ground truth solution
        gt_answer = _extract_boxed(answer) or answer

        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        trajectories: list[dict] = []
        for _ in range(n_per_problem):
            try:
                steps, step_scores = await _sample_trajectory(
                    lm, sg, prm, messages, question
                )
            except Exception:
                continue

            full_response = step_sep.join(steps)
            final_answer = _extract_boxed(full_response)
            is_correct = _check_correct(final_answer, gt_answer)
            trajectories.append(
                {
                    "steps": steps,
                    "step_scores": step_scores,
                    "final_answer": final_answer,
                    "is_correct": is_correct,
                }
            )

        results.append(
            {
                "problem": question,
                "ground_truth": gt_answer,
                "trajectories": trajectories,
            }
        )

    return results


async def _sample_trajectory(
    lm,
    sg,
    prm,
    messages: list[dict],
    prompt_str: str,
) -> tuple[list[str], list[float]]:
    """Sample one trajectory and return (steps, step_scores)."""
    from its_hub.types import ChatMessages

    chat_messages = ChatMessages(messages)
    steps: list[str] = []
    step_scores: list[float] = []

    # Incrementally generate and score
    for _ in range(sg.max_steps):
        forward_results = await sg.aforward(lm, [prompt_str], [steps])
        next_step, is_stopped = forward_results[0]
        steps.append(next_step)

        # Score trajectory up to this step
        current_text = sg._post_process(steps, stopped=True)
        score = await prm.ascore(chat_messages, current_text)
        step_scores.append(score if isinstance(score, float) else score)

        if is_stopped:
            break

    return steps, step_scores


def _write_jsonl(records: list[dict], path: str, split_name: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({**r, "split": split_name}) + "\n")
    print(f"Wrote {len(records)} problems to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-endpoint", default="http://localhost:8100/v1")
    parser.add_argument("--lm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--n-per-problem", type=int, default=8)
    parser.add_argument("--output-dir", default="data/trajectories")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--step-sep", default="\n\n")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    problems = _load_math_problems(args.num_problems, args.seed)
    train_problems, val_problems, test_problems = _split_problems(
        problems, seed=args.seed
    )

    print(f"Split: {len(train_problems)} train / {len(val_problems)} val / {len(test_problems)} test")

    for split_name, split_problems in [
        ("train", train_problems),
        ("val", val_problems),
        ("test", test_problems),
    ]:
        records = asyncio.run(
            _generate_and_score(
                split_problems,
                lm_endpoint=args.lm_endpoint,
                lm_model=args.lm_model,
                prm_model=args.prm_model,
                n_per_problem=args.n_per_problem,
                step_sep=args.step_sep,
                max_steps=args.max_steps,
                temperature=args.temperature,
            )
        )
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        _write_jsonl(records, out_path, split_name)

    print("Done.")


if __name__ == "__main__":
    main()
