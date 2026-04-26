"""MathVerifyAnswerBlock — sdg_hub block for answer extraction and correctness labelling."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import Field
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry

_BOXED_MARKER = r"\boxed{"


def _extract_boxed(text: str) -> str | None:
    """Return the last \\boxed{...} content with balanced brace matching.

    Uses a brace-counting scan so nested braces (e.g. \\frac{a}{b},
    \\begin{cases}) are handled correctly.
    """
    result = None
    start = 0
    while True:
        idx = text.find(_BOXED_MARKER, start)
        if idx == -1:
            break
        depth = 1
        i = idx + len(_BOXED_MARKER)
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            result = text[idx + len(_BOXED_MARKER) : i - 1]
        start = idx + 1
    return result.strip() if result is not None else None


def _extract_trajectory_text(value) -> str:
    """Unwrap LLMChatBlock message-dict output to plain text."""
    if isinstance(value, list):
        for msg in value:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(value)


def _is_correct(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify

        return bool(verify(parse(predicted), parse(ground_truth)))
    except Exception:
        return predicted.strip() == ground_truth.strip()


@BlockRegistry.register(
    "MathVerifyAnswerBlock",
    category="verification",
    description=(
        "Extracts the final \\boxed{} answer from a trajectory and verifies it "
        "against ground truth using math_verify (falls back to string equality)."
    ),
)
class MathVerifyAnswerBlock(BaseBlock):
    """Extracts and verifies the final answer from a reasoning trajectory.

    Input columns:
    - ``trajectory_text``: the full model-generated solution text.
    - ``ground_truth``: the reference answer string (may itself contain \\boxed{}).

    Output columns:
    - ``extracted_answer``: last \\boxed{...} content, or ``None``.
    - ``correct``: ``True`` if ``extracted_answer`` matches ``ground_truth``
      (via ``math_verify`` when available, string equality otherwise).
    """

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        trajectory_col = self.input_cols[0]
        gt_col = self.input_cols[1]
        # output_cols[0] = extracted_answer column name, output_cols[1] = correct column name
        answer_col = self.output_cols[0]
        correct_col = self.output_cols[1]

        extracted: list[str | None] = []
        correct: list[bool] = []

        for _, row in samples.iterrows():
            trajectory_text: str = _extract_trajectory_text(row[trajectory_col])
            ground_truth: str = str(row[gt_col])

            # Ground truth may itself be wrapped in \boxed{}
            gt_answer = _extract_boxed(ground_truth) or ground_truth.strip()
            pred_answer = _extract_boxed(trajectory_text)

            extracted.append(pred_answer)
            correct.append(_is_correct(pred_answer, gt_answer))

        out = samples.copy()
        out[answer_col] = extracted
        out[correct_col] = correct
        return out
