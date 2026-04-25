"""MLXProcessRewardScoreBlock — sdg_hub block for per-step PRM scoring on Apple Silicon."""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
from pydantic import Field
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


@BlockRegistry.register(
    "MLXProcessRewardScoreBlock",
    category="prm",
    description=(
        "Scores each step of a reasoning trajectory using Qwen2.5-Math-PRM-7B "
        "at 4-bit quantisation via MLX. Outputs step_scores as list[float]."
    ),
)
class MLXProcessRewardScoreBlock(BaseBlock):
    """Scores per-step trajectory quality using MLXProcessRewardModel.

    For each row the block:
    1. Splits ``trajectory_text`` on ``step_sep`` to recover individual steps.
    2. Calls ``MLXProcessRewardModel.score()`` once per prefix (problem + steps
       so far), producing one float per step.
    3. Writes the full ``list[float]`` to the ``step_scores`` output column.

    The underlying ``MLXProcessRewardModel`` is constructed lazily on the first
    ``generate()`` call so that importing this block does not load the 4-bit
    model weights — which is important for test environments without MLX.
    """

    # -----------------------------------------------------------------
    # Pydantic fields (all serialisable; model instance kept as private attr)
    # -----------------------------------------------------------------
    model_name: str = Field(
        default="Qwen/Qwen2.5-Math-PRM-7B",
        description="HuggingFace model name or local path for the PRM.",
    )
    step_sep: str = Field(
        default="\n\n",
        description="Step separator used by StepGeneration (must match LLMChatBlock output).",
    )

    # -----------------------------------------------------------------
    # Private lazy-init slot (not a Pydantic field)
    # -----------------------------------------------------------------
    _prm: Any = None  # holds MLXProcessRewardModel after first call

    def _get_prm(self):
        if self._prm is None:
            from its_hub.integration.mlx_prm import MLXProcessRewardModel

            self._prm = MLXProcessRewardModel(
                model_name=self.model_name,
                step_sep=self.step_sep,
            )
        return self._prm

    # -----------------------------------------------------------------
    # BaseBlock contract
    # -----------------------------------------------------------------
    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        prm = self._get_prm()
        step_scores_col: list[list[float]] = []

        for _, row in samples.iterrows():
            problem: str = row[self.input_cols[0]]
            trajectory_text: str = row[self.input_cols[1]]
            scores = self._score_trajectory(prm, problem, trajectory_text)
            step_scores_col.append(scores)

        out = samples.copy()
        output_col = (
            self.output_cols[0]
            if isinstance(self.output_cols, list)
            else self.output_cols
        )
        out[output_col] = step_scores_col
        return out

    def _score_trajectory(self, prm, problem: str, trajectory_text: str) -> list[float]:
        """Call PRM once per step prefix, accumulating per-step scores."""
        steps = trajectory_text.split(self.step_sep)
        scores: list[float] = []
        for i, _ in enumerate(steps):
            prefix = self.step_sep.join(steps[: i + 1])
            result = prm.score(problem, prefix)
            scores.append(float(result) if not isinstance(result, list) else float(result[0]))
        return scores
