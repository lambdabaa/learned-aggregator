"""ProcessRewardScoreBlock — sdg_hub block for per-step PRM scoring."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


@BlockRegistry.register(
    "ProcessRewardScoreBlock",
    category="prm",
    description=(
        "Scores each step of a reasoning trajectory using a process reward model. "
        "Defaults to TransformersProcessRewardModel (transformers + MPS/CUDA) which "
        "correctly handles Qwen2.5-Math-PRM-7B's classifier score head. "
        "Set backend='mlx' for Math-Shepherd-style models on Apple Silicon."
    ),
)
class ProcessRewardScoreBlock(BaseBlock):
    """Scores per-step trajectory quality using a pluggable PRM backend.

    For each row the block splits ``trajectory_text`` on ``step_sep`` to
    recover individual steps, calls the PRM once (transformers backend) or
    once per prefix (mlx backend), and writes the full ``list[float]`` of
    per-step scores to the output column.

    The PRM is instantiated lazily on the first ``generate()`` call so that
    importing this block does not load model weights — allowing tests to run
    without hardware.

    Config fields:
        model_name: HuggingFace model name or local path.
        step_sep: Step boundary string emitted by the policy LLM (``"\\n\\n"``
            by default, matching its_hub ``StepGeneration`` convention).
        backend: ``"transformers"`` (default) uses
            ``TransformersProcessRewardModel`` with fp16 on MPS/CUDA.
            ``"mlx"`` uses ``MLXProcessRewardModel`` for Math-Shepherd-style
            models on Apple Silicon.
    """

    model_name: str = Field(
        default="Qwen/Qwen2.5-Math-PRM-7B",
        description="HuggingFace model name or local path for the PRM.",
    )
    step_sep: str = Field(
        default="\n\n",
        description="Step separator used by the policy LLM (must match LLMChatBlock output).",
    )
    backend: Literal["transformers", "mlx"] = Field(
        default="transformers",
        description=(
            "PRM backend.  'transformers' (default) for classifier-head models like "
            "Qwen2.5-Math-PRM-7B.  'mlx' for Math-Shepherd-style generative-token models."
        ),
    )

    _prm: Any = None  # lazy-init slot, not a Pydantic field

    def _get_prm(self):
        if self._prm is None:
            if self.backend == "transformers":
                from its_hub.integration.transformers_prm import TransformersProcessRewardModel

                self._prm = TransformersProcessRewardModel(model_name=self.model_name)
            else:
                from its_hub.integration.mlx_prm import MLXProcessRewardModel

                self._prm = MLXProcessRewardModel(
                    model_name=self.model_name,
                    step_sep=self.step_sep,
                )
        return self._prm

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
            self.output_cols[0] if isinstance(self.output_cols, list) else self.output_cols
        )
        out[output_col] = step_scores_col
        return out

    @staticmethod
    def _extract_text(trajectory_text) -> str:
        """Normalise LLMChatBlock output to a plain string.

        LLMChatBlock stores responses as a list of message dicts
        ``[{"content": "...", "role": "assistant", ...}]``.
        This helper unwraps that to the raw text.
        """
        if isinstance(trajectory_text, str):
            return trajectory_text
        if isinstance(trajectory_text, list):
            for msg in trajectory_text:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
        return str(trajectory_text)

    def _score_trajectory(self, prm, problem: str, trajectory_text) -> list[float]:
        text = self._extract_text(trajectory_text)
        steps = text.split(self.step_sep)
        if self.backend == "transformers":
            # TransformersProcessRewardModel: one call, all steps at once
            return [float(s) for s in prm.score(problem, steps)]
        else:
            # MLXProcessRewardModel: one call per prefix (legacy interface)
            scores: list[float] = []
            for i in range(len(steps)):
                prefix = self.step_sep.join(steps[: i + 1])
                result = prm.score(problem, prefix)
                scores.append(float(result) if not isinstance(result, list) else float(result[0]))
            return scores
