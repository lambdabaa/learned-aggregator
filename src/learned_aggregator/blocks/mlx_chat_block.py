"""MLXChatBlock — fast diverse trajectory generation using mlx_lm directly.

The mlx_lm HTTP server compiles its sampling function with a baked-in random
state, making temperature sampling completely deterministic: every request with
the same prompt returns the same tokens regardless of seed.  This block calls
mlx_lm's Python API directly, setting mx.random.seed() before each row so
that each fanned-out trajectory gets a unique random path.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry

_SEED_MODULUS = 2**31 - 1  # keep seeds in int32 range for mx compatibility


def _row_seed(row_idx: int, global_seed: int) -> int:
    """Deterministic but unique seed per row so reruns are reproducible."""
    h = hashlib.md5(f"{global_seed}:{row_idx}".encode()).hexdigest()
    return int(h[:8], 16) % _SEED_MODULUS


@BlockRegistry.register(
    "MLXChatBlock",
    category="llm",
    description=(
        "Generates chat completions using mlx_lm directly (no HTTP server). "
        "Sets mx.random.seed per row so fanned-out trajectories are diverse "
        "despite MLX's compiled sampling graph."
    ),
)
class MLXChatBlock(BaseBlock):
    """Replaces LLMChatBlock for MLX-based trajectory generation.

    Each DataFrame row must contain a ``messages`` column with a list of
    ``{"role": ..., "content": ...}`` dicts.  The block generates one
    completion per row and writes it to the output column as a plain string
    (no message-dict wrapper).

    Config fields:
        model_name: HuggingFace model id or local path.
        temperature: Sampling temperature (> 0 enables sampling).
        max_tokens: Maximum new tokens to generate.
        global_seed: Base seed; row seeds are derived deterministically from
            ``global_seed`` and the row's DataFrame position.
    """

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    global_seed: int = 42

    _model: Any = None
    _tokenizer: Any = None

    def _load(self) -> None:
        if self._model is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        import mlx.core as mx
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self._load()

        input_col = self.input_cols[0]
        output_col = self.output_cols[0]

        results: list[str] = []
        for idx, (_, row) in enumerate(samples.iterrows()):
            messages = row[input_col]
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            seed = _row_seed(idx, self.global_seed)
            mx.random.seed(seed)
            sampler = make_sampler(temp=self.temperature)

            parts: list[str] = []
            for resp in stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
            ):
                parts.append(resp.text)
                if resp.finish_reason:
                    break

            results.append("".join(parts))

        out = samples.copy()
        out[output_col] = results
        return out
