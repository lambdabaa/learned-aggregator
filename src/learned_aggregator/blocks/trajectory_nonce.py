"""TrajectoryNonceBlock — makes fanned-out trajectory rows produce diverse LLM outputs.

RowMultiplierBlock creates N identical copies of each input row, so a
prompt-caching LLM server (e.g. mlx_lm) returns the same completion for every
copy.  This block appends a short unique tag to the system message so that
each copy produces a different prompt hash and therefore a different generation.

The tag is stripped-looking enough that it does not materially affect the
model's math reasoning; it just breaks cache identity.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


@BlockRegistry.register(
    "TrajectoryNonceBlock",
    category="transform",
    description=(
        "Appends a unique per-row tag to the system message so that fanned-out "
        "trajectory copies receive distinct prompts and an LLM server with prompt "
        "caching returns diverse completions."
    ),
)
class TrajectoryNonceBlock(BaseBlock):
    """Injects a per-row nonce into the first system message.

    Input columns:
    - ``messages``: list of ``{"role": ..., "content": ...}`` dicts
      (LLMChatBlock format).

    Output columns:
    - Same column name (in-place replacement of the messages list).

    The nonce is the DataFrame row's integer position (0-indexed), formatted
    as ``[t:{n}]`` and appended to the system message content.  If no system
    message is present the tag is prepended to the first user message.
    """

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        input_col = self.input_cols[0]
        output_col = self.output_cols[0]

        new_messages: list = []
        for idx, (_, row) in enumerate(samples.iterrows()):
            messages = row[input_col]
            tagged = _inject_nonce(messages, idx)
            new_messages.append(tagged)

        out = samples.copy()
        out[output_col] = new_messages
        return out


def _inject_nonce(messages: list, idx: int) -> list:
    """Return a copy of *messages* with ``[t:{idx}]`` appended to system/user."""
    tag = f"[t:{idx}]"
    result = []
    injected = False
    for msg in messages:
        msg = dict(msg)
        if not injected and msg.get("role") in ("system", "user"):
            msg["content"] = str(msg.get("content", "")) + f"\n{tag}"
            injected = True
        result.append(msg)
    if not injected and result:
        result[0] = dict(result[0])
        result[0]["content"] = str(result[0].get("content", "")) + f"\n{tag}"
    return result
