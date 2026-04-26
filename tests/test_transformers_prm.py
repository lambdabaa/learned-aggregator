"""Unit tests for TransformersProcessRewardModel using mocked transformers.

All tests bypass __init__ or mock at the model level so they run without
Apple Silicon hardware, GPU, or actual model weights.
"""

from __future__ import annotations

import asyncio
import math
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from its_hub.base import AbstractProcessRewardModel


# ---------------------------------------------------------------------------
# Helpers: build a TransformersProcessRewardModel with controlled mocks
# ---------------------------------------------------------------------------

def _make_transformer_mocks(
    extra0_positions: list[int],
    pos_neg_logits: list[tuple[float, float]],
    seq_len: int = 12,
):
    """Return (model_mock, tokenizer_mock) with deterministic logits.

    Args:
        extra0_positions: Token positions (0-indexed) where <extra_0> appears.
        pos_neg_logits: (positive_logit, negative_logit) for each <extra_0> position.
        seq_len: Total sequence length of the mock token sequence.
    """
    extra0_token_id = 77  # arbitrary ID not used elsewhere

    # Build token IDs with extra0 at specified positions
    token_ids = list(range(1, seq_len + 1))
    for pos in extra0_positions:
        token_ids[pos] = extra0_token_id

    # Build logits tensor: (1, seq_len, 2)
    logits = torch.zeros(1, seq_len, 2)
    for pos, (pos_logit, neg_logit) in zip(extra0_positions, pos_neg_logits):
        logits[0, pos, 0] = neg_logit   # class 0 = negative
        logits[0, pos, 1] = pos_logit   # class 1 = positive

    # Tokenizer mock
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "mock chat text"
    tokenizer.return_value = {
        "input_ids": torch.tensor([token_ids]),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }
    tokenizer.convert_tokens_to_ids.return_value = extra0_token_id

    # model(input_ids, ...) → TokenClassifierOutput with logits attribute
    token_classifier_out = MagicMock()
    token_classifier_out.logits = logits

    model = MagicMock()
    model.return_value = token_classifier_out  # model(...) returns output.logits

    return model, tokenizer


def _build_prm(
    extra0_positions: list[int] | None = None,
    pos_neg_logits: list[tuple[float, float]] | None = None,
    seq_len: int = 12,
):
    """Construct TransformersProcessRewardModel bypassing __init__."""
    from its_hub.integration.transformers_prm import TransformersProcessRewardModel

    if extra0_positions is None:
        extra0_positions = [4, 8]
    if pos_neg_logits is None:
        pos_neg_logits = [(2.0, 0.0)] * len(extra0_positions)

    model_mock, tokenizer_mock = _make_transformer_mocks(
        extra0_positions, pos_neg_logits, seq_len
    )

    prm = TransformersProcessRewardModel.__new__(TransformersProcessRewardModel)
    prm._device = "cpu"
    prm._tokenizer = tokenizer_mock
    prm._model = model_mock
    prm._extra0_token_id = 77  # matches _make_transformer_mocks

    return prm


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------

class TestTransformersProcessRewardModelInterface:
    def test_conforms_to_abstract_interface(self):
        prm = _build_prm()
        assert isinstance(prm, AbstractProcessRewardModel)

    def test_missing_transformers_raises_import_error(self):
        """Constructing without transformers installed raises ImportError."""
        with patch.dict(sys.modules, {"transformers": None}):
            from its_hub.integration.transformers_prm import TransformersProcessRewardModel

            with pytest.raises(ImportError, match="transformers"):
                TransformersProcessRewardModel()


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

class TestTransformersProcessRewardModelScore:
    def test_score_returns_list_of_floats_in_unit_interval(self):
        prm = _build_prm()
        with patch.object(prm, "_score_steps", return_value=[0.8, 0.6]):
            result = prm.score("What is 2+2?", ["step 1", "step 2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, float) for s in result)
        assert all(0.0 <= s <= 1.0 for s in result)

    def test_score_length_matches_step_count(self):
        prm = _build_prm()
        steps = ["s1", "s2", "s3"]
        with patch.object(prm, "_score_steps", return_value=[0.7, 0.8, 0.9]):
            result = prm.score("q", steps)
        assert len(result) == 3

    def test_softmax_numerics_at_extra0_position(self):
        """Positive-class probability uses 2-class softmax of the score head logits."""
        pos_logit, neg_logit = 3.0, 1.0
        shift = max(pos_logit, neg_logit)
        expected = math.exp(pos_logit - shift) / (
            math.exp(pos_logit - shift) + math.exp(neg_logit - shift)
        )

        # One <extra_0> at position 4 with controlled logits
        prm = _build_prm(extra0_positions=[4], pos_neg_logits=[(pos_logit, neg_logit)])
        result = prm._score_steps("question", ["single step"])
        assert result[0] == pytest.approx(expected, abs=1e-4)

    def test_two_steps_produce_two_scores(self):
        prm = _build_prm(
            extra0_positions=[3, 7],
            pos_neg_logits=[(2.0, 0.0), (1.0, 1.0)],
        )
        result = prm._score_steps("q", ["step one", "step two"])
        assert len(result) == 2
        # First step: high positive logit → score close to 1.0
        assert result[0] > 0.7
        # Second step: equal logits → score ≈ 0.5
        assert result[1] == pytest.approx(0.5, abs=0.05)

    def test_no_extra0_falls_back_to_last_token(self):
        """If no <extra_0> tokens in sequence, last-token score is returned for all steps."""
        prm = _build_prm(extra0_positions=[], pos_neg_logits=[])
        result = prm._score_steps("q", ["step 1", "step 2"])
        assert len(result) == 2
        assert all(0.0 <= s <= 1.0 for s in result)

    def test_ascore_returns_list_of_correct_length(self):
        prm = _build_prm()
        with patch.object(prm, "_score_steps", return_value=[0.7, 0.8]):
            result = asyncio.run(prm.ascore("q", ["s1", "s2"]))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_ascore_preserves_order(self):
        prm = _build_prm()
        expected = [0.1, 0.9, 0.5]
        with patch.object(prm, "_score_steps", return_value=expected):
            result = asyncio.run(prm.ascore("p", ["a", "b", "c"]))
        assert result == pytest.approx(expected)
