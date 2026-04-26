"""Tests for ProcessRewardScoreBlock and MathVerifyAnswerBlock."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# MathVerifyAnswerBlock — pure logic, no mocking needed
# ---------------------------------------------------------------------------

class TestMathVerifyAnswerBlock:
    def _make_block(self):
        from learned_aggregator.blocks.math_verify_answer import MathVerifyAnswerBlock

        return MathVerifyAnswerBlock(
            block_name="test_math_verify",
            input_cols=["trajectory_text", "ground_truth"],
            output_cols=["extracted_answer", "correct"],
        )

    def _df(self, trajectory_text: str, ground_truth: str) -> pd.DataFrame:
        return pd.DataFrame([{"trajectory_text": trajectory_text, "ground_truth": ground_truth}])

    def test_correct_boxed_answer(self):
        block = self._make_block()
        df = self._df(
            trajectory_text="Step 1: solve...\n\nTherefore, \\boxed{42}.",
            ground_truth="42",
        )
        result = block.generate(df)
        assert result["extracted_answer"].iloc[0] == "42"
        assert bool(result["correct"].iloc[0]) is True

    def test_wrong_answer(self):
        block = self._make_block()
        df = self._df(
            trajectory_text="Therefore, \\boxed{7}.",
            ground_truth="42",
        )
        result = block.generate(df)
        assert result["extracted_answer"].iloc[0] == "7"
        assert bool(result["correct"].iloc[0]) is False

    def test_no_boxed_returns_none_and_false(self):
        block = self._make_block()
        df = self._df(
            trajectory_text="I think the answer is 42.",
            ground_truth="42",
        )
        result = block.generate(df)
        assert result["extracted_answer"].iloc[0] is None
        assert bool(result["correct"].iloc[0]) is False

    def test_ground_truth_itself_boxed(self):
        """Ground truth wrapped in \\boxed{} is unwrapped before comparison."""
        block = self._make_block()
        df = self._df(
            trajectory_text="Therefore, \\boxed{5}.",
            ground_truth="\\boxed{5}",
        )
        result = block.generate(df)
        assert result["extracted_answer"].iloc[0] == "5"
        assert bool(result["correct"].iloc[0]) is True

    def test_math_verify_absent_falls_back_to_string(self):
        """When math_verify is not importable, string equality is used."""
        block = self._make_block()
        df = self._df(
            trajectory_text="Answer: \\boxed{x+1}.",
            ground_truth="x+1",
        )
        # Patch math_verify so it raises ImportError
        with patch.dict("sys.modules", {"math_verify": None}):
            result = block.generate(df)
        assert result["extracted_answer"].iloc[0] == "x+1"
        assert bool(result["correct"].iloc[0]) is True

    def test_batch_of_rows(self):
        block = self._make_block()
        df = pd.DataFrame([
            {"trajectory_text": "\\boxed{1}", "ground_truth": "1"},
            {"trajectory_text": "\\boxed{2}", "ground_truth": "9"},
            {"trajectory_text": "no answer here", "ground_truth": "3"},
        ])
        result = block.generate(df)
        assert list(result["correct"]) == [True, False, False]

    def test_output_cols_present(self):
        block = self._make_block()
        df = self._df("\\boxed{0}", "0")
        result = block.generate(df)
        assert "extracted_answer" in result.columns
        assert "correct" in result.columns

    def test_registered_in_block_registry(self):
        from sdg_hub.core.blocks.registry import BlockRegistry
        import learned_aggregator.blocks  # noqa: F401 — triggers registration

        block_class = BlockRegistry._get("MathVerifyAnswerBlock")
        from learned_aggregator.blocks.math_verify_answer import MathVerifyAnswerBlock

        assert block_class is MathVerifyAnswerBlock


# ---------------------------------------------------------------------------
# ProcessRewardScoreBlock — mock PRM to avoid hardware / model loading
# ---------------------------------------------------------------------------

class TestProcessRewardScoreBlock:
    """Tests use the default 'transformers' backend with a mocked PRM.

    The mock's score() accepts (problem, steps: list[str]) and returns
    list[float] — matching the TransformersProcessRewardModel interface.
    """

    _MOCK_TARGET = "learned_aggregator.blocks.prm_score.ProcessRewardScoreBlock._get_prm"

    def _make_block(self, step_sep: str = "\n\n"):
        from learned_aggregator.blocks.prm_score import ProcessRewardScoreBlock

        return ProcessRewardScoreBlock(
            block_name="test_prm_score",
            input_cols=["problem", "trajectory_text"],
            output_cols=["step_scores"],
            step_sep=step_sep,
        )

    def _mock_prm(self, scores: list[float]):
        """Return a mock PRM whose score(problem, steps) returns scores[:len(steps)]."""
        prm = MagicMock()
        prm.score.side_effect = lambda problem, steps: scores[: len(steps)]
        return prm

    def test_step_scores_length_matches_step_count(self):
        block = self._make_block()
        mock_prm = self._mock_prm([0.9, 0.7, 0.8])

        df = pd.DataFrame([{
            "problem": "What is 2+2?",
            "trajectory_text": "Step 1: add.\n\nStep 2: result.\n\nStep 3: done.",
        }])

        with patch(self._MOCK_TARGET, return_value=mock_prm):
            result = block.generate(df)

        step_scores = result["step_scores"].iloc[0]
        assert len(step_scores) == 3  # 3 steps separated by "\n\n"

    def test_scores_are_floats_in_unit_interval(self):
        block = self._make_block()
        mock_prm = self._mock_prm([0.5, 0.9])

        df = pd.DataFrame([{"problem": "q", "trajectory_text": "a\n\nb"}])

        with patch(self._MOCK_TARGET, return_value=mock_prm):
            result = block.generate(df)

        scores = result["step_scores"].iloc[0]
        assert all(isinstance(s, float) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_single_step_trajectory(self):
        block = self._make_block()
        mock_prm = self._mock_prm([0.75])

        df = pd.DataFrame([{"problem": "q", "trajectory_text": "single step only"}])

        with patch(self._MOCK_TARGET, return_value=mock_prm):
            result = block.generate(df)

        scores = result["step_scores"].iloc[0]
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.75)

    def test_prm_called_once_per_row(self):
        """Transformers backend calls PRM once per trajectory (not once per step)."""
        block = self._make_block()
        mock_prm = self._mock_prm([0.8, 0.6, 0.9])

        df = pd.DataFrame([{
            "problem": "p",
            "trajectory_text": "s1\n\ns2\n\ns3",
        }])

        with patch(self._MOCK_TARGET, return_value=mock_prm):
            block.generate(df)

        assert mock_prm.score.call_count == 1
        # Verify all steps are passed at once
        _, call_args, _ = mock_prm.score.mock_calls[0]
        assert call_args[1] == ["s1", "s2", "s3"]

    def test_batch_of_rows(self):
        block = self._make_block()
        mock_prm = self._mock_prm([0.5, 0.5, 0.5])

        df = pd.DataFrame([
            {"problem": "q1", "trajectory_text": "a\n\nb"},
            {"problem": "q2", "trajectory_text": "x\n\ny\n\nz"},
        ])

        with patch(self._MOCK_TARGET, return_value=mock_prm):
            result = block.generate(df)

        assert len(result["step_scores"].iloc[0]) == 2
        assert len(result["step_scores"].iloc[1]) == 3

    def test_lazy_init_does_not_load_on_import(self):
        """Importing the block should not trigger TransformersProcessRewardModel loading."""
        with patch("its_hub.integration.transformers_prm.TransformersProcessRewardModel") as mock_cls:
            from learned_aggregator.blocks import ProcessRewardScoreBlock  # noqa: F401

            mock_cls.assert_not_called()

    def test_registered_in_block_registry(self):
        from sdg_hub.core.blocks.registry import BlockRegistry
        import learned_aggregator.blocks  # noqa: F401

        block_class = BlockRegistry._get("ProcessRewardScoreBlock")
        from learned_aggregator.blocks.prm_score import ProcessRewardScoreBlock

        assert block_class is ProcessRewardScoreBlock
