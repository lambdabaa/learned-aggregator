"""Custom sdg_hub blocks for the learned-aggregator project.

Importing this package registers both blocks in the sdg_hub BlockRegistry,
making them available to flows loaded via Flow.from_yaml().
"""

from .math_verify_answer import MathVerifyAnswerBlock
from .mlx_prm_score import MLXProcessRewardScoreBlock

__all__ = ["MLXProcessRewardScoreBlock", "MathVerifyAnswerBlock"]
