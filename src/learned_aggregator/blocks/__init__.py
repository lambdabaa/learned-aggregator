"""Custom sdg_hub blocks for the learned-aggregator project.

Importing this package registers all blocks in the sdg_hub BlockRegistry,
making them available to flows loaded via Flow.from_yaml().
"""

from .math_verify_answer import MathVerifyAnswerBlock
from .mlx_chat_block import MLXChatBlock
from .prm_score import ProcessRewardScoreBlock
from .trajectory_nonce import TrajectoryNonceBlock

__all__ = ["ProcessRewardScoreBlock", "MathVerifyAnswerBlock", "TrajectoryNonceBlock", "MLXChatBlock"]
