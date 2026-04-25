import torch
import torch.nn as nn


class TrajectoryMLP(nn.Module):
    """2-layer MLP mapping a 10-dim feature vector to a scalar trajectory score.

    Input:  10-dim feature vector from extract_features()
    Output: scalar in (0, 1) via sigmoid, interpretable as P(trajectory correct)

    Checkpoint format (compatible with LearnedMLPAggregator in its_hub):
        torch.save({"state_dict": model.state_dict(), "hidden_width": hidden_width}, path)
    """

    def __init__(self, input_dim: int = 10, hidden_width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
