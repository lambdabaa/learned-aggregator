import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TrajectoryLSTM(nn.Module):
    """1-layer LSTM operating on raw step score sequences.

    Unlike TrajectoryMLP, this model receives the raw sequence [s1, ..., sN]
    rather than a hand-crafted feature vector, allowing it to capture
    sequential patterns (e.g. mid-trajectory dips that recover) directly.

    Checkpoint format:
        torch.save({"state_dict": model.state_dict(), "hidden_size": hidden_size}, path)
    """

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (batch, max_len, 1), lengths: (batch,) cpu int64
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return torch.sigmoid(self.head(h_n[-1])).squeeze(-1)


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
