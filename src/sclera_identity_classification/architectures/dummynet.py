import torch
from torch import nn


class DummyNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # expects input shaped [B, C, 2, 2]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 2 * 2, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
