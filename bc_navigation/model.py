"""Inference copy of the BC model architecture used by the checkpoint."""

import torch
from torch import nn


class BCPolicy(nn.Module):
    """10 fixed-scale inputs -> 64 -> 64 -> bounded turn/speed outputs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        raw = self.net(x)
        return torch.cat((torch.tanh(raw[:, :1]), torch.sigmoid(raw[:, 1:])), dim=1)
