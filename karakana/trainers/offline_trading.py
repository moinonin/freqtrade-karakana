"""Lightweight inference-only offline trading Q-policy contract for runtime consumers (e.g. Freqtrade)."""

from __future__ import annotations

import torch
import torch.nn as nn

ACTION_LABELS = ("go_long", "go_short", "do_nothing")
ACTION_TO_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}


class OfflineTradingQPolicy(nn.Module):
    """Q-network used for discrete offline trading decisions."""

    def __init__(self, observation_size: int = 4, action_count: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
