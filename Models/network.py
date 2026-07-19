"""
MegaBonk AI — Custom CNN Feature Extractor (Optional)
======================================================
Extended NatureCNN with adjustable architecture.
SB3's built-in CnnPolicy already uses NatureCNN by default —
this module is only needed if you want to customize the architecture.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MegaBonkCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for MegaBonk observations.

    Architecture (based on NatureCNN with modifications):
        Conv2d(stack_size, 32, 8, stride=4) → ReLU
        Conv2d(32, 64, 4, stride=2) → ReLU
        Conv2d(64, 64, 3, stride=1) → ReLU
        Flatten → Linear(64 * 7 * 7, features_dim)

    This is designed for 84×84 grayscale stacked inputs.
    For 84×84 input, the conv layers produce 7×7×64 = 3136 features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
    ) -> None:
        """
        Args:
            observation_space: The observation space (should be Box with shape (C, 84, 84)).
            features_dim: Output feature dimension for the policy/value heads.
        """
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # stack_size (e.g., 4)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size by doing a forward pass with dummy data
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape, dtype=torch.float32)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: images → feature vector.

        Args:
            observations: Batch of stacked frames, shape (B, C, H, W).

        Returns:
            Feature tensor, shape (B, features_dim).
        """
        # SB3 automatically normalizes uint8 images to [0, 1]
        return self.linear(self.cnn(observations))
