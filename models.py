"""Neural architectures for firing-pattern classification."""

from __future__ import annotations

import torch
from torch import nn


class NeuronClassifier1D(nn.Module):
    """Compact 1D CNN for RS/CH/FS voltage-trace classification.

    Adaptive pooling removes the million-parameter flattening bottleneck from
    the original implementation and keeps the model usable with other window
    lengths.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, voltage_windows: torch.Tensor) -> torch.Tensor:
        if voltage_windows.ndim != 3 or voltage_windows.shape[1] != 1:
            raise ValueError("Expected input shape (batch, 1, time_steps).")
        return self.classifier(self.features(voltage_windows))

