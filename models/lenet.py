"""LeNet-based 1D CNN for vibration signal classification."""

import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    1D LeNet-style CNN for vibration signal fault diagnosis.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (default: 1).
        dropout: Whether to apply dropout before the final FC layer.
    """

    def __init__(self, num_classes=10, in_channels=1, dropout=False):
        super().__init__()
        self.use_dropout = dropout

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return x
