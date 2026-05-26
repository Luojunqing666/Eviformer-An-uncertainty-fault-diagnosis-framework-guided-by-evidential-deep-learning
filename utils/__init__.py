"""Utility functions for Eviformer."""

import torch


def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def one_hot_embedding(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoding.

    Args:
        labels: Integer tensor of class indices.
        num_classes: Total number of classes.

    Returns:
        One-hot encoded tensor on the same device as labels.
    """
    device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cpu")
    y = torch.eye(num_classes, device=device)
    return y[labels]
