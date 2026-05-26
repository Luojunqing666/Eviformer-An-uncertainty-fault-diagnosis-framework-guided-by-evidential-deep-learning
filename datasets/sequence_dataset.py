"""PyTorch Dataset class for 1D vibration signal sequences."""

import torch
from torch.utils.data import Dataset
import numpy as np
from .transforms import Compose, Reshape


class SequenceDataset(Dataset):
    """
    Dataset for 1D vibration signal sequences.

    Args:
        list_data: DataFrame or dict with 'data' and 'label' columns.
        test: Whether this is a test dataset (no labels).
        transform: Data transform pipeline.
    """

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()

        if transform is None:
            self.transforms = Compose([Reshape()])
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        seq = self.transforms(seq)
        if self.test:
            return seq, item
        else:
            label = self.labels[item]
            return seq, label
