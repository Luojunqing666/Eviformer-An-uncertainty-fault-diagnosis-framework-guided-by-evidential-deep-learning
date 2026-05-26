"""Data augmentation transforms for 1D vibration signals."""

import numpy as np
import random
from scipy.signal import resample


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape:
    """Transpose the signal array (channels first <-> channels last)."""

    def __call__(self, seq):
        return seq.transpose()


class Retype:
    """Cast signal to float32."""

    def __call__(self, seq):
        return seq.astype(np.float32)


class Normalize:
    """
    Normalize signal values.

    Args:
        type: Normalization type. One of "0-1", "-1-1", "mean-std".
    """

    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
        elif self.type == "-1-1":
            seq = 2 * (seq - seq.min()) / (seq.max() - seq.min() + 1e-8) - 1
        elif self.type == "mean-std":
            seq = (seq - seq.mean()) / (seq.std() + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization type: {self.type}")
        return seq


class AddGaussian:
    """Add Gaussian noise to the signal."""

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian:
    """Randomly add Gaussian noise (50% probability)."""

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomScale:
    """Randomly scale the signal (50% probability)."""

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq * scale_matrix


class RandomStretch:
    """Randomly stretch/compress the signal in time (50% probability)."""

    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        seq_aug = np.zeros(seq.shape)
        length_orig = seq.shape[1]
        length_new = int(length_orig * (1 + (random.random() - 0.5) * self.sigma))
        for i in range(seq.shape[0]):
            y = resample(seq[i, :], length_new)
            if length_new < length_orig:
                if random.random() < 0.5:
                    seq_aug[i, :length_new] = y
                else:
                    seq_aug[i, length_orig - length_new:] = y
            else:
                if random.random() < 0.5:
                    seq_aug[i, :] = y[:length_orig]
                else:
                    seq_aug[i, :] = y[length_new - length_orig:]
        return seq_aug


class RandomCrop:
    """Randomly zero-out a segment of the signal (50% probability)."""

    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        max_index = seq.shape[1] - self.crop_len
        random_index = np.random.randint(max_index)
        seq[:, random_index:random_index + self.crop_len] = 0
        return seq
