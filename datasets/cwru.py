"""
CWRU Bearing Dataset loader.

Case Western Reserve University Bearing Data Center:
https://engineering.case.edu/bearingdatacenter
"""

import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .sequence_dataset import SequenceDataset
from .transforms import Compose, Reshape, Normalize, Retype


# Signal length per sample
SIGNAL_SIZE = 1024

# Dataset directory names
DATASET_NAMES = [
    "12k Drive End Bearing Fault Data",
    "12k Fan End Bearing Fault Data",
    "48k Drive End Bearing Fault Data",
    "Normal Baseline Data",
]

# Normal baseline data files
NORMAL_FILES = ["97.mat", "98.mat", "99.mat", "100.mat"]

# 12k Drive End Bearing Fault Data (different RPMs)
FAULT_FILES_12K_DE = {
    1797: ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat", "234.mat"],
    1772: ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat", "235.mat"],
    1750: ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat", "236.mat"],
    1730: ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat", "237.mat"],
}

# Axis suffix for data extraction
AXIS_SUFFIXES = ["_DE_time", "_FE_time", "_BA_time"]

# Fault labels (1-9 for different fault types)
FAULT_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def data_load(filename, axisname, label, signal_size=SIGNAL_SIZE, stride=None):
    """
    Load and segment vibration data from a .mat file.

    Args:
        filename: Path to the .mat file.
        axisname: Filename used to determine the data key.
        label: Integer label for this data.
        signal_size: Length of each signal segment.
        stride: Stride between segments (default: same as signal_size).

    Returns:
        Tuple of (data_list, label_list).
    """
    if stride is None:
        stride = signal_size

    datanumber = axisname.split(".")[0]
    if int(datanumber) < 100:
        realaxis = "X0" + datanumber + AXIS_SUFFIXES[0]
    else:
        realaxis = "X" + datanumber + AXIS_SUFFIXES[0]

    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += stride
        end += stride
    return data, lab


def get_files(root, rpm=1797, signal_size=SIGNAL_SIZE, stride=None):
    """
    Load all data files (normal + fault) for a given RPM.

    Args:
        root: Root directory of the CWRU dataset.
        rpm: Motor speed in RPM (1797, 1772, 1750, or 1730).
        signal_size: Length of each signal segment.
        stride: Stride between segments.

    Returns:
        List of [data_list, label_list].
    """
    fault_files = FAULT_FILES_12K_DE[rpm]
    data_root = os.path.join(root, DATASET_NAMES[0])

    # Load normal data (label=0)
    normal_idx = {1797: 0, 1772: 1, 1750: 2, 1730: 3}[rpm]
    normal_path = os.path.join(data_root, NORMAL_FILES[normal_idx])
    data, lab = data_load(normal_path, NORMAL_FILES[normal_idx], label=0,
                          signal_size=signal_size, stride=stride)

    # Load fault data (labels 1-9)
    for i in tqdm(range(len(fault_files)), desc="Loading fault data"):
        path = os.path.join(data_root, fault_files[i])
        data_i, lab_i = data_load(path, fault_files[i], label=FAULT_LABELS[i],
                                  signal_size=signal_size, stride=stride)
        data += data_i
        lab += lab_i

    return [data, lab]


def get_data_transforms(dataset_type="train", normalize_type="-1-1"):
    """
    Get data transform pipeline.

    Args:
        dataset_type: "train" or "val".
        normalize_type: Normalization type ("0-1", "-1-1", or "mean-std").

    Returns:
        Compose transform pipeline.
    """
    return Compose([Reshape(), Normalize(normalize_type), Retype()])


def load_cwru_dataset(data_dir, normalize_type="0-1", test_size=0.2,
                      random_state=40, batch_size=16, num_workers=0,
                      rpm=1797, signal_size=SIGNAL_SIZE, stride=None):
    """
    Load CWRU dataset and create DataLoaders.

    Args:
        data_dir: Root directory of the CWRU dataset.
        normalize_type: Normalization type for transforms.
        test_size: Fraction of data for validation.
        random_state: Random seed for train/val split.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.
        rpm: Motor speed in RPM.
        signal_size: Length of each signal segment.
        stride: Stride between segments.

    Returns:
        Dictionary with 'train' and 'val' DataLoaders.
    """
    list_data = get_files(data_dir, rpm=rpm, signal_size=signal_size, stride=stride)
    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})

    train_pd, val_pd = train_test_split(
        data_pd, test_size=test_size, random_state=random_state,
        stratify=data_pd["label"]
    )

    train_dataset = SequenceDataset(
        list_data=train_pd, transform=get_data_transforms('train', normalize_type)
    )
    val_dataset = SequenceDataset(
        list_data=val_pd, transform=get_data_transforms('val', normalize_type)
    )

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers),
    }
    return dataloaders
