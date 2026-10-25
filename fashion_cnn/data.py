"""Fixed split validation and training-only augmentation.

Dataset imports are lazy: reading split metadata never downloads data.
"""

from pathlib import Path

import numpy as np

from .artifacts import file_hash


def load_split(project_dir, expected_hash=None):
    """Read the recorded 54k/6k split and reject omissions or duplicate indices."""
    path = Path(project_dir) / "results" / "split_indices.npz"
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError("Split file does not match the recorded experiment.")
    with np.load(path, allow_pickle=False) as saved:
        train, validation = saved["train_indices"], saved["val_indices"]
        seed_array = saved["seed"]
    for indices, length in ((train, 54000), (validation, 6000)):
        if indices.ndim != 1 or len(indices) != length or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Split indices must be one-dimensional integer arrays of size 54k/6k.")
    if seed_array.ndim != 0 or not np.issubdtype(seed_array.dtype, np.integer):
        raise ValueError("Split seed must be an integer scalar.")
    combined = np.concatenate((train, validation))
    if not np.array_equal(np.sort(combined), np.arange(60000)):
        raise ValueError("Every source index must occur exactly once across both splits.")
    return train, validation, int(seed_array.item())


def stratified_split(labels, val_per_class=600, seed=42):
    """Keep class counts balanced; preserve the recorded NumPy permutation order."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not labels.size or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    if type(val_per_class) is not int or val_per_class < 1:
        raise ValueError("val_per_class must be a positive integer")
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        if len(members) <= val_per_class:
            raise ValueError("Each class needs both training and validation samples.")
        shuffled = rng.permutation(members)
        val_parts.append(shuffled[:val_per_class])
        train_parts.append(shuffled[val_per_class:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def make_loaders(project_dir, seed, batch_size=64, augmentation=False,
                 train_indices=None, val_indices=None):
    """Create independent dataset views; validation always uses ToTensor only."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

    if (train_indices is None) != (val_indices is None):
        raise ValueError("Provide both training and validation indices, or neither.")
    if train_indices is None:
        train_indices, val_indices, _ = load_split(project_dir)
    train_transform = Compose([
        RandomCrop(28, padding=2), RandomHorizontalFlip(p=0.5), ToTensor(),
    ]) if augmentation else ToTensor()
    train_source = FashionMNIST(Path(project_dir) / "data", train=True,
                               download=False, transform=train_transform)
    val_source = FashionMNIST(Path(project_dir) / "data", train=True,
                             download=False, transform=ToTensor())
    train_loader = DataLoader(
        Subset(train_source, train_indices.tolist()), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_source, val_indices.tolist()), batch_size=batch_size,
        shuffle=False, generator=torch.Generator().manual_seed(seed),
        num_workers=0, drop_last=False,
    )
    return train_loader, val_loader, val_source.classes


def download_dataset(project_dir):
    """Download through torchvision; do not inspect or evaluate the test split."""
    from torchvision.datasets import FashionMNIST

    dataset = FashionMNIST(Path(project_dir) / "data", train=True, download=True)
    print(f"Fashion-MNIST ready: {len(dataset)} training-source images, {len(dataset.classes)} classes.")
    return dataset
