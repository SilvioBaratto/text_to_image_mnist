"""
MNIST data loading utilities.
Provides functions to load and prepare MNIST dataset for training.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple

import sys
sys.path.append('.')
import config


def get_mnist_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset and create train/validation/test data loaders.

    Args:
        batch_size: Batch size for training (default: 128)
        data_dir: Directory to store/load MNIST data
        val_split: Fraction of training data to use for validation (default: 0.2)

    Returns:
        train_loader: DataLoader for training set (48,000 samples with default split)
        val_loader: DataLoader for validation set (12,000 samples with default split)
        test_loader: DataLoader for test set (10,000 samples)
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Define transform to convert images to tensors
    # MNIST images are normalized to [0, 1] range
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load MNIST training dataset (60,000 samples)
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Load MNIST test dataset (10,000 samples)
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # Split training dataset into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Use fixed seed for reproducibility
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    """
    Flatten batch of MNIST images from (batch, 1, 28, 28) to (batch, 784).

    Args:
        images: Batch of images with shape (batch_size, 1, 28, 28)

    Returns:
        flattened: Flattened images with shape (batch_size, 784)
    """
    batch_size = images.size(0)
    flattened = images.view(batch_size, -1)
    return flattened


def labels_to_onehot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Convert batch of integer labels to one-hot encoding.

    Args:
        labels: Integer labels with shape (batch_size,)
        num_classes: Number of classes (default: 10 for MNIST)

    Returns:
        onehot: One-hot encoded labels with shape (batch_size, num_classes)
    """
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot
