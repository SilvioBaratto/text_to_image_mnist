"""MNIST data loading and preprocessing."""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple

import sys
sys.path.append('.')
import config
from src.utils.prompt_generator import generate_prompts_batch
from src.model.text_encoder import encode_texts_batch


def get_mnist_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    data_dir: str = config.DATA_DIR,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST with train/val/test split."""
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True  # 0 workers to avoid fork issues with text encoder
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def flatten_images(images: torch.Tensor) -> torch.Tensor:
    """Flatten (batch, 1, 28, 28) to (batch, 784)."""
    return images.view(images.size(0), -1)


def labels_to_onehot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Convert integer labels to one-hot encoding."""
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot


def labels_to_text_embeddings(labels: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """Convert integer labels to text embeddings via natural language prompts.

    Args:
        labels: (batch_size,) tensor of integer labels (0-9)
        device: Device to place embeddings on

    Returns:
        torch.Tensor: (batch_size, 384) text embeddings
    """
    # Convert labels to list of integers
    labels_list = labels.cpu().tolist()

    # Generate natural language prompts
    prompts = generate_prompts_batch(labels_list)

    # Encode prompts to embeddings
    embeddings = encode_texts_batch(prompts, device=device)

    return embeddings
