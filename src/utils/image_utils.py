"""
Image saving and visualization utilities.
Handles saving generated MNIST images and displaying them.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from typing import Optional

import sys
sys.path.append('.')
import config


def save_generated_image(
    image: torch.Tensor,
    digit: int,
    output_dir: str = config.OUTPUT_DIR,
    show: bool = False
) -> str:
    """
    Save generated MNIST image to file.

    Args:
        image: Generated image tensor of shape (784,) or (1, 28, 28)
        digit: The digit label (0-9)
        output_dir: Directory to save images
        show: Whether to display the image (default: False)

    Returns:
        filepath: Path to saved image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to numpy array and reshape to 28x28
    if image.dim() == 1:
        # Flatten format (784,)
        image_np = image.cpu().detach().numpy().reshape(28, 28)
    elif image.dim() == 3:
        # Image format (1, 28, 28)
        image_np = image.cpu().detach().numpy().squeeze()
    else:
        raise ValueError(f"Expected image shape (784,) or (1, 28, 28), got {image.shape}")

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{digit}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Create figure and save
    plt.figure(figsize=(3, 3))
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.title(f"Generated Digit: {digit}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=100)

    if show:
        plt.show()
    else:
        plt.close()

    return filepath


def visualize_training_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a grid of training/generated samples.

    Args:
        images: Batch of images with shape (batch_size, 784) or (batch_size, 1, 28, 28)
        labels: Integer labels with shape (batch_size,)
        num_samples: Number of samples to display (default: 10)
        save_path: Optional path to save the visualization
    """
    # Limit to available samples
    num_samples = min(num_samples, images.size(0))

    # Convert to numpy
    if images.dim() == 2:
        # Flatten format (batch_size, 784)
        images_np = images.cpu().detach().numpy().reshape(-1, 28, 28)
    elif images.dim() == 4:
        # Image format (batch_size, 1, 28, 28)
        images_np = images.cpu().detach().numpy().squeeze(1)
    else:
        raise ValueError(f"Expected image shape (batch, 784) or (batch, 1, 28, 28), got {images.shape}")

    labels_np = labels.cpu().detach().numpy()

    # Create grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(images_np[i], cmap='gray')
        axes[i].set_title(f"Label: {labels_np[i]}")
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()


def unflatten_image(image: torch.Tensor) -> torch.Tensor:
    """
    Convert flattened image (784,) to 2D image (1, 28, 28).

    Args:
        image: Flattened image tensor of shape (784,) or (batch, 784)

    Returns:
        unflattened: Image tensor of shape (1, 28, 28) or (batch, 1, 28, 28)
    """
    if image.dim() == 1:
        return image.view(1, 28, 28)
    elif image.dim() == 2:
        return image.view(-1, 1, 28, 28)
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {image.shape}")
