"""Image saving and terminal visualization utilities."""

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
    """Save generated digit image to file."""
    os.makedirs(output_dir, exist_ok=True)

    if image.dim() == 1:
        image_np = image.cpu().detach().numpy().reshape(28, 28)
    elif image.dim() == 3:
        image_np = image.cpu().detach().numpy().squeeze()
    else:
        raise ValueError(f"Expected shape (784,) or (1, 28, 28), got {image.shape}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"generated_{digit}_{timestamp}.png")

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
    """Create a grid of sample images with labels."""
    num_samples = min(num_samples, images.size(0))

    if images.dim() == 2:
        images_np = images.cpu().detach().numpy().reshape(-1, 28, 28)
    elif images.dim() == 4:
        images_np = images.cpu().detach().numpy().squeeze(1)
    else:
        raise ValueError(f"Expected shape (batch, 784) or (batch, 1, 28, 28), got {images.shape}")

    labels_np = labels.cpu().detach().numpy()

    _, axes = plt.subplots(2, 5, figsize=(10, 4))
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
    """Convert (784,) to (1, 28, 28) or (batch, 784) to (batch, 1, 28, 28)."""
    if image.dim() == 1:
        return image.view(1, 28, 28)
    elif image.dim() == 2:
        return image.view(-1, 1, 28, 28)
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {image.shape}")


def display_image_in_terminal(image: torch.Tensor, digit: Optional[int] = None) -> None:
    """Render generated digit as Unicode block art in terminal."""
    if image.dim() == 1:
        image_np = image.cpu().detach().numpy().reshape(28, 28)
    elif image.dim() == 3:
        image_np = image.cpu().detach().numpy().squeeze()
    else:
        raise ValueError(f"Expected shape (784,) or (1, 28, 28), got {image.shape}")

    block_chars = " ░░▒▒▓▓██"
    image_np = np.clip(image_np, 0, 1)
    image_np = np.power(image_np, 0.7)

    print(f"\n{'='*80}")
    if digit is not None:
        print(f"{'Generated Digit: ' + str(digit):^80}")
    else:
        print(f"{'Generated Image':^80}")
    print(f"{'='*80}\n")

    for i in range(0, 28, 2):
        line = "    "
        for j in range(28):
            pixel = (image_np[i, j] + image_np[i+1, j]) / 2 if i + 1 < 28 else image_np[i, j]
            char_idx = min(int(pixel * len(block_chars)), len(block_chars) - 1)
            line += block_chars[char_idx] * 2
        print(line)

    print(f"\n{'='*80}\n")
