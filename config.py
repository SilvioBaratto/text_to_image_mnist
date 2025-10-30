"""
Configuration file for Text-to-Image MNIST Generator.
Contains hyperparameters and model configuration.
"""

import torch

# Model architecture
LATENT_DIM = 20  # Latent space dimension (10-20 range sufficient for MNIST)
HIDDEN_DIM = 512  # Hidden layer dimension
INPUT_DIM = 784  # 28x28 flattened MNIST images
LABEL_DIM = 10  # Number of digit classes (0-9)

# Training hyperparameters - Optimized for quality on Apple Silicon
BATCH_SIZE = 64  # Smaller batch size for better generalization
LEARNING_RATE = 0.0003  # Lower learning rate for stable convergence
NUM_EPOCHS = 150  # More epochs for thorough training
KL_WEIGHT = 0.1  # Weight for KL divergence term (PyTorch version only)

# Learning rate scheduling
WARMUP_EPOCHS = 10  # Linear warmup period (epochs)
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate for cosine decay
LR_SCHEDULE = "cosine_with_warmup"  # Options: "cosine_with_warmup", "step", "none"

# KL annealing (for Pyro SVI training)
KL_WARMUP_EPOCHS = 20  # KL annealing period (0 to 1.0)

# Regularization
WEIGHT_DECAY = 1e-5  # L2 regularization
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping threshold

# Paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

# Training settings
SAVE_INTERVAL = 5  # Save checkpoint every N epochs
TEST_INTERVAL = 10  # Evaluate on test set every N epochs
LOG_INTERVAL = 100  # Log training stats every N batches
EARLY_STOP_PATIENCE = 10  # Early stopping patience (epochs)


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU

    Returns:
        device: torch.device object for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU (GPU not available)")

    return device


# Device configuration - automatically detects MPS for Apple Silicon Macs
DEVICE = get_device()
