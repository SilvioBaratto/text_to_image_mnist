"""Hyperparameters and configuration for CVAE training."""

import torch

# Architecture
LATENT_DIM = 20
HIDDEN_DIM = 512
INPUT_DIM = 784
LABEL_DIM = 10

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
NUM_EPOCHS = 150
KL_WEIGHT = 0.1

# Scheduling
WARMUP_EPOCHS = 10
MIN_LEARNING_RATE = 1e-6
LR_SCHEDULE = "cosine_with_warmup"
KL_WARMUP_EPOCHS = 20

# Regularization
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_VALUE = 1.0

# Paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"

# Logging
SAVE_INTERVAL = 5
TEST_INTERVAL = 10
LOG_INTERVAL = 100
EARLY_STOP_PATIENCE = 10


def get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


DEVICE = get_device()
