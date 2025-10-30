"""Hyperparameters and configuration for CVAE training."""

import torch

# Architecture
LATENT_DIM = 20
HIDDEN_DIM = 512
INPUT_DIM = 784
TEXT_EMBEDDING_DIM = 384  # SentenceTransformer all-MiniLM-L6-v2 output dimension
LABEL_DIM = TEXT_EMBEDDING_DIM  # For backward compatibility

# Training (adjusted for 384-dim text embeddings)
BATCH_SIZE = 32  # Reduced: text encoding + larger model = more memory
LEARNING_RATE = 0.0001  # Lower: more parameters need gentler updates
NUM_EPOCHS = 200  # Increased: semantic learning takes longer to converge
KL_WEIGHT = 0.1

# Scheduling
WARMUP_EPOCHS = 15  # Longer warmup for stable text embedding learning
MIN_LEARNING_RATE = 1e-6
LR_SCHEDULE = "cosine_with_warmup"
KL_WARMUP_EPOCHS = 30  # Longer KL warmup: larger embedding space needs time

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
