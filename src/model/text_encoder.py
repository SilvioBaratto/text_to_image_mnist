"""Semantic text encoder using SentenceTransformer for natural language understanding."""

import torch
from sentence_transformers import SentenceTransformer

# Global model instance (loaded once)
_text_encoder_model = None


def get_text_encoder_model():
    """Load and cache the SentenceTransformer model."""
    global _text_encoder_model
    if _text_encoder_model is None:
        print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
        _text_encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Text encoder loaded successfully!")
    return _text_encoder_model


def encode_text_to_embedding(text: str, device: str = 'cpu') -> torch.Tensor:
    """Encode text prompt to 384-dimensional semantic embedding.

    Args:
        text: Natural language prompt (e.g., "I want to draw a zero")
        device: Device to place tensor on ('cpu', 'cuda', 'mps')

    Returns:
        torch.Tensor: 384-dimensional embedding vector
    """
    model = get_text_encoder_model()

    # Encode text (returns numpy array)
    embedding = model.encode(text, convert_to_tensor=False)

    # Convert to torch tensor and move to device
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    return embedding_tensor


def encode_texts_batch(texts: list[str], device: str = 'cpu') -> torch.Tensor:
    """Encode a batch of text prompts to embeddings.

    Args:
        texts: List of text prompts
        device: Device to place tensor on

    Returns:
        torch.Tensor: (batch_size, 384) embedding matrix
    """
    model = get_text_encoder_model()

    # Batch encode (more efficient)
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

    # Convert to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    return embeddings_tensor
