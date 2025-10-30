"""
Conditional Variational Autoencoder (CVAE) implementation for MNIST.
Architecture based on https://pyro.ai/examples/cvae.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
sys.path.append('.')
import config


class Encoder(nn.Module):
    """
    Encoder network: Image + label → latent distribution parameters.

    Input: 784 (flattened image) + 10 (one-hot label) = 794 dimensions
    Output: mean (μ) and log_variance (log σ²) for latent space
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super(Encoder, self).__init__()

        # Concatenate image and label
        combined_dim = input_dim + label_dim

        # Hidden layer
        self.fc1 = nn.Linear(combined_dim, hidden_dim)

        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Flattened image tensor of shape (batch_size, 784)
            y: One-hot encoded label tensor of shape (batch_size, 10)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Concatenate image and label
        combined = torch.cat([x, y], dim=1)

        # Hidden layer with ReLU activation
        h = F.relu(self.fc1(combined))

        # Output mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network: Latent + label → reconstructed image.

    Input: 20 (latent) + 10 (one-hot label) = 30 dimensions
    Output: 784 dimensions (reconstructed image)
    """

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.INPUT_DIM
    ):
        super(Decoder, self).__init__()

        # Concatenate latent and label
        combined_dim = latent_dim + label_dim

        # Hidden layer
        self.fc1 = nn.Linear(combined_dim, hidden_dim)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            y: One-hot encoded label tensor of shape (batch_size, 10)

        Returns:
            reconstructed: Reconstructed image of shape (batch_size, 784)
        """
        # Concatenate latent and label
        combined = torch.cat([z, y], dim=1)

        # Hidden layer with ReLU activation
        h = F.relu(self.fc1(combined))

        # Output layer with sigmoid activation
        reconstructed = torch.sigmoid(self.fc2(h))

        return reconstructed


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder combining encoder and decoder.
    Implements reparameterization trick and loss computation.
    """

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super(CVAE, self).__init__()

        self.encoder = Encoder(input_dim, label_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, label_dim, hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ × ε where ε ~ N(0,1)

        Args:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)

        Returns:
            z: Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log σ²)
        eps = torch.randn_like(std)     # ε ~ N(0,1)
        z = mu + std * eps              # z = μ + σ × ε
        return z

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through CVAE.

        Args:
            x: Flattened image tensor of shape (batch_size, 784)
            y: One-hot encoded label tensor of shape (batch_size, 10)

        Returns:
            reconstructed: Reconstructed image (batch_size, 784)
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Encode to get latent distribution parameters
        mu, logvar = self.encoder(x, y)

        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode to reconstruct image
        reconstructed = self.decoder(z, y)

        return reconstructed, mu, logvar

    def generate(self, y: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate image from label by sampling from prior N(0,1).

        Args:
            y: One-hot encoded label tensor of shape (batch_size, 10)
            num_samples: Number of samples to generate per label

        Returns:
            generated: Generated image of shape (batch_size * num_samples, 784)
        """
        # Get device from input tensor
        device = y.device
        batch_size = y.size(0)
        latent_dim = config.LATENT_DIM

        # Sample from standard normal prior
        z = torch.randn(batch_size * num_samples, latent_dim, device=device)

        # Repeat labels for multiple samples
        y_repeated = y.repeat(num_samples, 1)

        # Decode to generate image
        with torch.no_grad():
            generated = self.decoder(z, y_repeated)

        return generated


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: Reconstruction loss + beta * KL divergence.

    Args:
        recon_x: Reconstructed image (batch_size, 784)
        x: Original image (batch_size, 784)
        mu: Mean of latent distribution (batch_size, latent_dim)
        logvar: Log variance of latent distribution (batch_size, latent_dim)
        beta: Weight for KL divergence term (KL annealing factor)

    Returns:
        total_loss: Total loss (scalar)
        recon_loss: Reconstruction loss (scalar)
        kl_loss: KL divergence (scalar)
    """
    # Reconstruction loss (Binary Cross Entropy)
    # Use sum reduction as per CLAUDE.md guidelines
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence: -0.5 × Σ(1 + log(σ²) - μ² - σ²)
    # Use sum reduction as per CLAUDE.md guidelines
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss with beta weighting for KL annealing
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
