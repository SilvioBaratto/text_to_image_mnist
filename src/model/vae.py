"""Conditional VAE for MNIST digit generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
sys.path.append('.')
import config


class Encoder(nn.Module):
    """Convolutional encoder: maps image+label pairs to latent distribution parameters (μ, log σ²)."""

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super().__init__()

        # Convolutional layers for spatial feature extraction
        # 28x28 -> 14x14 -> 7x7
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # 28->14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 14->7

        # After conv layers: 64 channels * 7 * 7 = 3136 features
        conv_output_dim = 64 * 7 * 7
        combined_dim = conv_output_dim + label_dim

        # Fully connected layers for latent parameters
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape flattened input to 2D spatial: (batch, 784) -> (batch, 1, 28, 28)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)

        # Convolutional feature extraction
        h = F.relu(self.conv1(x))  # (batch, 32, 14, 14)
        h = F.relu(self.conv2(h))  # (batch, 64, 7, 7)

        # Flatten conv features and concatenate with label
        h = h.view(batch_size, -1)  # (batch, 3136)
        combined = torch.cat([h, y], dim=1)  # (batch, 3136 + 10)

        # Latent parameters
        h = F.relu(self.fc1(combined))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Convolutional decoder: reconstructs images from latent codes and labels."""

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.INPUT_DIM
    ):
        super().__init__()
        combined_dim = latent_dim + label_dim

        # Fully connected layers to expand latent code
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)  # Prepare for reshape to (64, 7, 7)

        # Transposed convolutional layers for spatial reconstruction
        # 7x7 -> 14x14 -> 28x28
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # 7->14
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)   # 14->28

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Combine latent and label
        combined = torch.cat([z, y], dim=1)

        # Expand through fully connected layers
        h = F.relu(self.fc1(combined))
        h = F.relu(self.fc2(h))

        # Reshape to spatial: (batch, 3136) -> (batch, 64, 7, 7)
        batch_size = h.size(0)
        h = h.view(batch_size, 64, 7, 7)

        # Transposed convolutions for upsampling
        h = F.relu(self.deconv1(h))  # (batch, 32, 14, 14)
        h = torch.sigmoid(self.deconv2(h))  # (batch, 1, 28, 28)

        # Flatten back to match expected output: (batch, 1, 28, 28) -> (batch, 784)
        return h.view(batch_size, -1)


class CVAE(nn.Module):
    """Conditional VAE with reparameterization trick."""

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, label_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, label_dim, hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution: z = μ + σ·ε where ε ~ N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, y)
        return reconstructed, mu, logvar

    def generate(self, y: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate images by sampling from N(0,1) prior."""
        device = y.device
        batch_size = y.size(0)

        z = torch.randn(batch_size * num_samples, config.LATENT_DIM, device=device)
        y_repeated = y.repeat(num_samples, 1)

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
    """Compute VAE loss with optional KL annealing.

    Returns (total_loss, reconstruction_loss, kl_divergence).
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
