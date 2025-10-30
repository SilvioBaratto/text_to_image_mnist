"""Conditional VAE for MNIST digit generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
sys.path.append('.')
import config


class Encoder(nn.Module):
    """Maps image+label pairs to latent distribution parameters (μ, log σ²)."""

    def __init__(
        self,
        input_dim: int = config.INPUT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        latent_dim: int = config.LATENT_DIM
    ):
        super().__init__()
        combined_dim = input_dim + label_dim

        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(combined))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Reconstructs images from latent codes and labels."""

    def __init__(
        self,
        latent_dim: int = config.LATENT_DIM,
        label_dim: int = config.LABEL_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        output_dim: int = config.INPUT_DIM
    ):
        super().__init__()
        combined_dim = latent_dim + label_dim

        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, y], dim=1)
        h = F.relu(self.fc1(combined))
        return torch.sigmoid(self.fc2(h))


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
