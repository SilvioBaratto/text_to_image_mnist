"""
Comprehensive training logger for Pyro CVAE.
Logs shapes, values, images, and training metrics.
"""

import os
import logging
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('.')
import config


class TrainingLogger:
    """
    Comprehensive logger for tracking training progress, tensor shapes,
    and visualizations.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = f"pyro_cvae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_dir = log_dir

        # Setup file logger
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # CSV file for metrics
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.csv")
        self._init_metrics_csv()

        # Training state
        self.epoch_metrics = []

        self.logger.info("=" * 80)
        self.logger.info(f"Training Logger Initialized: {experiment_name}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Metrics file: {self.metrics_file}")
        self.logger.info("=" * 80)

    def _init_metrics_csv(self):
        """Initialize CSV file for metrics."""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'kl_weight',
                'train_loss',
                'test_loss',
                'train_batches',
                'test_batches',
                'timestamp'
            ])

    def log_tensor_info(
        self,
        name: str,
        tensor: torch.Tensor,
        log_values: bool = False
    ):
        """
        Log detailed tensor information.

        Args:
            name: Tensor name
            tensor: Tensor to log
            log_values: Whether to log value statistics
        """
        info = [
            f"{name}:",
            f"shape={tuple(tensor.shape)}",
            f"dtype={tensor.dtype}",
            f"device={tensor.device}"
        ]

        if log_values:
            info.extend([
                f"min={tensor.min().item():.4f}",
                f"max={tensor.max().item():.4f}",
                f"mean={tensor.mean().item():.4f}",
                f"std={tensor.std().item():.4f}"
            ])

        self.logger.info(" | ".join(info))

    def log_batch_info(
        self,
        epoch: int,
        batch_idx: int,
        images: torch.Tensor,
        labels: torch.Tensor,
        images_flat: torch.Tensor,
        labels_onehot: torch.Tensor
    ):
        """
        Log batch information including shapes and sample data.

        Args:
            epoch: Current epoch
            batch_idx: Batch index
            images: Original images (batch, 1, 28, 28)
            labels: Integer labels (batch,)
            images_flat: Flattened images (batch, 784)
            labels_onehot: One-hot labels (batch, 10)
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"BATCH INFO - Epoch {epoch}, Batch {batch_idx}")
        self.logger.info(f"{'='*80}")

        # Log shapes
        self.log_tensor_info("images (original)", images, log_values=True)
        self.log_tensor_info("labels (integer)", labels, log_values=False)
        self.log_tensor_info("images_flat", images_flat, log_values=True)
        self.log_tensor_info("labels_onehot", labels_onehot, log_values=False)

        # Log sample labels with text
        sample_size = min(5, labels.size(0))
        label_text = ", ".join([
            f"{labels[i].item()}" for i in range(sample_size)
        ])
        self.logger.info(f"Sample labels: [{label_text}]")

        # Verify one-hot encoding
        self.logger.info("\nOne-hot verification:")
        for i in range(min(3, labels.size(0))):
            digit = int(labels[i].item())
            onehot_vec = labels_onehot[i].cpu().numpy()
            expected_idx = int(np.argmax(onehot_vec))
            is_correct = (digit == expected_idx) and (onehot_vec[digit] == 1.0)
            self.logger.info(
                f"  Label {i}: digit={digit}, "
                f"onehot_argmax={expected_idx}, "
                f"correct={is_correct}"
            )

        # Verify flattening
        self.logger.info("\nFlattening verification:")
        original_pixels = images.shape[2] * images.shape[3]
        flat_pixels = images_flat.shape[1]
        self.logger.info(f"  Original: {images.shape[2]}×{images.shape[3]} = {original_pixels}")
        self.logger.info(f"  Flattened: {flat_pixels}")
        self.logger.info(f"  Match: {original_pixels == flat_pixels}")

        self.logger.info(f"{'='*80}\n")

    def log_model_output(
        self,
        epoch: int,
        batch_idx: int,
        reconstructed: torch.Tensor,
        z_loc: torch.Tensor,
        z_scale: torch.Tensor,
        loss: float
    ):
        """
        Log model output information.

        Args:
            epoch: Current epoch
            batch_idx: Batch index
            reconstructed: Reconstructed images
            z_loc: Latent mean
            z_scale: Latent scale
            loss: Batch loss
        """
        self.logger.info(f"\nMODEL OUTPUT - Epoch {epoch}, Batch {batch_idx}")
        self.log_tensor_info("reconstructed", reconstructed, log_values=True)
        self.log_tensor_info("z_loc (latent mean)", z_loc, log_values=True)
        self.log_tensor_info("z_scale (latent std)", z_scale, log_values=True)
        self.logger.info(f"Loss: {loss:.2f}\n")

    def log_epoch_start(self, epoch: int, total_epochs: int, kl_weight: float):
        """Log epoch start."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"EPOCH {epoch}/{total_epochs} - KL Weight: {kl_weight:.2f}")
        self.logger.info("=" * 80)

    def log_epoch_end(
        self,
        epoch: int,
        kl_weight: float,
        train_loss: float,
        test_loss: float,
        train_batches: int,
        test_batches: int
    ):
        """
        Log epoch end and save metrics.

        Args:
            epoch: Epoch number
            kl_weight: KL annealing weight
            train_loss: Training loss
            test_loss: Test loss
            train_batches: Number of training batches
            test_batches: Number of test batches
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"EPOCH {epoch} SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"KL Weight:    {kl_weight:.3f}")
        self.logger.info(f"Train Loss:   {train_loss:.2f}")
        self.logger.info(f"Test Loss:    {test_loss:.2f}")
        self.logger.info(f"Loss Diff:    {abs(train_loss - test_loss):.2f}")
        self.logger.info(f"Train Batches: {train_batches}")
        self.logger.info(f"Test Batches:  {test_batches}")
        self.logger.info("=" * 80 + "\n")

        # Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                kl_weight,
                train_loss,
                test_loss,
                train_batches,
                test_batches,
                timestamp
            ])

        # Store in memory
        self.epoch_metrics.append({
            'epoch': epoch,
            'kl_weight': kl_weight,
            'train_loss': train_loss,
            'test_loss': test_loss
        })

    def log_generated_samples(
        self,
        epoch: int,
        generated_images: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Log information about generated samples.

        Args:
            epoch: Epoch number
            generated_images: Generated images (num_digits, 784) or (num_digits, 1, 28, 28)
            labels: Labels for generated images
        """
        self.logger.info(f"\nGENERATED SAMPLES - Epoch {epoch}")
        self.log_tensor_info("generated_images", generated_images, log_values=True)
        self.log_tensor_info("labels", labels, log_values=False)

        # Log which digits were generated
        digits = labels.cpu().numpy()
        self.logger.info(f"Generated digits: {list(digits)}")

        # Check value range
        img_min = generated_images.min().item()
        img_max = generated_images.max().item()
        in_range = (img_min >= 0.0) and (img_max <= 1.0)
        self.logger.info(f"Pixel value range: [{img_min:.4f}, {img_max:.4f}] - Valid: {in_range}\n")

    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training and test loss curves.

        Args:
            save_path: Path to save plot (default: logs/experiment_name_curves.png)
        """
        if not self.epoch_metrics:
            self.logger.warning("No metrics to plot yet")
            return

        if save_path is None:
            save_path = os.path.join(self.log_dir, f"{self.experiment_name}_curves.png")

        epochs = [m['epoch'] for m in self.epoch_metrics]
        train_losses = [m['train_loss'] for m in self.epoch_metrics]
        test_losses = [m['test_loss'] for m in self.epoch_metrics]
        kl_weights = [m['kl_weight'] for m in self.epoch_metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, test_losses, label='Test Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (-ELBO)')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # KL annealing schedule
        ax2.plot(epochs, kl_weights, label='KL Weight', marker='o', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Weight')
        ax2.set_title('KL Annealing Schedule')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Training curves saved: {save_path}")

    def log_dataset_info(
        self,
        train_size: int,
        test_size: int,
        batch_size: int,
        num_train_batches: int,
        num_test_batches: int
    ):
        """Log dataset information."""
        self.logger.info("\nDATASET INFO")
        self.logger.info("=" * 80)
        self.logger.info(f"Training samples:   {train_size:,}")
        self.logger.info(f"Test samples:       {test_size:,}")
        self.logger.info(f"Batch size:         {batch_size}")
        self.logger.info(f"Training batches:   {num_train_batches}")
        self.logger.info(f"Test batches:       {num_test_batches}")
        self.logger.info("=" * 80 + "\n")

    def log_model_architecture(self, cvae):
        """Log model architecture details."""
        self.logger.info("\nMODEL ARCHITECTURE")
        self.logger.info("=" * 80)
        self.logger.info(f"Input dim:  {cvae.input_dim}")
        self.logger.info(f"Label dim:  {cvae.label_dim}")
        self.logger.info(f"Latent dim: {cvae.latent_dim}")

        # Count parameters
        encoder_params = sum(p.numel() for p in cvae.encoder.parameters())
        decoder_params = sum(p.numel() for p in cvae.decoder.parameters())
        total_params = encoder_params + decoder_params

        self.logger.info(f"\nEncoder parameters: {encoder_params:,}")
        self.logger.info(f"Decoder parameters: {decoder_params:,}")
        self.logger.info(f"Total parameters:   {total_params:,}")
        self.logger.info("=" * 80 + "\n")

    def log_training_config(self, config_dict: Dict[str, Any]):
        """Log training configuration."""
        self.logger.info("\nTRAINING CONFIG")
        self.logger.info("=" * 80)
        for key, value in config_dict.items():
            self.logger.info(f"{key:20s}: {value}")
        self.logger.info("=" * 80 + "\n")
