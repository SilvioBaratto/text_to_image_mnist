"""Training script for CVAE with KL annealing."""

import os
import csv
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

# Fix tokenizer multiprocessing warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import config
from src.model.vae import CVAE, loss_function
from src.utils.data_loader import get_mnist_dataloaders, flatten_images, labels_to_text_embeddings
from src.utils.image_utils import visualize_training_samples


def get_kl_annealing_factor(epoch: int, warmup_epochs: int = config.KL_WARMUP_EPOCHS) -> float:
    """Linear KL annealing from 0 to 1.0."""
    return 1.0 if epoch >= warmup_epochs else epoch / warmup_epochs


def train_epoch(model, train_loader, optimizer, device, epoch, beta):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        x = flatten_images(images)
        y = labels_to_text_embeddings(labels, device=device)

        recon, mu, logvar = model(x, y)
        loss, recon_loss, kl_loss = loss_function(recon, x, mu, logvar, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item() / x.size(0):.2f}',
            'beta': f'{beta:.2f}'
        })

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon / len(train_loader.dataset)
    avg_kl = total_kl / len(train_loader.dataset)

    return avg_loss, avg_recon, avg_kl


def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            x = flatten_images(images)
            y = labels_to_text_embeddings(labels, device=device)

            recon, mu, logvar = model(x, y)
            loss, recon_loss, kl_loss = loss_function(recon, x, mu, logvar, beta=1.0)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    avg_loss = total_loss / len(test_loader.dataset)
    avg_recon = total_recon / len(test_loader.dataset)
    avg_kl = total_kl / len(test_loader.dataset)

    return avg_loss, avg_recon, avg_kl


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filepath)
    print(f"Saved checkpoint: {filepath}")


def visualize_samples(model, device, epoch, output_dir="outputs"):
    """Generate and save sample images for all digits."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    labels = torch.arange(0, 10, device=device)
    text_embeddings = labels_to_text_embeddings(labels, device=device)

    with torch.no_grad():
        generated = model.generate(text_embeddings, num_samples=1)

    save_path = os.path.join(output_dir, f'pytorch_samples_epoch_{epoch}.png')
    visualize_training_samples(generated, labels, num_samples=10, save_path=save_path)
    print(f"Generated samples: {save_path}")


def plot_training_curves(train_losses, test_losses, save_path="training_curves.png"):
    """Plot and save training/validation curves for total, reconstruction, and KL losses."""
    epochs = range(1, len(train_losses['total']) + 1)
    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses['total'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, test_losses['total'], 'r-', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_losses['recon'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, test_losses['recon'], 'r-', label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_losses['kl'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, test_losses['kl'], 'r-', label='Test', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('KL Divergence', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


def main():
    """Main training loop."""
    print("="*80)
    print("CVAE Training")
    print("="*80)

    device = config.DEVICE
    print(f"\nDevice: {device}")

    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        batch_size=config.BATCH_SIZE,
        data_dir=config.DATA_DIR,
        val_split=0.2
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")  # type: ignore

    print("\nInitializing model...")
    model = CVAE(
        input_dim=config.INPUT_DIM,
        label_dim=config.LABEL_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nTraining for {config.NUM_EPOCHS} epochs")
    print(f"KL annealing: 0→1.0 over {config.KL_WARMUP_EPOCHS} epochs")
    print(f"Learning rate: {config.LEARNING_RATE}")

    best_val_loss = float('inf')

    train_losses = {'total': [], 'recon': [], 'kl': []}
    val_losses = {'total': [], 'recon': [], 'kl': []}
    test_losses = {'total': [], 'recon': [], 'kl': []}

    csv_path = "training_losses.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Train_Recon', 'Train_KL',
                         'Val_Loss', 'Val_Recon', 'Val_KL',
                         'Test_Loss', 'Test_Recon', 'Test_KL', 'Beta'])
    print(f"Logging to: {csv_path}")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        beta = get_kl_annealing_factor(epoch)

        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - Beta: {beta:.3f}")
        print(f"{'='*80}")

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, epoch, beta
        )

        val_loss, val_recon, val_kl = evaluate(model, val_loader, device)

        train_losses['total'].append(train_loss)
        train_losses['recon'].append(train_recon)
        train_losses['kl'].append(train_kl)
        val_losses['total'].append(val_loss)
        val_losses['recon'].append(val_recon)
        val_losses['kl'].append(val_kl)

        print(f"\nTrain - Loss: {train_loss:.2f}, Recon: {train_recon:.2f}, KL: {train_kl:.2f}")
        print(f"Val   - Loss: {val_loss:.2f}, Recon: {val_recon:.2f}, KL: {val_kl:.2f}")

        if epoch % config.TEST_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
            test_loss, test_recon, test_kl = evaluate(model, test_loader, device)
            print(f"Test  - Loss: {test_loss:.2f}, Recon: {test_recon:.2f}, KL: {test_kl:.2f}")
        else:
            test_loss, test_recon, test_kl = None, None, None

        test_losses['total'].append(test_loss if test_loss is not None else float('nan'))
        test_losses['recon'].append(test_recon if test_recon is not None else float('nan'))
        test_losses['kl'].append(test_kl if test_kl is not None else float('nan'))

        csv_writer.writerow([epoch, train_loss, train_recon, train_kl,
                            val_loss, val_recon, val_kl,
                            test_loss if test_loss is not None else '',
                            test_recon if test_recon is not None else '',
                            test_kl if test_kl is not None else '',
                            beta])
        csv_file.flush()

        if epoch % config.SAVE_INTERVAL == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'model_pytorch_epoch_{epoch}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
            visualize_samples(model, device, epoch)
            plot_training_curves(train_losses, val_losses, save_path="training_curves.png")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model_pytorch.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)

    csv_file.close()
    print(f"\nLosses saved to: {csv_path}")

    plot_training_curves(train_losses, val_losses, save_path="training_curves.png")

    print(f"\n{'='*80}")
    print("Final Evaluation")
    print(f"{'='*80}")
    final_test_loss, final_test_recon, final_test_kl = evaluate(model, test_loader, device)
    print(f"Test - Loss: {final_test_loss:.2f}, Recon: {final_test_recon:.2f}, KL: {final_test_kl:.2f}")

    print(f"\n{'='*80}")
    print("Training complete")
    print(f"Best val loss: {best_val_loss:.2f}")
    print(f"Final test loss: {final_test_loss:.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
