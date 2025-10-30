"""Text-to-image generation for MNIST digits using CVAE."""

import os
import argparse
import torch

import sys
sys.path.append('.')

import config
from src.model.vae import CVAE
from src.model.text_encoder import parse_text_to_digit
from src.utils.data_loader import labels_to_onehot
from src.utils.image_utils import save_generated_image, display_image_in_terminal


def generate_from_text(text_prompt: str, model: CVAE, device: torch.device, num_samples: int = 1):
    """Generate digit images from natural language prompt."""
    digit = parse_text_to_digit(text_prompt)
    print(f"Parsed digit: {digit}")

    label = torch.tensor([digit], dtype=torch.long, device=device)
    label_onehot = labels_to_onehot(label).to(device)

    model.eval()
    with torch.no_grad():
        generated_flat = model.generate(label_onehot, num_samples=num_samples)
        generated = generated_flat.view(num_samples, 1, 28, 28)

    return generated, digit


def main():
    parser = argparse.ArgumentParser(description="Generate MNIST digits from text prompts")
    parser.add_argument("prompt", type=str, help="Text prompt (e.g., 'Print number 5')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best_model_pytorch.pt)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate (default: 1)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for generated images")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display image in terminal (default: display)")

    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model_pytorch.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        if os.path.exists(config.CHECKPOINT_DIR):
            checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pt')]
            for ckpt in sorted(checkpoints):
                print(f"  - {ckpt}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")

    model = CVAE(
        input_dim=config.INPUT_DIM,
        label_dim=config.LABEL_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    print(f"\nGenerating from prompt: '{args.prompt}'")
    generated_images, digit = generate_from_text(args.prompt, model, device, args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)

    for img in generated_images:
        filepath = save_generated_image(img, digit, output_dir=args.output_dir)
        print(f"Saved: {filepath}")

    if not args.no_display:
        display_image_in_terminal(generated_images[0], digit)

    print("Generation complete!")


if __name__ == "__main__":
    main()
