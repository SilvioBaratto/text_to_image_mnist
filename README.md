# Text-to-Image MNIST Generator

A Conditional Variational Autoencoder (CVAE) that generates MNIST digit images from natural language text prompts.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a text-conditioned VAE that can generate handwritten digit images (0-9) from natural language commands. Simply type "Print number 3" or "Generate digit 7" and the model will create a realistic MNIST-style handwritten digit.

**Example Usage:**

```bash
python main.py "Print number 5"
# Generates: outputs/generated_5_20250130_143022.png
```

## Training Progression

Watch the model learn to generate all 10 digits during training:

<p align="center">
  <img src="training_progression.gif" alt="Training Progression" width="800"/>
</p>

The animation shows how the VAE progressively learns to generate clearer, more recognizable digits from random noise as training progresses through the epochs.

## Model Architecture

### Conditional VAE (CVAE)

The model consists of three main components:

#### 1. Semantic Text Encoder

Uses **SentenceTransformer** (all-MiniLM-L6-v2) to encode natural language prompts into 384-dimensional semantic embeddings:

- **No keyword parsing** - understands full sentence meaning
- Supports diverse natural language: "I want to draw a zero", "Can you show me what five looks like?", "Generate the digit three"
- Learns associations between text semantics and digit images
- Pre-trained language model provides rich contextual understanding

#### 2. Encoder Network (Convolutional)

Maps images and labels to a latent distribution using convolutional layers:

- **Input**: 28×28 grayscale image (reshaped from 784-dim vector)
- **Conv Layer 1**: 32 filters (3×3, stride=2) → 32×14×14 feature maps
- **Conv Layer 2**: 64 filters (3×3, stride=2) → 64×7×7 feature maps
- **Flatten + Concat**: 3136 features + 10 (one-hot label) = 3146 dimensions
- **Hidden Layer**: 512 units with ReLU activation
- **Output**: 20-dimensional latent space (mean μ and log variance log σ²)
- **Reparameterization Trick**: z = μ + σ × ε, where ε ~ N(0,1)

#### 3. Decoder Network (Transposed Convolutions)

Reconstructs images from latent codes and labels using deconvolution:

- **Input**: 20 (latent vector) + 10 (one-hot label) = 30 dimensions
- **Hidden Layers**: 512 units → 3136 units (reshaped to 64×7×7)
- **Deconv Layer 1**: 64→32 filters (3×3, stride=2) → 32×14×14
- **Deconv Layer 2**: 32→1 filter (3×3, stride=2) → 1×28×28
- **Output**: 784 pixels with sigmoid activation (flattened 28×28 image)

### Loss Function

The model is trained with a composite loss:

- **Reconstruction Loss**: Binary Cross-Entropy (BCE) between input and reconstructed images
- **KL Divergence**: Regularizes the latent space to follow N(0,1) distribution

```
Total Loss = Reconstruction Loss + KL Divergence
```

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- torchvision
- NumPy
- Matplotlib
- sentence-transformers 5.1.2+ (for semantic text understanding)
- tqdm (progress bars)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/SilvioBaratto/text_to_image_mnist.git
cd text_to_image_mnist
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the VAE on the MNIST dataset:

```bash
python src/train.py
```

**Training Details:**

- Dataset: MNIST (60,000 training images)
- Text prompts: 28 diverse templates automatically generated per digit
- Batch size: 32 (adjusted for text encoding overhead)
- Epochs: 200 (semantic learning requires more iterations)
- Optimizer: Adam (lr=0.0001)
- Training time: ~30-60 minutes on Apple Silicon GPU (MPS)
- KL annealing: Gradual warmup over 30 epochs for stability

Checkpoints are saved to `checkpoints/` every 5 epochs.

**Note:** The first epoch will download the SentenceTransformer model (~80MB) once.

### Generating Images

Generate digit images from text prompts:

```bash
# Natural language - semantic understanding!
python main.py "I want to draw a zero"
python main.py "Can you show me what five looks like?"
python main.py "Please generate the digit three"
python main.py "Draw a handwritten seven"

# Simple forms also work
python main.py "Generate digit 7"
python main.py "Show me a five"

# Use specific checkpoint
python main.py "I want to draw a two" --checkpoint checkpoints/model_epoch_50.pt
```

Generated images are saved to `outputs/` with timestamps.

## Configuration

Key hyperparameters in `config.py`:

| Parameter       | Value | Description             |
| --------------- | ----- | ----------------------- |
| `latent_dim`    | 20    | Latent space dimensions |
| `hidden_dim`    | 512   | Hidden layer size       |
| `text_embedding_dim` | 384 | SentenceTransformer embedding size |
| `batch_size`    | 32    | Training batch size (reduced for text encoding) |
| `learning_rate` | 0.0001 | Adam optimizer LR (lower for stability) |
| `epochs`        | 200   | Training epochs (more for semantic learning) |
| `kl_warmup_epochs` | 30 | KL annealing warmup period |

## Technical Details

### Conditional Generation with Semantic Text Understanding

The model conditions both the encoder and decoder on **384-dimensional text embeddings** from SentenceTransformer. This allows the VAE to:

- **Understand semantic meaning** of natural language prompts (not just keywords)
- Learn associations between text semantics and visual digit representations
- Generate specific digits from diverse phrasings
- Leverage pre-trained language understanding for richer conditioning
- Support creative and varied prompt formulations

### Reparameterization Trick

To enable backpropagation through stochastic sampling, we use:

```python
z = mu + exp(0.5 * log_var) * epsilon
```

where `epsilon ~ N(0,1)` is sampled noise.

### Training Considerations

**KL Vanishing**: If KL divergence becomes very small while reconstruction loss stays high, consider:

- Reducing KL weight initially (multiply by 0.1)
- Using KL annealing schedule
- Increasing training epochs

**Blurry Outputs**: VAEs naturally produce slightly blurred images compared to GANs. This is expected and acceptable - digits should still be clearly recognizable.

## Expected Results

Generated images should be:

- Clearly recognizable as the requested digit (0-9)
- Similar to MNIST handwriting style
- Show natural variation across generations
- Slightly blurry (normal for VAEs)

Success criteria: The system reliably generates the correct digit from text prompts.

## References

- [Pyro CVAE Tutorial](https://pyro.ai/examples/cvae.html) - Official Conditional VAE implementation guide
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Original VAE paper (Kingma & Welling, 2013)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Dataset information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset by Yann LeCun
- PyTorch team for the deep learning framework
- Pyro probabilistic programming library for CVAE inspiration
