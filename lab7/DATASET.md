# Dataset Description

## Source
**MNIST Handwritten Digit Database**
Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
Loaded directly via Keras:
`tf.keras.datasets.mnist.load_data()` —
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist

## Contents
- Grayscale images of handwritten digits (0–9).
- Full dataset: 60,000 training images, 10,000 test images.
- Image shape: `28 × 28 × 1` (single channel, 8-bit pixel values `0–255`).
- Labels (0–9) are included but **not used as a training target** anywhere
  in this experiment — the target for every model is the input image itself
  (`x → Encoder → z → Decoder → x̂`). Labels are used only once, purely for
  coloring the VAE's 2-D latent-space scatter plot (Plot 6).

## Preprocessing
1. Pixel values rescaled from `[0, 255]` to `[0, 1]` (`float32`).
2. A fixed-seed random subset is drawn for the laboratory run:
   - **10,000 training images** (from the 60,000 available)
   - **2,000 test images** (from the 10,000 available)
   - Sampled with `numpy.random.default_rng(42)` for reproducibility.
   - The full MNIST dataset may be substituted if compute allows — just skip
     the subsampling step.
3. Two input formats are derived from the same images:
   - **Flattened**: `(N, 784)` — used by the fully connected autoencoder.
   - **Spatial**: `(N, 28, 28, 1)` — used by the CAE, denoising CAE, and VAE.

## Noise Corruption (Denoising Autoencoder only)
Applied on-the-fly to the spatial (`28×28×1`) training/test images; the
**clean** image is always kept as the reconstruction target.

- **Gaussian noise**: `x̃ = clip(x + N(0, σ²), 0, 1)`, with
  `σ ∈ {0.1, 0.2, 0.3}` (model trained at `σ = 0.3`, evaluated at all three).
- **Salt-and-pepper noise**: a random `p` fraction of pixels forced to `0` or
  `1`, with `p ∈ {0.05, 0.10, 0.20}` (used for evaluation/comparison).

## Why MNIST
- Small, fixed image size works directly with fully connected, convolutional,
  and variational architectures without modification.
- Class structure (10 visually distinct but sometimes-similar digit shapes)
  gives meaningful, interpretable latent-space clustering for the VAE.
- Low resolution keeps training times short enough for a single lab session,
  including on a free-tier Colab GPU.
