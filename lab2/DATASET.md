# Dataset Information — Fashion-MNIST

## Source

- **Name:** Fashion-MNIST
- **Provider:** Zalando Research
- **Link:** https://github.com/zalandoresearch/fashion-mnist
- **Access method:** Loaded automatically in the notebook via `tensorflow.keras.datasets.fashion_mnist.load_data()` — **no manual download required.**

## Description

Fashion-MNIST is a dataset of Zalando's article images, intended as a drop-in replacement for the original MNIST digits dataset. It consists of 28x28 grayscale images of clothing items, each associated with one of 10 categories. It's widely used as a benchmark for image classification models because it's harder than digit-MNIST while remaining small and fast to train on.

## Basic Statistics

| Property | Value |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image dimensions | 28 x 28 pixels, grayscale (1 channel) |
| Pixel value range | 0–255 (raw), normalized to 0–1 in preprocessing |
| Number of classes | 10 |
| Missing values | None |

## Classes

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Preprocessing Applied in the Notebook

1. **Normalization:** pixel values scaled from `[0, 255]` to `[0, 1]` (`x / 255.0`)
2. **Flattening:** each `28x28` image reshaped into a `784`-length vector, since the model is a fully-connected MLP (not a CNN)
3. **Label encoding:** integer class labels (`0`–`9`) one-hot encoded into 10-dimensional vectors via `keras.utils.to_categorical`

## How to Get the Data

No manual steps are needed — the first time the notebook runs `keras.datasets.fashion_mnist.load_data()`, TensorFlow automatically downloads the dataset (~30 MB) and caches it locally under `~/.keras/datasets/`. An internet connection is required the first time only.

## License / Usage

Fashion-MNIST is released by Zalando Research under the MIT License and is free to use for research and educational purposes. See the [official repository](https://github.com/zalandoresearch/fashion-mnist) for citation details if required by your course/report.
