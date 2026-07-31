# Dataset Information — CIFAR-10

## Overview

| Property | Detail |
|---|---|
| **Name** | CIFAR-10 (Canadian Institute For Advanced Research, 10 classes) |
| **Created by** | Alex Krizhevsky, Vinod Nair, Geoffrey Hinton (University of Toronto) |
| **Loaded via** | `tensorflow.keras.datasets.cifar10.load_data()` |
| **Original source** | https://www.cs.toronto.edu/~kriz/cifar.html |
| **Download** | Automatic on first run — Keras downloads and caches the dataset, no manual steps required |
| **Cache location** | `~/.keras/datasets/cifar-10-batches-py/` (Linux/Mac) or `%USERPROFILE%\.keras\datasets\` (Windows) |
| **Download size** | ~163 MB (compressed) |
| **License** | Free for research/educational use (see original source page) |

## Composition

| Split | Images | Shape |
|---|---|---|
| Training | 50,000 | `(50000, 32, 32, 3)` |
| Test | 10,000 | `(10000, 32, 32, 3)` |
| **Total** | **60,000** | — |

- **Image size:** 32 × 32 pixels
- **Color channels:** 3 (RGB)
- **Pixel value range (raw):** 0–255 (`uint8`)
- **Pixel value range (after preprocessing in this notebook):** 0.0–1.0 (`float32`, via `/255.0` normalization)
- **Balance:** Perfectly balanced — 6,000 images per class (5,000 train + 1,000 test per class)

## Classes

10 mutually exclusive classes, label indices `0`–`9`:

| Index | Class |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Label Format

- **Raw labels:** integer arrays of shape `(N, 1)`, values `0`–`9`
- **Used in this notebook as:**
  - `y_train`, `y_test` — raw integer labels (used for confusion matrix, classification report, flattened via `.flatten()`)
  - `y_train_cat`, `y_test_cat` — one-hot encoded via `tf.keras.utils.to_categorical(y, 10)`, shape `(N, 10)`, used for model training with `categorical_crossentropy` loss

## Preprocessing Applied in the Notebook

1. **Normalization:** pixel values scaled from `[0, 255]` to `[0.0, 1.0]` (`astype('float32') / 255.0`)
2. **Label encoding:** integer labels one-hot encoded to shape `(N, 10)` for categorical cross-entropy training
3. No resizing, augmentation, or cropping is applied — images are used at native 32×32 resolution

## Known Characteristics / Caveats

- CIFAR-10 images are low resolution (32×32), so fine-grained detail is limited; some classes
  (e.g., `cat` vs. `dog`, `automobile` vs. `truck`) are commonly confused by CNNs, which is
  visible in the confusion matrix produced by this notebook.
- No validation split is provided natively — the notebook creates one from the training set
  via `validation_split=0.1` in `model.fit(...)`, so training effectively uses 45,000 images
  and validates on 5,000.
- No missing values or corrupted samples; the dataset is a clean, standard benchmark.

## Citation

If citing this dataset in a report:

> Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
> Technical report, University of Toronto.
