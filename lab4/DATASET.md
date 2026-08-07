# Dataset Information

## CIFAR-10

| Property | Details |
|---|---|
| **Name** | CIFAR-10 (Canadian Institute For Advanced Research) |
| **Task type** | Multi-class image classification |
| **Number of classes** | 10 |
| **Class labels** | `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` |
| **Total images** | 60,000 |
| **Training set** | 50,000 images (5,000 per class) |
| **Test set** | 10,000 images (1,000 per class) |
| **Image size** | 32 × 32 pixels, RGB (3 channels) |
| **Format** | NumPy arrays (via Keras dataset loader) |
| **Source / access** | `tf.keras.datasets.cifar10.load_data()` — downloaded automatically on first run, cached locally (typically under `~/.keras/datasets/`) |
| **License** | Free for research/educational use; see the [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html) for details |

### Preprocessing Applied in This Experiment
1. **Normalization** — pixel values scaled from `[0, 255]` to `[0, 1]` (`astype("float32") / 255.0`).
2. **Label encoding** — integer class labels converted to one-hot vectors via `to_categorical(y, num_classes=10)`.
3. **Resizing** — images are upscaled from 32×32 to the pretrained backbone's expected input size (default 96×96×3) using a Keras `Resizing` layer inside the model graph (not a separate preprocessing step on disk).
4. **Train/validation split** — the last 10% of the 50,000 training images is held out as a validation set (`x_val`, `y_val`); the remaining 90% is used for training.
5. **Test set** — the original 10,000-image CIFAR-10 test set is used, untouched, for final evaluation.

### Notes
- No manual download or file placement is required — the first run will fetch and cache the dataset automatically. An internet connection is needed the first time only.
- Because images are small (32×32) and are resized up to 96×96 for the pretrained backbones, some upscaling artifacts are expected; this is a standard workaround when applying ImageNet-pretrained models to CIFAR-scale data.
- If working in an offline/restricted environment, pre-download `cifar-10-python.tar.gz` and place it in the Keras cache directory (`~/.keras/datasets/cifar-10-batches-py`) before running.
