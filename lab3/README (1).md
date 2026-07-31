# Lab 3 — Convolutional Neural Networks on CIFAR-10

This lab explores core CNN building blocks (kernels, strides, padding, pooling, feature maps)
and then trains, evaluates, and compares CNN architectures on the CIFAR-10 image
classification dataset using TensorFlow/Keras.

## What This Notebook Does

1. **Data loading & exploration** — loads CIFAR-10, inspects shapes, and visualizes sample
   images and the class distribution.
2. **Kernel size experiment** — compares output feature map shapes for 3×3, 5×5, and 7×7
   convolution kernels.
3. **Stride / padding experiment** — compares Keras output shapes against the manually
   computed formula `(N - F + 2P) / S + 1` for `valid` and `same` padding at strides 1 and 2.
4. **Feature map visualization** — extracts and plots feature maps produced by the first
   convolutional layer.
5. **Pooling comparison** — trains a small CNN with Max Pooling vs. Average Pooling and
   compares test accuracy.
6. **Baseline CNN** — builds, trains (20 epochs), and evaluates a CNN
   (Conv→ReLU→MaxPool ×2 → Dense → Softmax) on CIFAR-10, reporting accuracy, loss,
   precision, recall, F1-score, and a full classification report.
7. **Training curves** — plots training vs. validation accuracy and loss.
8. **Analytical questions** — manual calculation of output size and trainable parameter
   count for given conv layer specs.
9. **Ablation studies**:
   - Max Pooling vs. Average Pooling on the full 20-epoch architecture.
   - Effect of increasing filter count (16 → 64) on accuracy and training time.
10. **Confusion matrix** — visualizes per-class prediction performance.
11. **Result packaging** — zips all generated plots into `generated_images.zip`.

## Dataset Information

- **Name:** CIFAR-10
- **Source:** Loaded automatically via `tensorflow.keras.datasets.cifar10` (downloaded from
  the official CIFAR-10 source on first run, cached locally by Keras — no manual download
  needed).
- **Size:** 60,000 32×32 color images in 10 classes (50,000 training / 10,000 test images).
- **Classes (10):** `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
- **Image shape:** `(32, 32, 3)` — RGB, pixel values 0–255 (normalized to `[0, 1]` in the
  notebook before training).
- **Labels:** Integer class indices `0`–`9`, one-hot encoded (`to_categorical`) for training.

No dataset files need to be manually placed in the project folder — Keras handles the
download and caches it under `~/.keras/datasets/`.

## Dependencies

See `requirements.txt`. Core libraries:

| Package | Purpose |
|---|---|
| `tensorflow` | Model building, training, CIFAR-10 dataset loader |
| `numpy` | Array / numerical operations |
| `matplotlib` | Plotting sample images, curves, distributions |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Precision, recall, F1-score, classification report, confusion matrix |
| `jupyter` | Running the `.ipynb` notebook |

> **Note:** The last two cells use `google.colab.files.download(...)` to download the
> results zip. This only works inside Google Colab. If running locally or in plain
> Jupyter, either skip that final cell or replace it with a normal file save (the zip is
> already written to disk as `generated_images.zip`, so you can retrieve it directly from
> the working directory).

## Execution Instructions

### 1. Set up the environment

```bash
# (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the notebook

```bash
jupyter notebook Lab3.ipynb
```

or, if you use JupyterLab:

```bash
jupyter lab Lab3.ipynb
```

### 3. Run the cells

Run all cells top to bottom (**Kernel → Restart & Run All**). Notes:

- The first training run (baseline CNN) uses 20 epochs on the full 50,000-image training
  set; on CPU this can take a while. A GPU is recommended but not required.
  The notebook prints `GPU Available: [...]` at the top so you can confirm whether TensorFlow
  detected one.
- Generated plots (`sample_images.png`, `class_distribution.png`, `feature_maps.png`,
  `accuracy_curve.png`, `loss_curve.png`, `confusion_matrix.png`) are saved to the working
  directory as each corresponding cell runs.
- The final two cells (zip + Colab download) are optional utility cells for packaging
  results — safe to skip if not using Google Colab.

### 4. Outputs

After a full run, you should have:

- Console output: dataset shapes, kernel/stride/padding comparison tables, model summary,
  training logs, test accuracy/loss, precision/recall/F1, classification report.
- PNG files: sample images, class distribution, feature maps, accuracy curve, loss curve,
  confusion matrix.
- `generated_images.zip`: bundle of the above PNGs (if the zipping cell is run).

## Project Structure

```
.
├── Lab3.ipynb              # Main notebook
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore                # Git ignore rules
└── (generated at runtime)
    ├── sample_images.png
    ├── class_distribution.png
    ├── feature_maps.png
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── confusion_matrix.png
    └── generated_images.zip
```

## Reproducibility

Random seeds are fixed at the top of the notebook:

```python
np.random.seed(42)
tf.random.set_seed(42)
```

Results may still vary slightly across hardware/backends (GPU non-determinism, TF version
differences).
