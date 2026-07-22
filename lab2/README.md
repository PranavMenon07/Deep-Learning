# Lab 2 — Multi-Layer Perceptron for Multi-Class Image Classification

**Course:** CS3807 – Deep Learning Laboratory
**Experiment 2:** Multi-Layer Perceptron (MLP) built with Keras/TensorFlow to classify clothing images from the Fashion-MNIST dataset into 10 categories.

## Overview

This experiment implements a fully-connected feed-forward neural network (MLP) for multi-class image classification. It covers dataset exploration, preprocessing, model construction, training, evaluation, and hyperparameter optimization via a manual randomized search with K-Fold cross-validation.

### Tasks covered
| Task | Description |
|------|-------------|
| Task 1 | Dataset exploration — shapes, class distribution, sample image visualization |
| Task 2 | Data preprocessing — pixel normalization (0–1), flattening 28x28 images to 784-length vectors, one-hot label encoding |
| Task 3 | Model construction — baseline MLP (Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)) |
| Task 4 | Model training — 20 epochs, batch size 32, Adam optimizer, categorical cross-entropy loss |
| Task 5 | Model evaluation — accuracy, precision, recall, F1-score, confusion matrix, classification report |
| Section 7 | Hyperparameter optimization — manual randomized search + K-Fold cross-validation implemented directly in Keras (no SciKeras, to avoid a known SciKeras/scikit-learn version-compatibility bug) |
| Section 8 | 9 mandatory plots saved as PNGs for the report |

## Repository Structure

```
lab2-mlp-fashion-mnist/
├── README.md                          # This file
├── DATASET.md                         # Dataset information
├── requirements.txt                   # Python dependencies
├── lab2_mlp_fashion_mnist.ipynb       # Main notebook (Google Colab / Jupyter)
└── figures/                           # Generated plots (created on first run)
```

## Model Details

- **Architecture (baseline):** `Input(784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Epochs:** 20 · **Batch size:** 32 · **Validation split:** 10%
- **Reproducibility:** `SEED = 42` set for both NumPy and TensorFlow
- **Hyperparameter search:** Manual `ParameterSampler`-based randomized search combined with `KFold` cross-validation (pure Keras — deliberately avoids the `scikeras` wrapper due to a `__sklearn_tags__` incompatibility with newer scikit-learn versions)

## Outputs Produced

Running the notebook generates 9 plots inside `figures/`:

| File | Description |
|---|---|
| `01_sample_images.png` | Grid of sample Fashion-MNIST images with class labels |
| `02_class_distribution.png` | Bar chart of training samples per class |
| `03_training_accuracy.png` | Training accuracy per epoch |
| `04_validation_accuracy.png` | Validation accuracy per epoch |
| `05_training_loss.png` | Training loss per epoch |
| `06_validation_loss.png` | Validation loss per epoch |
| `07_confusion_matrix.png` | Confusion matrix on the test set |
| `08_hyperparameter_search.png` | Results of the randomized hyperparameter search |
| `09_accuracy_comparison.png` | Baseline vs. tuned model accuracy comparison |

Console output includes dataset shapes, model summary, per-epoch training logs, and final test-set metrics (accuracy, precision, recall, F1-score, confusion matrix, classification report).

## Dependencies

See `requirements.txt`. Key libraries: TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Seaborn.

## Dataset

See `DATASET.md`. Fashion-MNIST is downloaded automatically via `keras.datasets.fashion_mnist.load_data()` — no manual download needed.

---

## Execution Instructions

### Option A — Run in Google Colab (recommended)

1. Go to [Google Colab](https://colab.research.google.com).
2. Upload `lab2_mlp_fashion_mnist.ipynb` via **File → Upload Notebook**, or open it directly from your GitHub repo via **File → Open Notebook → GitHub**.
3. Run all cells: **Runtime → Run all**. The first two cells install/pin package versions (`scikeras`, `scikit-learn<1.6`) — these are Colab-specific setup cells and are safe to skip if your local environment already satisfies `requirements.txt`.
4. Fashion-MNIST downloads automatically the first time `keras.datasets.fashion_mnist.load_data()` runs — no manual dataset upload needed.
5. Generated plots will appear in a `figures/` folder in the Colab session. The last two cells zip the `figures/` folder and trigger a browser download of `fashion_mnist_plots.zip` (Colab-only — see Notes below).

### Option B — Run locally with Jupyter

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/deep-learning.git
   cd deep-learning/lab2-mlp-fashion-mnist
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter and run the notebook**
   ```bash
   jupyter notebook lab2_mlp_fashion_mnist.ipynb
   ```
   Then run all cells: **Cell → Run All**. Skip the first two `!pip install` cells and the final Colab-download cell (see Notes below) — they are not needed/won't work outside Colab.

5. **Check outputs**
   - Console output shows dataset shapes, model summary, training progress per epoch, and final evaluation metrics.
   - All 9 plots are saved automatically inside a `figures/` folder.

### Option C — Run as a plain Python script

```bash
jupyter nbconvert --to script lab2_mlp_fashion_mnist.ipynb
python lab2_mlp_fashion_mnist.py
```
(Remove/comment out the Colab-only cells first — see Notes below.)

### Notes / Troubleshooting

- **Cells to skip when running locally:**
  - `!pip install scikeras` and `!pip install "scikit-learn<1.6" --force-reinstall` — Colab environment setup only; use `requirements.txt` instead.
  - The final cell using `from google.colab import files` — only works inside Colab; it downloads the zipped plots to your browser. Locally, just open the `figures/` folder directly.
- **Training time:** Training a 20-epoch MLP on Fashion-MNIST (60,000 images) on CPU typically takes a few minutes; a GPU runtime in Colab (**Runtime → Change runtime type → GPU**) will speed this up significantly, especially for the hyperparameter search step.
- **Reproducibility:** Results should be consistent across runs thanks to `SEED = 42`, though minor floating-point differences between hardware/TensorFlow versions are normal.
