# Experiment 4 – Comparative Study of Deep CNN Architectures Using Transfer Learning

**Course:** CS3807 – Deep Learning Laboratory
**Notebook:** `experiment4_transfer_learning.ipynb`

## Overview

This experiment applies transfer learning to the CIFAR-10 image classification task using pretrained CNN backbones (VGG16 by default, with ResNet50 and MobileNetV2 also supported). The pipeline covers:

- **Task 1** – Dataset loading and preparation
- **Task 2** – Transfer learning model setup (pretrained base + custom classification head)
- **Task 3** – Model training with the convolutional base frozen (feature extraction)
- **Task 4** – Fine-tuning (unfreezing the last few layers of the base model)
- **Task 5** – Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- **Bonus** – A small hyperparameter study (learning rate, batch size, optimizer) — commented out by default in `main`

## Dataset Information

- **Name:** CIFAR-10
- **Source:** Loaded automatically via `tf.keras.datasets.cifar10.load_data()` (downloaded on first run; no manual download needed)
- **Size:** 60,000 32×32 color images across 10 classes (50,000 train / 10,000 test)
- **Classes:** `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
- **Preprocessing:**
  - Pixel values normalized to `[0, 1]`
  - Labels one-hot encoded (`to_categorical`, 10 classes)
  - Images are resized from 32×32 to the backbone's expected input size (default 96×96) via a `Resizing` layer inside the model
  - A 90/10 split of the training set is used to create a validation set

## Dependencies

Python 3.9+ is recommended. Install the required packages with:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

| Package | Purpose |
|---|---|
| `tensorflow` (includes `keras`) | Model building, pretrained applications (VGG16/ResNet50/MobileNetV2), training |
| `numpy` | Array/data handling |
| `matplotlib` | Plotting sample images, training curves, confusion matrix |
| `scikit-learn` | Evaluation metrics (accuracy, precision, recall, F1, confusion matrix, classification report) |
| `seaborn` | Confusion matrix heatmap |

> A GPU (with CUDA-enabled TensorFlow) is strongly recommended — training VGG16/ResNet50 on CPU will be slow.

## Execution Instructions

### Option A: Run as a Jupyter Notebook
1. Ensure dependencies are installed (see above).
2. Launch Jupyter: `jupyter notebook` or `jupyter lab`.
3. Open `experiment4_transfer_learning.ipynb`.
4. Run all cells in order (**Cell → Run All**, or run sequentially with Shift+Enter).
   - The first cell runs the full pipeline: data loading → model build → training → fine-tuning → evaluation.
   - The second cell zips the `outputs/` folder into `output_images.zip` for easy download.

### Option B: Run as a Python Script
1. Export the notebook to a script (optional, since the notebook is already script-like):
   ```bash
   jupyter nbconvert --to script experiment4_transfer_learning.ipynb
   ```
2. Run it:
   ```bash
   python experiment4_transfer_learning.py
   ```

### What Happens on Run
1. CIFAR-10 is downloaded and preprocessed; a sample image grid is saved.
2. A VGG16-based model is built with the base frozen, and trained for 10 epochs (feature extraction).
3. The model is evaluated **before** fine-tuning.
4. The last 4 layers of the base model are unfrozen and fine-tuned for 5 epochs at a low learning rate.
5. Training/loss curves are plotted and saved.
6. The model is evaluated **again after** fine-tuning, and a before/after accuracy summary is printed.

### Output Artifacts (saved to `outputs/`)
- `sample_images.png` – Sample CIFAR-10 images with labels
- `accuracy_curve_VGG16.png`, `loss_curve_VGG16.png` – Training/validation curves
- `confusion_matrix.png` – Confusion matrix heatmap
- `misclassified_images.png` – Examples of misclassified test images

Running the notebook's second cell packages all of the above into `output_images.zip`.

### Optional: Hyperparameter Study
The `hyperparameter_study()` function runs a small grid search (learning rate, batch size, optimizer) using VGG16. It is defined but **not called by default**. To run it, uncomment the corresponding lines at the bottom of the script/notebook:
```python
# results = hyperparameter_study(x_tr, y_tr, x_val, y_val, x_test, y_test_cat, y_test)
# for r in results:
#     print(r)
```

### Customization
- **Change backbone:** pass `base_name="ResNet50"` or `base_name="MobileNetV2"` to `build_transfer_model()`.
- **Adjust training:** modify `lr`, `batch_size`, `epochs`, and `optimizer_name` in the `train_model()` / `fine_tune_model()` calls.
- **Unfreeze more/fewer layers during fine-tuning:** change `unfreeze_last_n` in `fine_tune_model()`.
