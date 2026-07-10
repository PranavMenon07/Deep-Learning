# Lab 1 — Single Layer Perceptron for Binary Classification

**Course:** CS3807 – Deep Learning Laboratory
**Experiment 1:** Single Layer Perceptron (SLP) trained on the Banknote Authentication dataset to classify banknotes as genuine or forged.

## Overview

This experiment implements a single-layer perceptron **from scratch using NumPy** (no deep learning framework) and applies it to a binary classification problem. It covers the full pipeline: dataset exploration, exploratory data analysis (EDA), preprocessing, model implementation, training, and evaluation.

### Tasks covered
| Task | Description |
|------|-------------|
| Task 1 | Dataset exploration (shape, missing values, summary statistics) |
| Task 2 | Exploratory Data Analysis — histograms, correlation heatmap, scatter plot, boxplots |
| Task 3 | Data preprocessing — feature scaling (`StandardScaler`) and train/test split |
| Task 4 & 5 | Perceptron implementation (step activation, perceptron learning rule) and training |
| Task 6 | Model evaluation — accuracy, precision, recall, F1-score, confusion matrix |
| Extra | Training dynamics plots (error vs. epoch, weight evolution, bias evolution) and a learning-rate comparison study |

## Repository Structure

```
lab1-perceptron/
├── README.md                          # This file
├── DATASET.md                         # Dataset information
├── requirements.txt                   # Python dependencies
├── lab1_perceptron.ipynb              # Main notebook (Google Colab / Jupyter)
├── data/
│   └── data_banknote_authentication.txt   # Dataset (add manually, see DATASET.md)
└── plots/                             # Generated figures (created on first run)
```

## Model Details

- **Architecture:** Single-layer perceptron (4 input features → 1 output neuron)
- **Activation:** Step function (`1` if `w·x + b >= 0`, else `0`)
- **Learning rule:** Classic perceptron update rule — `Δw = lr * (target - prediction) * x`
- **Hyperparameters (default):** learning rate = `0.01`, epochs = `50`, random seed = `42`
- **Train/test split:** 80% / 20%, stratified on the class label

## Outputs Produced

Running the notebook generates the following plots inside `plots/`:

- `histograms.png` — feature distributions
- `correlation_heatmap.png` — feature correlation matrix
- `scatter_plot.png` — variance vs. skewness colored by class
- `boxplots.png` — feature spread/outliers
- `combined_eda.png` — all EDA plots combined in a single figure
- `confusion_matrix.png` — test set confusion matrix
- `error_vs_epoch.png` — misclassifications per epoch (convergence curve)
- `weight_evolution.png` — how each weight changes over training
- `bias_evolution.png` — how the bias changes over training
- `learning_rate_comparison.png` — convergence comparison across learning rates `[0.001, 0.01, 0.1]`

It also prints accuracy, precision, recall, F1-score, and the confusion matrix on the test set to the console.

## How to Run

See **Execution Instructions** below.

## Dependencies

See `requirements.txt`. Full details in this README's Execution Instructions section.

## Dataset

See `DATASET.md` for source, features, and download instructions.

---

## Execution Instructions

### Option A — Run in Google Colab (recommended, no setup required)

1. Go to [Google Colab](https://colab.research.google.com).
2. Click **File → Upload Notebook** and select `lab1_perceptron.ipynb`, or open it directly from your GitHub repo via **File → Open Notebook → GitHub**.
3. Upload the dataset file `data_banknote_authentication.txt` to the Colab session (left sidebar → **Files** → drag and drop), or mount Google Drive and update `DATA_PATH` accordingly.
4. Run all cells: **Runtime → Run all**.
5. Generated plots will appear in a `plots/` folder in the Colab session, and can be downloaded individually or as a zip (a cell in the notebook already zips them into `plots.zip`).

### Option B — Run locally with Jupyter

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/deep-learning.git
   cd deep-learning/lab1-perceptron
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

4. **Download the dataset**
   Download `data_banknote_authentication.txt` from the [UCI Banknote Authentication dataset page](https://archive.ics.uci.edu/dataset/267/banknote+authentication) and place it in the same folder as the notebook (or update `DATA_PATH` in the code to point to wherever you saved it).

5. **Launch Jupyter and run the notebook**
   ```bash
   jupyter notebook lab1_perceptron.ipynb
   ```
   Then run all cells: **Cell → Run All**.

6. **Check outputs**
   - Console output will show dataset statistics, per-epoch training progress, and final test-set metrics (accuracy, precision, recall, F1, confusion matrix).
   - All plots will be saved automatically inside a `plots/` folder.

### Option C — Run as a plain Python script

If you prefer a `.py` script instead of a notebook, export the notebook first:
```bash
jupyter nbconvert --to script lab1_perceptron.ipynb
python lab1_perceptron.py
```

### Notes / Troubleshooting

- **`FileNotFoundError` for the dataset:** Make sure `data_banknote_authentication.txt` is in the same directory as the notebook/script, or update the `DATA_PATH` variable at the top of the code.
- **`google.colab` import error when running locally:** The last cell mounts Google Drive and only works inside Colab — skip that cell (or delete it) when running locally/in Jupyter.
- Results are reproducible thanks to `RANDOM_STATE = 42` used for both `numpy` and the train/test split.
