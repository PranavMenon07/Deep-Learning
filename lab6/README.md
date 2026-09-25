# CS3807 — Deep Learning Laboratory — Experiment 6
## End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

Shiv Nadar University Chennai · B.Tech Artificial Intelligence & Data Science · Semester V · AY 2026–27

This repository implements the complete Experiment 6 pipeline: sequence classification on the
UCI HAR dataset with Vanilla RNN / LSTM / GRU, a CNN + LSTM/GRU video-understanding pipeline on
a UCF101 subset, and a synthetic sequence-to-sequence (reversal) task with an encoder–decoder LSTM.

## Repository Contents

| File | Description |
|---|---|
| `Experiment_6_RNN_LSTM_GRU.ipynb` | Main Colab/Jupyter notebook — the full runnable pipeline. |
| `Experiment_6_Report.tex` | LaTeX lab report, structured to match the experiment sheet. |
| `Experiment_6_Report.pdf` | Compiled report (generated locally, not tracked — see below). |
| `DATASET.md` | Description of the datasets used and how to obtain them. |
| `requirements.txt` | Python dependencies for running the notebook locally. |
| `setup.sh` | One-shot environment setup script (venv + dependencies). |
| `.gitignore` | Excludes datasets, model artifacts, plots and LaTeX build files from version control. |

## Quick Start

### Option A — Google Colab (recommended)
1. Upload `Experiment_6_RNN_LSTM_GRU.ipynb` to [Google Colab](https://colab.research.google.com).
2. `Runtime > Change runtime type > GPU`.
3. Run all cells top to bottom. The UCI HAR dataset downloads automatically; see
   `DATASET.md` for the video (UCF101) subset, which must be supplied manually.

### Option B — Local setup
```bash
git clone <this-repo-url>
cd <repo-folder>
bash setup.sh
source venv/bin/activate
jupyter notebook Experiment_6_RNN_LSTM_GRU.ipynb
```

See `DATASET.md` for dataset download instructions and expected folder layout.

## Pipeline Overview

```
Sequential Data (UCI HAR) → Preprocessing → RNN / LSTM / GRU → Evaluation
Video (UCF101 subset)      → Frames → CNN (MobileNetV2) Features → LSTM/GRU → Action Prediction
Synthetic integer sequences → Encoder (LSTM) → Context → Decoder (LSTM) → Reversed Sequence
```

Full details, architecture diagrams, formulas, required plots and discussion-question answers are
in `Experiment_6_Report.tex` / the compiled PDF.

## Building the Report

```bash
pdflatex Experiment_6_Report.tex
pdflatex Experiment_6_Report.tex   # run twice for cross-references/ToC if added
```

Requires a TeX distribution (TeX Live / MiKTeX) with `tikz`, `booktabs`, `hyperref`. Copy the
`plotN_*.png` files produced by the notebook into the same folder as the `.tex` file before
compiling, so the figure includes resolve.

## Notes

- The test split is never used for model selection or hyperparameter tuning, per the experiment
  protocol.
- All numeric results (accuracy, F1, parameter counts, training times) in the report must come
  from your own notebook execution — none are pre-filled.
- Raw datasets, trained model weights, and generated plots are intentionally excluded from
  version control (see `.gitignore`); re-run the notebook to regenerate them.

## References

1. Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press, 2016.
2. Hochreiter & Schmidhuber — "Long Short-Term Memory," *Neural Computation*, 1997.
3. Cho et al. — "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
   Machine Translation," EMNLP, 2014.
4. Anguita et al. — "A Public Domain Dataset for Human Activity Recognition Using
   Smartphones," ESANN, 2013.
5. Soomro, Zamir, Shah — "UCF101: A Dataset of 101 Human Action Classes From Videos in the
   Wild," 2012.
