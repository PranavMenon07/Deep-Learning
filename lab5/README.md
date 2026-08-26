# Experiment 5 -- CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

**Course:** CS3807 -- Deep Learning Laboratory
**Institution:** Shiv Nadar University Chennai
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**AY:** 2026--27
**Name:** CY Pranav Menon
**Roll No:** 24011101014

## Objective

A systematic study of the effect of weight initialization, regularization
(L2, Dropout, Batch Normalization), optimization algorithms (SGD, Momentum,
RMSProp, Adam), CNN hyperparameters, transfer learning, fine-tuning, and
5-fold cross-validation on image classification performance, using a single
architecture (MobileNetV2) and the Oxford-IIIT Pet dataset.

See [`DATASET.md`](DATASET.md) for dataset details and download instructions.

## Folder Structure

```
Lab5/
├── README.md                          -- this file
├── DATASET.md                         -- dataset description & download instructions
├── requirements.txt                   -- Python dependencies
├── .gitignore                         -- files/folders excluded from version control
├── Experiment5_Lab_Report.tex         -- LaTeX lab report (source)
├── Experiment5_Lab_Report.pdf         -- compiled lab report
├── CNN_Pet_Classification_Experiment.py   -- main CNN + MobileNetV2 training script
├── Experiment5_All_Plots.py           -- generates Plots 1-15 (full sweep, longer runtime)
├── Plots_13_14_15.py                  -- standalone script for CV / confusion matrix / misclassified images
└── plots/                             -- generated PNG plots (referenced by the LaTeX report)
    ├── plot01_weight_init_train_loss.png
    ├── plot02_weight_init_val_accuracy.png
    ├── ...
    └── plot15_misclassified_images.png
```

> Note: the LaTeX report expects the plots inside a folder literally named
> `Lab5/` relative to the `.tex` file (`\graphicspath{{Lab5/}}`). If you keep
> plots inside `Lab5/plots/` instead, either move them up one level or update
> the `\graphicspath` line in the `.tex` file accordingly.

## Setup

All scripts are written to run in **Google Colab** (GPU runtime recommended)
and download the dataset automatically -- no manual setup is required there.

To run locally instead:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

1. **`CNN_Pet_Classification_Experiment.py`** -- trains the custom CNN and
   MobileNetV2 transfer-learning model end-to-end, including 5-fold CV and
   final test evaluation.
2. **`Experiment5_All_Plots.py`** -- runs every sub-experiment required by
   the assignment brief and produces Plots 1--15. This is the most
   comprehensive script but also the longest-running.
3. **`Plots_13_14_15.py`** -- a lighter, standalone script that only
   reproduces the 5-fold CV bar chart (Plot 13), confusion matrix (Plot 14),
   and misclassified-image grid (Plot 15).

Each script saves its output plots to `/content/plots/` in Colab -- download
that folder and place the PNGs into `Lab5/plots/` (or `Lab5/`, matching the
LaTeX `\graphicspath`) before compiling the report.

## Compiling the Report

```bash
pdflatex Experiment5_Lab_Report.tex
pdflatex Experiment5_Lab_Report.tex   # run twice to resolve references
```

## Results

See `Experiment5_Lab_Report.pdf` for the full write-up, including all 15
plots, the discussion questions, and the final model evaluation.

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. S. Ioffe, C. Szegedy, "Batch Normalization: Accelerating Deep Network
   Training by Reducing Internal Covariate Shift," ICML, 2015.
3. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear
   Bottlenecks," CVPR, 2018.
4. O. M. Parkhi et al., "Cats and Dogs," CVPR, 2012.
5. TensorFlow Documentation: https://www.tensorflow.org
6. Keras Documentation: https://keras.io
