# Experiment 7 — Autoencoders, CAE, Denoising AE, VAE

**CS3807 – Deep Learning Laboratory**
End-to-end study of a fully connected autoencoder, a convolutional
autoencoder (CAE), a denoising CAE, and a variational autoencoder (VAE) on
MNIST — including all required plots, metrics, and a latent-dimension study.

## Repository Structure

```
.
├── Experiment7_Autoencoders_CAE_DenoisingAE_VAE.ipynb   # Colab notebook (all models + plots)
├── Experiment7_Report_Filled.tex                        # Overleaf/LaTeX lab report
├── lab7/                                                 # Saved figures, referenced by the .tex report (gitignored by default)
├── DATASET.md                                            # Dataset description
├── requirements.txt                                      # Python dependencies
├── setup.sh                                               # One-shot environment setup
├── .gitignore
└── README.md
```

## Quick Start

### Option A — Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Experiment7_Autoencoders_CAE_DenoisingAE_VAE.ipynb`
   (`File → Upload notebook`).
3. `Runtime → Change runtime type → GPU`.
4. Run all cells top to bottom (`Runtime → Run all`).
5. Add `os.makedirs('lab7', exist_ok=True)` once near the top, and a
   `plt.savefig('lab7/<name>.png', dpi=200, bbox_inches='tight')` line before
   each `plt.show()` you want to keep, so the figures land in `lab7/` for the
   report.
6. Download the `lab7/` folder (or individual PNGs) plus the generated CSVs
   (`consolidated_results.csv`, `latent_dimension_study.csv`).

### Option B — Local / any Jupyter environment
```bash
git clone <this-repo-url>
cd <repo>
./setup.sh              # creates a virtual env and installs requirements.txt
source .venv/bin/activate
jupyter notebook Experiment7_Autoencoders_CAE_DenoisingAE_VAE.ipynb
```
See `requirements.txt` for exact package versions. A GPU is optional but
significantly speeds up training (especially the VAE, 30 epochs).

## Building the Report

1. Run the notebook and save all plots into `lab7/` (see filenames referenced
   in `Experiment7_Report_Filled.tex`, e.g. `plot1_fc_original_vs_recon.png`).
2. Open `Experiment7_Report_Filled.tex` in Overleaf (or compile locally with
   `pdflatex`), upload the `lab7/` folder alongside it.
3. Replace your name/registration number (marked in red as `[FILL IN: ...]`)
   and swap in your own run's MSE/MAE/SSIM/timing numbers where they differ
   from the representative values already filled in.

## Models Covered

| Model              | Architecture summary                                   |
|--------------------|----------------------------------------------------------|
| Fully Connected AE | 784→128→32→16(latent)→32→128→784, ReLU + sigmoid output |
| Convolutional AE   | Conv(32)→Pool→Conv(64)→Pool→Conv(64)→UpSample→Conv(32)→UpSample→Conv(1) |
| Denoising CAE      | Same as CAE, trained on Gaussian-noised inputs (σ=0.3), evaluated at σ∈{0.1,0.2,0.3} and salt-and-pepper noise |
| VAE                | Conv encoder → (μ, log σ²) → reparameterization → z∈ℝ² → transposed-conv decoder |

## Required Plots Generated

1. Original vs. reconstructed images (FC-AE)
2. Training/validation loss (FC-AE)
3. FC-AE vs. CAE reconstruction comparison
4. Clean vs. noisy vs. denoised images
5. Noise level vs. MSE/MAE/SSIM
6. VAE latent-space visualization (colored by digit)
7. Randomly generated VAE images
8. Latent-space interpolation
9. Training/validation loss (VAE)
10. Reconstruction-error histogram + top-5 high-error samples
11. Latent-dimension vs. MSE (additional study)

## License / Academic Note

This repository is coursework for CS3807 – Deep Learning Laboratory. All
numerical results should reflect your own execution before submission.
