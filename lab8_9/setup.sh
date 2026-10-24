#!/usr/bin/env bash
# =============================================================================
# Automated Setup and Execution Script for Experiments 8 & 9
# CS3807 -- Deep Learning Laboratory (Shiv Nadar University Chennai)
# Student: CY Pranav Menon (24011101014)
# =============================================================================

set -e

echo "=== [1/5] Setting up Python Environment & Installing Dependencies ==="
pip install -r requirements.txt

echo "=== [2/5] Downloading and Unpacking TensorFlow Flowers Dataset ==="
mkdir -p dataset
if [ ! -f "dataset/flower_photos.tgz" ]; then
    echo "Downloading flower_photos.tgz..."
    curl -o dataset/flower_photos.tgz https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
fi

if [ ! -d "dataset/flower_photos" ]; then
    echo "Extracting dataset archive..."
    tar -xzf dataset/flower_photos.tgz -C dataset/
fi

echo "=== [3/5] Running Attention Ablation Experiments & Generating Metrics ==="
python run_experiments.py
python generate_remaining_plots.py

echo "=== [4/5] Generating Complete Jupyter Notebook ==="
python make_notebook.py

echo "=== [5/5] Compiling LaTeX Report with Tectonic ==="
if command -v tectonic &> /dev/null; then
    tectonic report.tex
    cp report.pdf CS3807_Experiments_8_9_CY_Pranav_Menon_24011101014.pdf
    echo "Report compiled successfully: CS3807_Experiments_8_9_CY_Pranav_Menon_24011101014.pdf"
elif command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode report.tex
    pdflatex -interaction=nonstopmode report.tex
    cp report.pdf CS3807_Experiments_8_9_CY_Pranav_Menon_24011101014.pdf
    echo "Report compiled successfully with pdflatex."
else
    echo "Notice: tectonic or pdflatex not found in PATH. You can compile report.tex on Overleaf or local LaTeX installation."
fi

echo "=== All tasks completed successfully! ==="
