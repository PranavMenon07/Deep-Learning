#!/usr/bin/env bash
# Experiment 7 — local environment setup
# Usage: chmod +x setup.sh && ./setup.sh

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "==> Using $($PYTHON_BIN --version)"

echo "==> Creating virtual environment in ${VENV_DIR}/"
$PYTHON_BIN -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r requirements.txt

echo "==> Creating output folder for figures (lab7/), used by the LaTeX report"
mkdir -p lab7

echo ""
echo "Setup complete."
echo "Activate the environment with:  source ${VENV_DIR}/bin/activate"
echo "Launch the notebook with:       jupyter notebook Experiment7_Autoencoders_CAE_DenoisingAE_VAE.ipynb"
