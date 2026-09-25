#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — sets up a local Python environment for Experiment 6
# (CS3807 Deep Learning Laboratory: RNN/LSTM/GRU + CNN-RNN video + Seq2Seq).
#
# Usage:
#   bash setup.sh
#   source venv/bin/activate
#   jupyter notebook Experiment_6_RNN_LSTM_GRU.ipynb
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="venv"
DATA_DIR="data"

echo "== Experiment 6 environment setup =="

# 1. Check Python version
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: $PYTHON_BIN not found. Install Python 3.10+ first." >&2
    exit 1
fi
echo "Using: $($PYTHON_BIN --version)"

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at ./$VENV_DIR — skipping creation."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3. Upgrade pip and install dependencies
echo "Installing dependencies from requirements.txt ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Create expected data directories (contents are gitignored)
mkdir -p "$DATA_DIR/UCI_HAR_Dataset" "$DATA_DIR/UCF101_subset"
echo "Created placeholder data directories under ./$DATA_DIR/"
echo "  -> See DATASET.md for how to populate them."

# 5. Register Jupyter kernel (optional, convenient for `jupyter notebook`)
python -m ipykernel install --user --name exp6-dl-lab --display-name "Experiment 6 (DL Lab)" \
    || echo "Note: kernel registration failed/skipped (not required to run the notebook)."

# 6. Check for a LaTeX engine (for building the report)
if command -v pdflatex &> /dev/null; then
    echo "pdflatex found: $(pdflatex --version | head -n1)"
else
    echo "Note: pdflatex not found. Install TeX Live (Linux/macOS) or MiKTeX (Windows) to"
    echo "      compile Experiment_6_Report.tex into a PDF."
fi

echo ""
echo "== Setup complete =="
echo "Activate the environment with:  source $VENV_DIR/bin/activate"
echo "Then launch:                     jupyter notebook Experiment_6_RNN_LSTM_GRU.ipynb"
echo "Dataset instructions:            see DATASET.md"
