#!/bin/bash

# Script to set up the Python virtual environment inside the Docker container
# Run this script INSIDE the container after starting it

set -e

echo "Setting up Python virtual environment 'unc'..."

# Check if we're in the workspace directory (repo root with pyproject.toml)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: This script should be run from the /workspace directory (repo root)"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "unc" ]; then
    echo "Removing existing virtual environment..."
    rm -rf unc
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv unc

# Activate virtual environment
source unc/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package + the JAX CUDA stack, editable.
# This mirrors the validated RTX 5090 (Blackwell, sm_120) host recipe.
echo "Installing conformal_human_motion_prediction with CUDA extras (editable)..."
pip install -e ".[cuda]"

# torch must be the cu128 build; the default PyPI wheel is cu130 and reports
# torch.cuda.is_available() == False on the sm_120 driver. The ".[cuda]" install above
# pulled the default (cu130) torch, so uninstall it first — otherwise pip sees torch
# "already satisfied" and silently skips the cu128 reinstall.
echo "Swapping in the cu128 torch/torchvision build..."
pip uninstall -y torch torchvision
# cu128 index ONLY — do NOT add --extra-index-url pypi here: pip would then pick the
# higher-versioned default (cu130) wheel from PyPI and undo the swap. torch's own deps
# are already present from the ".[cuda]" install above, so the bare index install resolves.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# Note: JAX/XLA needs a recent ptxas for sm_120a. The CUDA 12.8 base image ships one;
# if XLA still aborts with "PTX version ... does not support target 'sm_120a'", prepend a
# newer ptxas to PATH (see the repo's _cuda_ptxas_shim note in CLAUDE.md).

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source unc/bin/activate"
echo ""
echo "Or add this to your .bashrc in the container:"
echo "  echo 'source /workspace/unc/bin/activate' >> ~/.bashrc"
