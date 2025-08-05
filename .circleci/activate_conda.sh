#!/bin/bash

# Activate conda environment for CircleCI steps
# This script should be sourced in each step that needs conda

set -e

echo "🔧 Activating conda environment..."

# Add miniconda to PATH
export PATH="$HOME/miniconda/bin:$PATH"

# Source bashrc to load conda
source ~/.bashrc

# Activate environment
conda activate samo-dl-stable

echo "✅ Conda environment activated!"
echo "Current Python: $(which python)"
echo "Current conda env: $CONDA_DEFAULT_ENV" 