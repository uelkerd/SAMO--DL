#!/bin/bash

# Setup conda environment for CircleCI
# This script ensures conda is properly initialized and activated

set -e

echo "🔧 Setting up conda environment..."

# Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda" ]; then
    echo "📦 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
fi

# Add miniconda to PATH
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda for bash
conda init bash

# Source bashrc to load conda
source ~/.bashrc

# Create environment if it doesn't exist
if ! conda env list | grep -q "samo-dl-stable"; then
    echo "🏗️ Creating conda environment..."
    conda env create -f environment.yml
fi

# Activate environment
conda activate samo-dl-stable

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install -e ".[test,dev,prod]"
pip install httpx python-multipart psycopg2-binary

echo "✅ Conda environment setup complete!"
echo "Current Python: $(which python)"
echo "Current conda env: $CONDA_DEFAULT_ENV" 