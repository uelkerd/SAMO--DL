#!/bin/bash
#
# 🚀 SAMO Deep Learning - Environment Rebuild Script
# This script automates the steps from docs/summaries/environment-crisis-resolution.md
# to create a fresh, stable Conda environment.
#

set -e # Exit immediately if a command exits with a non-zero status.

ENV_NAME="samo-dl-stable"
PYTHON_VERSION="3.11" # Using a stable, recent version

echo "🚨 Deactivating any existing Conda environment..."
conda deactivate || echo "No active environment to deactivate."

echo "🔥 Removing old environment '$ENV_NAME' if it exists..."
conda env remove -n "$ENV_NAME" || echo "Environment '$ENV_NAME' not found, creating new."

echo "🐍 Creating fresh Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "✅ Activating new environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "📦 Installing all project dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🎉 Environment rebuild complete! You are now in the '${ENV_NAME}' environment."
echo "👉 To activate in a new terminal, run: conda activate ${ENV_NAME}"
