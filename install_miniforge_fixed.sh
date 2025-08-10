#!/bin/bash

# Fixed Miniforge Installation Script
# This script demonstrates the corrected approach to downloading and verifying Miniforge

set -euxo pipefail

echo "🚀 Installing Miniforge with corrected checksum verification..."

# Download Miniforge installer
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

if command -v wget >/dev/null 2>&1; then
    wget -O miniconda.sh "$MINIFORGE_URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o miniconda.sh "$MINIFORGE_URL"
else
    echo "Error: neither wget nor curl available to download Miniforge." >&2
    exit 1
fi

# Get the installer basename and latest release info
BASENAME=$(basename "$MINIFORGE_URL")
curl -sSfL https://api.github.com/repos/conda-forge/miniforge/releases/latest -o latest_release.json
TAG=$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' latest_release.json | head -n1)
rm -f latest_release.json

# FIXED: Use the individual checksum file for the specific installer
CHECKSUM_URL="https://github.com/conda-forge/miniforge/releases/download/$TAG/Miniforge3-$TAG-Linux-x86_64.sh.sha256"

echo "Resolved installer: $BASENAME"
echo "Checksum URL: $CHECKSUM_URL"

# Download the checksum file
if command -v curl >/dev/null 2>&1; then
    curl -sSfL "$CHECKSUM_URL" -o miniconda.sha256sum
elif command -v wget >/dev/null 2>&1; then
    wget -qO miniconda.sha256sum "$CHECKSUM_URL"
else
    echo "Warning: neither curl nor wget available to fetch checksum; aborting for safety." >&2
    exit 1
fi

# Verify checksum
ACTUAL_SUM=$(sha256sum miniconda.sh | awk '{print $1}')
EXPECTED_SUM=$(cat miniconda.sha256sum | awk '{print $1}')

echo "Actual checksum:   $ACTUAL_SUM"
echo "Expected checksum: $EXPECTED_SUM"

if [ -z "$EXPECTED_SUM" ]; then
    echo "❌ Checksum file empty or unavailable. Aborting to be safe." >&2
    rm -f miniconda.sh miniconda.sha256sum
    exit 1
fi

if [ "$ACTUAL_SUM" != "$EXPECTED_SUM" ]; then
    echo "❌ Checksum verification failed! Expected $EXPECTED_SUM, got $ACTUAL_SUM" >&2
    rm -f miniconda.sh miniconda.sha256sum
    exit 1
fi

echo "✅ Checksum verification successful!"
rm -f miniconda.sha256sum

# Install Miniforge
chmod +x miniconda.sh
bash miniconda.sh -b -p "$HOME/miniforge3"
rm -f miniconda.sh

echo "✅ Miniforge installation completed successfully!"
echo "📍 Installed at: $HOME/miniforge3"
echo "🔧 To use: export PATH=\"\$HOME/miniforge3/bin:\$PATH\""