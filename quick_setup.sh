#!/bin/bash
# SAMO-DL Quick Setup Script
set -e

echo "🚀 SAMO-DL Infrastructure Setup"
echo "================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $python_version"

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install core dependencies
echo "📦 Installing core dependencies..."
if [ -f "requirements/requirements.txt" ]; then
    python3 -m pip install -r requirements/requirements.txt
elif [ -f "dependencies/requirements_unified.txt" ]; then
    python3 -m pip install -r dependencies/requirements_unified.txt  
else
    echo "❌ No requirements file found"
    exit 1
fi

# Install development dependencies (optional)
if [ -f "requirements/requirements-dev.txt" ]; then
    echo "📦 Installing development dependencies..."
    python3 -m pip install -r requirements/requirements-dev.txt
fi

# Install ML dependencies (optional)
if [ "$1" = "--ml" ] && [ -f "requirements/requirements-ml.txt" ]; then
    echo "🧠 Installing ML dependencies..."
    python3 -m pip install -r requirements/requirements-ml.txt
fi

# Verify installation
echo "🧪 Verifying installation..."
python3 -c "import fastapi, uvicorn, pydantic; print('✅ Core dependencies OK')"

if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch available"
else
    echo "⚠️  PyTorch not installed (use --ml flag for ML dependencies)"
fi

echo ""
echo "🎉 Setup complete!"
echo "💡 To run tests: python -m pytest tests/"
echo "💡 To start API: uvicorn src.unified_ai_api:app --reload"
