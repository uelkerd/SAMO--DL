#!/bin/bash
# SAMO Deep Learning - Development Environment Setup
# Simple, working development environment setup

set -e

echo "🚀 Setting up SAMO Development Environment..."

# Check Python version
echo "✅ Checking Python..."
python3 --version || { echo "❌ Python3 not found"; exit 1; }

# Install core API dependencies
echo "✅ Installing core dependencies..."
pip3 install --user -r requirements.txt || {
  echo "⚠️ requirements.txt not found, installing minimal dev tools";
  pip3 install --user pytest ruff;
}

# Add local bin to PATH (idempotent)
echo "✅ Setting up PATH..."
export PATH="$HOME/.local/bin:$PATH"
grep -qF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc

# Set PYTHONPATH (idempotent)
echo "✅ Setting up PYTHONPATH..."
WORKSPACE_PATH=$(pwd)
export PYTHONPATH="$WORKSPACE_PATH/src:$PYTHONPATH"
grep -qF "export PYTHONPATH=\"$WORKSPACE_PATH/src:\$PYTHONPATH\"" ~/.bashrc || echo "export PYTHONPATH=\"\$WORKSPACE_PATH/src:\$PYTHONPATH\"" >> ~/.bashrc

# Test API import
echo "✅ Testing API import..."
python3 -c "
import sys
sys.path.insert(0, '$WORKSPACE_PATH/src')
from src.unified_ai_api import app
print('✅ API imports successfully!')
"

# Test API health check
echo "✅ Testing API health check..."
python3 -c "
import sys
sys.path.insert(0, '$WORKSPACE_PATH/src')
from src.unified_ai_api import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.get('/health')
assert response.status_code == 200, f'Health check failed: {response.status_code}'
print('✅ API health check passed!')
data = response.json()
print(f'Models status: {data.get(\"models\", {})}')
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Source your bashrc: source ~/.bashrc"
echo "  2. Test the API: python3 -c 'from src.unified_ai_api import app; print(\"✅ Working!\")'"
echo "  3. Install ML dependencies when needed (torch, transformers, etc.)"
echo ""
echo "💡 Note: Models show as 'unavailable' until you install ML dependencies"