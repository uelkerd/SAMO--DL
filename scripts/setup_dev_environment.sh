#!/bin/bash
# SAMO Deep Learning - Development Environment Setup
# Simple, working development environment setup

set -euo pipefail

echo "🚀 Setting up SAMO Development Environment..."

# Check Python version
echo "✅ Checking Python..."
python3 --version || { echo "❌ Python3 not found"; exit 1; }

# Install core API dependencies
echo "✅ Installing core dependencies (dev)..."
python3 -m pip install --user -r requirements-dev.txt || {
  echo "❌ Failed to install development dependencies. Please check your requirements files for conflicts.";
  exit 1;
}

# Add local bin to PATH (idempotent)
echo "✅ Setting up PATH..."
export PATH="$HOME/.local/bin:$PATH"
# Ensure bashrc exists to avoid grep failures under set -e
[ -f ~/.bashrc ] || touch ~/.bashrc
grep -qF "export PATH=\"$HOME/.local/bin:$PATH\"" ~/.bashrc || echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> ~/.bashrc

# Set PYTHONPATH (idempotent)
echo "✅ Setting up PYTHONPATH..."
WORKSPACE_PATH=$(pwd)
export PYTHONPATH="$WORKSPACE_PATH/src:$PYTHONPATH"
# Write the expanded workspace path; keep $PYTHONPATH literal for shells
grep -qF "export PYTHONPATH=\"$WORKSPACE_PATH/src:\$PYTHONPATH\"" ~/.bashrc || echo "export PYTHONPATH=\"$WORKSPACE_PATH/src:\$PYTHONPATH\"" >> ~/.bashrc

# Combined API import and health check (skips gracefully if FastAPI missing)
python3 - <<'PY'
import importlib.util

print("✅ Testing API import and health check...")

if importlib.util.find_spec('fastapi') is None:
    print('⚠️ FastAPI not installed; skipping API tests.')
else:
    from src.unified_ai_api import app  # noqa: F401
    print('✅ API imports successfully!')

    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200, f'Health check failed: {response.status_code}'
    print('✅ API health check passed!')
    data = response.json()
    print(f'Models status: {data.get("models", {})}')
PY

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Source your bashrc: source ~/.bashrc"
echo "  2. Test the API: python3 -c 'from src.unified_ai_api import app; print(\"✅ Working!\")'"
echo "  3. Install ML dependencies when needed (torch, transformers, etc.)"
echo ""
echo "💡 Note: Models show as 'unavailable' until you install ML dependencies"