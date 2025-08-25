"""Test configuration with graceful dependency handling."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Graceful imports with fallbacks
try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠️  FastAPI TestClient not available - some tests will be skipped")
    TestClient = None
    FASTAPI_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - ML tests will be skipped")
    TORCH_AVAILABLE = False

try:
    from src.unified_ai_api import app
    API_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  API module not available: {e}")
    app = None
    API_AVAILABLE = False

# Test fixtures
@pytest.fixture
def client():
    if not FASTAPI_AVAILABLE or not API_AVAILABLE:
        pytest.skip("FastAPI or API module not available")
    return TestClient(app)

@pytest.fixture  
def mock_torch():
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch

# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API module"
    )
