import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Test configuration
    from fastapi.testclient import TestClient

    from src.unified_ai_api import app


"""
SAMO Deep Learning - Pytest Configuration and Shared Fixtures
Provides common test utilities, fixtures, and configuration.
"""

os.environ["TESTING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_journal_entry():
    """Provide sample journal entry for testing."""
    return {
        "text": "I had an amazing day today! I completed my machine learning project and felt so proud of my accomplishment. The weather was beautiful and I went for a walk in the park.",
        "user_id": 1,
        "is_private": False,
        "expected_emotions": ["joy", "pride", "admiration"],
    }


@pytest.fixture
def sample_audio_data():
    """Provide sample audio data for voice processing tests."""
    # Create a simple sine wave for testing

    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t)

    return {
        "audio_data": audio_data,
        "sample_rate": sample_rate,
        "expected_text": "test audio transcription",
    }


@pytest.fixture
def mock_bert_model():
    """Mock BERT model for testing without loading actual weights."""
    with patch("src.models.emotion_detection.bert_classifier.BertModel") as mock_model:
        mock_instance = Mock()
        mock_instance.config.hidden_size = 768
        mock_model.from_pretrained.return_value = mock_instance
        yield mock_model


@pytest.fixture
def mock_t5_model():
    """Mock T5 model for testing without loading actual weights."""
    with patch("src.models.summarization.t5_summarizer.T5ForConditionalGeneration") as mock_model:
        mock_instance = Mock()
        mock_model.from_pretrained.return_value = mock_instance
        yield mock_model


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing without loading actual weights."""
    with patch("src.models.voice_processing.whisper_transcriber.whisper") as mock_whisper:
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "test transcription"}
        mock_whisper.load_model.return_value = mock_model
        yield mock_whisper


@pytest.fixture(scope="session")
def cpu_device():
    """Ensure tests run on CPU regardless of GPU availability."""
    return torch.device("cpu")


@pytest.fixture
def api_client():
    """Provide FastAPI test client."""
    return TestClient(app)


# Custom markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
    config.addinivalue_line("markers", "model: marks tests that load ML models")


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available hardware."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
