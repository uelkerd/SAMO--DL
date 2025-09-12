"""Test imports for reorganized structure to ensure no breakage."""

import pytest

def test_core_imports():
    """Test core imports after reorganization."""
    from src.models.emotion.bert_classifier import BERTEmotionClassifier
    from src.models.summarization.t5_summarizer import T5Summarizer
    from src.models.voice.whisper_transcriber import WhisperTranscriber
    from src.inference.text_emotion_service import HFEmotionService
    from src.api_rate_limiter import TokenBucketRateLimiter
    from src.data.database import get_db
    from src.common.env import get_environment
    from scripts.bin.setup_environment import resolve_repo_path  # Test script imports
    print("All core imports successful after reorganization!")

def test_relative_imports():
    """Test relative imports within packages."""
    from src.models.emotion import bert_classifier
    from src.data import database
    from src.inference import text_emotion_service
    print("Relative imports working correctly!")

def test_script_paths():
    """Test that script paths are correct."""
    import sys
    from pathlib import Path
    repo_root = Path(".").resolve()
    assert repo_root / "configs" / "environment.yml"
    assert repo_root / "requirements" / "base.txt"
    assert repo_root / "scripts" / "bin" / "setup_environment.sh"
    print("File paths after reorganization verified!")

if __name__ == "__main__":
    test_core_imports()
    test_relative_imports()
    test_script_paths()