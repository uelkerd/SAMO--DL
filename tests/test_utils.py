#!/usr/bin/env python3
"""
Simple Test Utilities for SAMO Deep Learning

This module provides minimal, focused utilities for test files.
Keeps scope small and focused on common testing tasks.
"""

import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def create_temp_audio_file(duration: float = 2.0, sample_rate: int = 16000) -> Path:
    """Create a temporary audio file for testing.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Path to temporary audio file
    """
    # Create simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(440 * 2 * np.pi * t)  # 440 Hz tone
    
    # Save as WAV file
    import wave
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    return Path(temp_file.name)


def create_temp_json_file(data: Dict[str, Any]) -> Path:
    """Create a temporary JSON file for testing.
    
    Args:
        data: Data to write to JSON file
        
    Returns:
        Path to temporary JSON file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w')
    json.dump(data, temp_file)
    temp_file.close()
    return Path(temp_file.name)


def create_sample_text_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Create sample text data for testing.
    
    Args:
        num_samples: Number of sample entries to create
        
    Returns:
        List of sample text entries
    """
    sample_texts = [
        "I had an amazing day today! I completed my machine learning project.",
        "The weather was beautiful and I went for a walk in the park.",
        "I'm feeling grateful for all the support I've received.",
        "This is a challenging problem that requires careful analysis.",
        "The results exceeded my expectations and I'm very satisfied."
    ]
    
    return [
        {
            "id": i,
            "text": sample_texts[i % len(sample_texts)],
            "user_id": i + 1,
            "is_private": i % 2 == 0
        }
        for i in range(num_samples)
    ]


def assert_dict_structure(data: Dict[str, Any], expected_keys: List[str]) -> None:
    """Assert that a dictionary has the expected structure.
    
    Args:
        data: Dictionary to check
        expected_keys: List of expected keys
        
    Raises:
        AssertionError: If structure doesn't match
    """
    missing_keys = set(expected_keys) - set(data.keys())
    extra_keys = set(data.keys()) - set(expected_keys)
    
    if missing_keys:
        raise AssertionError(f"Missing expected keys: {missing_keys}")
    
    if extra_keys:
        raise AssertionError(f"Unexpected extra keys: {extra_keys}")


def cleanup_temp_files(file_paths: List[Path]) -> None:
    """Clean up temporary test files.
    
    Args:
        file_paths: List of file paths to clean up
    """
    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors in tests


def create_mock_response(status_code: int = 200, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a mock API response for testing.
    
    Args:
        status_code: HTTP status code
        data: Response data
        
    Returns:
        Mock response dictionary
    """
    return {
        "status_code": status_code,
        "data": data or {},
        "headers": {"Content-Type": "application/json"},
        "success": 200 <= status_code < 300
    }
