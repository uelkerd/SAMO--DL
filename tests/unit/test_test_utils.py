#!/usr/bin/env python3
"""
Unit tests for test utility functions.

This module tests the utility functions in tests/test_utils.py
to ensure they work correctly and handle edge cases properly.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from tests.test_utils import (
    create_temp_audio_file,
    create_temp_json_file,
    create_sample_text_data,
    assert_dict_structure,
    cleanup_temp_files,
    create_mock_response
)


class TestCreateTempAudioFile:
    """Test audio file creation utility."""
    
    @staticmethod
    def test_create_temp_audio_file_success():
        """Test successful audio file creation."""
        temp_file = create_temp_audio_file(duration=1.0, sample_rate=8000)
        
        assert isinstance(temp_file, Path)
        assert temp_file.exists()
        assert temp_file.stat().st_size > 0
        
        # Cleanup
        temp_file.unlink()
    
    @staticmethod
    def test_create_temp_audio_file_invalid_duration():
        """Test error handling for invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            create_temp_audio_file(duration=-1.0, sample_rate=16000)
        
        with pytest.raises(ValueError, match="Duration must be positive"):
            create_temp_audio_file(duration=0.0, sample_rate=16000)
    
    @staticmethod
    def test_create_temp_audio_file_invalid_sample_rate():
        """Test error handling for invalid sample rate."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            create_temp_audio_file(duration=1.0, sample_rate=0)
        
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            create_temp_audio_file(duration=1.0, sample_rate=-8000)
    
    @staticmethod
    def test_create_temp_audio_file_cleanup_error():
        """Test cleanup error handling."""
        temp_file = create_temp_audio_file(duration=0.5, sample_rate=8000)
        
        # Simulate cleanup error by making file read-only
        temp_file.chmod(0o444)
        
        try:
            with pytest.raises(PermissionError):
                temp_file.unlink()
        finally:
            # Restore permissions and cleanup
            temp_file.chmod(0o644)
            temp_file.unlink()


class TestCreateTempJsonFile:
    """Test JSON file creation utility."""
    
    @staticmethod
    def test_create_temp_json_file_success():
        """Test successful JSON file creation."""
        test_data = {"key": "value", "number": 42}
        temp_file = create_temp_json_file(test_data)
        
        assert isinstance(temp_file, Path)
        assert temp_file.exists()
        
        # Verify content
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            assert loaded_data == test_data
        
        # Cleanup
        temp_file.unlink()
    
    @staticmethod
    def test_create_temp_json_file_empty_data():
        """Test JSON file creation with empty data."""
        temp_file = create_temp_json_file({})
        
        assert temp_file.exists()
        assert temp_file.stat().st_size > 0
        
        # Cleanup
        temp_file.unlink()


class TestCreateSampleTextData:
    """Test sample text data creation utility."""
    
    @staticmethod
    def test_create_sample_text_data_default():
        """Test default sample data creation."""
        data = create_sample_text_data()
        
        assert len(data) == 5
        assert all(isinstance(item, dict) for item in data)
        assert all('id' in item for item in data)
        assert all('text' in item for item in data)
        assert all('user_id' in item for item in data)
        assert all('is_private' in item for item in data)
    
    @staticmethod
    def test_create_sample_text_data_custom_count():
        """Test sample data creation with custom count."""
        data = create_sample_text_data(num_samples=3)
        
        assert len(data) == 3
        assert data[0]['id'] == 0
        assert data[1]['id'] == 1
        assert data[2]['id'] == 2
    
    @staticmethod
    def test_create_sample_text_data_zero_count():
        """Test sample data creation with zero count."""
        data = create_sample_text_data(num_samples=0)
        
        assert len(data) == 0
        assert isinstance(data, list)


class TestAssertDictStructure:
    """Test dictionary structure validation utility."""
    
    @staticmethod
    def test_assert_dict_structure_success():
        """Test successful structure validation."""
        data = {"a": 1, "b": 2, "c": 3}
        expected_keys = ["a", "b", "c"]
        
        # Should not raise any exception
        assert_dict_structure(data, expected_keys)
    
    @staticmethod
    def test_assert_dict_structure_missing_keys():
        """Test structure validation with missing keys."""
        data = {"a": 1, "b": 2}
        expected_keys = ["a", "b", "c"]
        
        with pytest.raises(AssertionError, match="Missing expected keys"):
            assert_dict_structure(data, expected_keys)
    
    @staticmethod
    def test_assert_dict_structure_extra_keys():
        """Test structure validation with extra keys."""
        data = {"a": 1, "b": 2, "c": 3, "d": 4}
        expected_keys = ["a", "b", "c"]
        
        with pytest.raises(AssertionError, match="Unexpected extra keys"):
            assert_dict_structure(data, expected_keys)
    
    @staticmethod
    def test_assert_dict_structure_empty_expected():
        """Test structure validation with empty expected keys."""
        data = {"a": 1, "b": 2}
        expected_keys = []
        
        with pytest.raises(AssertionError, match="Unexpected extra keys"):
            assert_dict_structure(data, expected_keys)


class TestCleanupTempFiles:
    """Test temporary file cleanup utility."""
    
    @staticmethod
    def test_cleanup_temp_files_success():
        """Test successful file cleanup."""
        # Create temporary files
        temp_files = []
        for _ in range(3):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_files.append(Path(temp_file.name))
            temp_file.close()
        
        # Verify files exist
        assert all(f.exists() for f in temp_files)
        
        # Cleanup
        cleanup_temp_files(temp_files)
        
        # Verify files are gone
        assert not any(f.exists() for f in temp_files)
    
    @staticmethod
    def test_cleanup_temp_files_nonexistent():
        """Test cleanup with nonexistent files."""
        nonexistent_files = [Path("/nonexistent/file1"), Path("/nonexistent/file2")]
        
        # Should not raise any exception
        cleanup_temp_files(nonexistent_files)
    
    @staticmethod
    def test_cleanup_temp_files_empty_list():
        """Test cleanup with empty list."""
        # Should not raise any exception
        cleanup_temp_files([])


class TestCreateMockResponse:
    """Test mock response creation utility."""
    
    @staticmethod
    def test_create_mock_response_default():
        """Test default mock response creation."""
        response = create_mock_response()
        
        assert response["status_code"] == 200
        assert response["data"] == {}
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["success"] is True
    
    @staticmethod
    def test_create_mock_response_custom_status():
        """Test mock response with custom status code."""
        response = create_mock_response(status_code=404)
        
        assert response["status_code"] == 404
        assert response["success"] is False
    
    @staticmethod
    def test_create_mock_response_custom_data():
        """Test mock response with custom data."""
        custom_data = {"error": "Not found"}
        response = create_mock_response(status_code=404, data=custom_data)
        
        assert response["data"] == custom_data
        assert response["success"] is False
    
    @staticmethod
    def test_create_mock_response_success_codes():
        """Test success flag for different status codes."""
        # Success codes
        assert create_mock_response(200)["success"] is True
        assert create_mock_response(201)["success"] is True
        assert create_mock_response(299)["success"] is True
        
        # Error codes
        assert create_mock_response(400)["success"] is False
        assert create_mock_response(500)["success"] is False
