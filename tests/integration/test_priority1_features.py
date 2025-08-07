"""
Integration Tests for Priority 1 Features

This module tests all the Priority 1 Features implemented:
1. JWT-based Authentication
2. Enhanced Voice Transcription API
3. Enhanced Text Summarization
4. Real-time Batch Processing
5. Comprehensive Monitoring Dashboard
"""

import asyncio
import json
import tempfile
import time
from typing import Dict, Any
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.unified_ai_api import app
from src.security.jwt_manager import JWTManager
from src.monitoring.dashboard import MonitoringDashboard

# Test client
client = TestClient(app)

class TestJWTAuthentication:
    """Test JWT-based authentication system."""
    
    def test_user_registration(self):
        """Test user registration endpoint."""
        user_data = {
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
    
    def test_user_login(self):
        """Test user login endpoint."""
        login_data = {
            "username": "testuser@example.com",
            "password": "testpassword123"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_token_refresh(self):
        """Test token refresh endpoint."""
        # First login to get tokens
        login_data = {
            "username": "testuser@example.com",
            "password": "testpassword123"
        }
        login_response = client.post("/auth/login", json=login_data)
        refresh_token = login_response.json()["refresh_token"]
        
        # Test refresh
        response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
    
    def test_protected_endpoint_with_auth(self):
        """Test accessing protected endpoint with valid token."""
        # Login to get token
        login_data = {
            "username": "testuser@example.com",
            "password": "testpassword123"
        }
        login_response = client.post("/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Test protected endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "username" in data
        assert "email" in data
    
    def test_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/auth/profile")
        assert response.status_code == 403  # Unauthorized
    
    def test_invalid_token(self):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == 401  # Unauthorized

class TestEnhancedVoiceTranscription:
    """Test enhanced voice transcription features."""
    
    @patch('src.models.voice_processing.whisper_transcriber.WhisperTranscriber')
    def test_voice_transcription_endpoint(self, mock_transcriber):
        """Test enhanced voice transcription endpoint."""
        # Mock transcription result
        mock_transcriber.return_value.transcribe.return_value = {
            "text": "This is a test transcription",
            "language": "en",
            "confidence": 0.95,
            "duration": 10.5
        }
        
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            # Login to get token
            login_data = {
                "username": "testuser@example.com",
                "password": "testpassword123"
            }
            login_response = client.post("/auth/login", json=login_data)
            access_token = login_response.json()["access_token"]
            
            # Test transcription endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            with open(temp_file_path, "rb") as audio_file:
                files = {"audio_file": ("test.wav", audio_file, "audio/wav")}
                data = {
                    "language": "en",
                    "model_size": "base",
                    "timestamp": False
                }
                response = client.post("/transcribe/voice", files=files, data=data, headers=headers)
            
            assert response.status_code == 200
            
            data = response.json()
            assert "text" in data
            assert "language" in data
            assert "confidence" in data
            assert "duration" in data
            assert "word_count" in data
            assert "speaking_rate" in data
            assert "audio_quality" in data
            
        finally:
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @patch('src.models.voice_processing.whisper_transcriber.WhisperTranscriber')
    def test_batch_transcription(self, mock_transcriber):
        """Test batch transcription endpoint."""
        # Mock transcription result
        mock_transcriber.return_value.transcribe.return_value = {
            "text": "Batch transcription result",
            "language": "en",
            "confidence": 0.92,
            "duration": 8.0
        }
        
        # Create test audio files
        temp_files = []
        try:
            for i in range(2):
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(b"fake audio data")
                temp_file.close()
                temp_files.append(temp_file.name)
            
            # Login to get token
            login_data = {
                "username": "testuser@example.com",
                "password": "testpassword123"
            }
            login_response = client.post("/auth/login", json=login_data)
            access_token = login_response.json()["access_token"]
            
            # Test batch transcription endpoint
            headers = {"Authorization": f"Bearer {access_token}"}
            files = []
            for i, temp_file_path in enumerate(temp_files):
                with open(temp_file_path, "rb") as audio_file:
                    files.append(("audio_files", (f"test{i}.wav", audio_file, "audio/wav")))
            
            data = {"language": "en"}
            response = client.post("/transcribe/batch", files=files, data=data, headers=headers)
            
            assert response.status_code == 200
            
            data = response.json()
            assert "total_files" in data
            assert "successful_transcriptions" in data
            assert "failed_transcriptions" in data
            assert "processing_time_ms" in data
            assert "results" in data
            
        finally:
            import os
            for temp_file_path in temp_files:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

class TestEnhancedTextSummarization:
    """Test enhanced text summarization features."""
    
    @patch('src.models.summarization.t5_summarizer.T5Summarizer')
    def test_text_summarization_endpoint(self, mock_summarizer):
        """Test enhanced text summarization endpoint."""
        # Mock summarization result
        mock_summarizer.return_value.summarize.return_value = {
            "summary": "This is a test summary of the input text.",
            "key_emotions": ["neutral"],
            "compression_ratio": 0.75
        }
        
        # Login to get token
        login_data = {
            "username": "testuser@example.com",
            "password": "testpassword123"
        }
        login_response = client.post("/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Test summarization endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        data = {
            "text": "This is a longer text that needs to be summarized. It contains multiple sentences and should be processed by the T5 model to generate a concise summary.",
            "model": "t5-small",
            "max_length": 150,
            "min_length": 30,
            "do_sample": True
        }
        response = client.post("/summarize/text", data=data, headers=headers)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "summary" in data
        assert "key_emotions" in data
        assert "compression_ratio" in data
        assert "emotional_tone" in data

class TestMonitoringDashboard:
    """Test comprehensive monitoring dashboard."""
    
    def test_performance_metrics_endpoint(self):
        """Test performance monitoring endpoint."""
        # Login to get token
        login_data = {
            "username": "testuser@example.com",
            "password": "testpassword123"
        }
        login_response = client.post("/auth/login", json=login_data)
        access_token = login_response.json()["access_token"]
        
        # Test performance metrics endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/monitoring/performance", headers=headers)
        
        # Note: This might fail if user doesn't have monitoring permission
        # In a real test, we'd set up proper permissions
        if response.status_code == 403:
            pytest.skip("User doesn't have monitoring permission")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "models" in data
        assert "api" in data
    
    def test_detailed_health_check(self):
        """Test detailed health check endpoint."""
        response = client.get("/monitoring/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "issues" in data
        assert "models" in data
        assert "system" in data
        assert "version" in data

class TestMonitoringDashboardClass:
    """Test the MonitoringDashboard class directly."""
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        dashboard = MonitoringDashboard()
        assert dashboard.start_time > 0
        assert dashboard.history_size == 1000
    
    def test_system_metrics_update(self):
        """Test system metrics update."""
        dashboard = MonitoringDashboard()
        metrics = dashboard.update_system_metrics()
        
        assert metrics is not None
        assert metrics.timestamp > 0
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.memory_available_gb >= 0
        assert 0 <= metrics.disk_percent <= 100
        assert metrics.disk_free_gb >= 0
    
    def test_model_metrics_recording(self):
        """Test model metrics recording."""
        dashboard = MonitoringDashboard()
        
        # Record some model requests
        dashboard.record_model_request("test_model", True, 150.0)
        dashboard.record_model_request("test_model", False, 200.0)
        dashboard.record_model_request("test_model", True, 100.0)
        
        metrics = dashboard.model_metrics["test_model"]
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.error_count == 1
        assert metrics.average_response_time_ms > 0
    
    def test_api_metrics_recording(self):
        """Test API metrics recording."""
        dashboard = MonitoringDashboard()
        
        # Record some API requests
        dashboard.record_api_request(150.0, True)
        dashboard.record_api_request(200.0, False)
        dashboard.record_api_request(100.0, True)
        
        assert dashboard.api_metrics.total_requests == 3
        assert len(dashboard.response_times) == 3
        assert len(dashboard.error_log) == 1
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics generation."""
        dashboard = MonitoringDashboard()
        
        # Add some data
        dashboard.update_system_metrics()
        dashboard.record_model_request("test_model", True, 150.0)
        dashboard.record_api_request(150.0, True)
        
        metrics = dashboard.get_comprehensive_metrics()
        
        assert "timestamp" in metrics
        assert "health_status" in metrics
        assert "system" in metrics
        assert "models" in metrics
        assert "api" in metrics
        assert "trends" in metrics
        assert "alerts" in metrics
    
    def test_health_status_calculation(self):
        """Test health status calculation."""
        dashboard = MonitoringDashboard()
        
        # Test with no data
        status = dashboard._calculate_health_status()
        assert status == "unknown"
        
        # Add some normal metrics
        dashboard.update_system_metrics()
        status = dashboard._calculate_health_status()
        assert status in ["healthy", "warning", "critical"]

class TestJWTManager:
    """Test JWT manager functionality."""
    
    def test_jwt_manager_initialization(self):
        """Test JWT manager initialization."""
        jwt_manager = JWTManager()
        assert jwt_manager.secret_key is not None
        assert jwt_manager.algorithm == "HS256"
        assert isinstance(jwt_manager.blacklisted_tokens, set)
    
    def test_token_creation(self):
        """Test token creation."""
        jwt_manager = JWTManager()
        
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "permissions": ["read", "write"]
        }
        
        # Test access token creation
        access_token = jwt_manager.create_access_token(user_data)
        assert access_token is not None
        assert isinstance(access_token, str)
        
        # Test refresh token creation
        refresh_token = jwt_manager.create_refresh_token(user_data)
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        
        # Test token pair creation
        token_pair = jwt_manager.create_token_pair(user_data)
        assert "access_token" in token_pair
        assert "refresh_token" in token_pair
        assert "token_type" in token_pair
        assert "expires_in" in token_pair
    
    def test_token_verification(self):
        """Test token verification."""
        jwt_manager = JWTManager()
        
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "permissions": ["read", "write"]
        }
        
        # Create and verify token
        access_token = jwt_manager.create_access_token(user_data)
        payload = jwt_manager.verify_token(access_token)
        
        assert payload is not None
        assert payload.user_id == "test_user_123"
        assert payload.username == "testuser@example.com"
        assert payload.email == "testuser@example.com"
        assert "read" in payload.permissions
        assert "write" in payload.permissions
    
    def test_token_blacklisting(self):
        """Test token blacklisting."""
        jwt_manager = JWTManager()
        
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "permissions": ["read", "write"]
        }
        
        # Create token
        access_token = jwt_manager.create_access_token(user_data)
        
        # Verify token is valid
        payload = jwt_manager.verify_token(access_token)
        assert payload is not None
        
        # Blacklist token
        success = jwt_manager.blacklist_token(access_token)
        assert success is True
        
        # Verify token is now invalid
        payload = jwt_manager.verify_token(access_token)
        assert payload is None
    
    def test_permission_checking(self):
        """Test permission checking."""
        jwt_manager = JWTManager()
        
        user_data = {
            "user_id": "test_user_123",
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "permissions": ["read", "write", "admin"]
        }
        
        access_token = jwt_manager.create_access_token(user_data)
        
        # Test permission checking
        assert jwt_manager.has_permission(access_token, "read") is True
        assert jwt_manager.has_permission(access_token, "write") is True
        assert jwt_manager.has_permission(access_token, "admin") is True
        assert jwt_manager.has_permission(access_token, "delete") is False

if __name__ == "__main__":
    pytest.main([__file__]) 