# 🧪 Testing Framework Guide

Welcome to the SAMO Brain Testing Framework! This guide covers comprehensive testing strategies, from unit tests to end-to-end validation, ensuring our AI system maintains high quality and reliability.

## 🚀 **Quick Start (5 minutes)**

### **Run All Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-asyncio pytest-xdist

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term
```

### **Test Structure Overview**
```
tests/
├── unit/                    # Unit tests (fast, isolated)
├── integration/             # Integration tests (API, database)
├── e2e/                     # End-to-end tests (complete workflows)
├── performance/             # Performance and load tests
├── conftest.py             # Shared fixtures and configuration
└── fixtures/               # Test data and models
```

---

## 🔬 **Unit Testing**

### **Core Testing Principles**

**Test Structure (AAA Pattern):**
```python
def test_emotion_detection_happy_case():
    # Arrange - Set up test data and dependencies
    text = "I am feeling happy today!"
    model = EmotionDetectionModel()
    
    # Act - Execute the function being tested
    result = model.predict(text)
    
    # Assert - Verify the expected outcome
    assert result['predicted_emotion'] == 'happy'
    assert result['confidence'] > 0.8
    assert 'probabilities' in result
```

### **Model Testing**

```python
# tests/unit/test_emotion_detection.py
import pytest
import torch
from unittest.mock import Mock, patch
from src.models.emotion_detection.bert_classifier import EmotionDetectionModel

class TestEmotionDetectionModel:
    """Test suite for emotion detection model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return EmotionDetectionModel()
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "I am feeling happy today!",
            "I feel sad about the news",
            "I am excited for the party"
        ]
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'model')
        assert model.device in ['cpu', 'cuda']
    
    def test_text_preprocessing(self, model, sample_texts):
        """Test text preprocessing."""
        for text in sample_texts:
            processed = model._preprocess_text(text)
            assert isinstance(processed, str)
            assert len(processed) > 0
            assert processed.strip() == processed
    
    @patch('torch.nn.functional.softmax')
    @patch('torch.no_grad')
    def test_prediction_pipeline(self, mock_no_grad, mock_softmax, model, sample_texts):
        """Test complete prediction pipeline."""
        # Mock model outputs
        mock_softmax.return_value = torch.tensor([[0.8, 0.1, 0.1]])
        
        for text in sample_texts:
            result = model.predict(text)
            
            # Verify result structure
            assert isinstance(result, dict)
            assert 'predicted_emotion' in result
            assert 'confidence' in result
            assert 'probabilities' in result
            assert 'prediction_time_ms' in result
            
            # Verify data types
            assert isinstance(result['predicted_emotion'], str)
            assert isinstance(result['confidence'], float)
            assert isinstance(result['probabilities'], dict)
            assert isinstance(result['prediction_time_ms'], (int, float))
            
            # Verify value ranges
            assert 0 <= result['confidence'] <= 1
            assert result['prediction_time_ms'] >= 0
    
    def test_confidence_calculation(self, model):
        """Test confidence calculation logic."""
        probabilities = {
            'happy': 0.8,
            'sad': 0.1,
            'excited': 0.1
        }
        
        confidence = model._calculate_confidence(probabilities)
        assert confidence == 0.8
        assert 0 <= confidence <= 1
    
    def test_emotion_mapping(self, model):
        """Test emotion label mapping."""
        # Test valid emotion indices
        for i, emotion in enumerate(model.emotion_labels):
            mapped = model._map_index_to_emotion(i)
            assert mapped == emotion
        
        # Test invalid index
        with pytest.raises(ValueError):
            model._map_index_to_emotion(999)
    
    @pytest.mark.parametrize("text,expected_emotion", [
        ("I am feeling happy today!", "happy"),
        ("I feel sad about the news", "sad"),
        ("I am excited for the party", "excited"),
        ("I feel anxious about the presentation", "anxious"),
        ("I am grateful for your help", "grateful")
    ])
    def test_emotion_detection_accuracy(self, model, text, expected_emotion):
        """Test emotion detection accuracy with parametrized inputs."""
        result = model.predict(text)
        assert result['predicted_emotion'] == expected_emotion
    
    def test_error_handling(self, model):
        """Test error handling for invalid inputs."""
        # Test empty text
        with pytest.raises(ValueError):
            model.predict("")
        
        # Test None input
        with pytest.raises(ValueError):
            model.predict(None)
        
        # Test very long text
        long_text = "This is a very long text " * 1000
        with pytest.raises(ValueError):
            model.predict(long_text)
```

### **API Testing**

```python
# tests/unit/test_api_models.py
import pytest
from pydantic import ValidationError
from src.api_models import PredictionRequest, PredictionResponse, ErrorResponse

class TestPredictionRequest:
    """Test suite for prediction request model."""
    
    def test_valid_request(self):
        """Test valid prediction request."""
        request = PredictionRequest(text="I am feeling happy!")
        assert request.text == "I am feeling happy!"
        assert request.threshold == 0.5  # default value
    
    def test_request_with_custom_threshold(self):
        """Test request with custom threshold."""
        request = PredictionRequest(text="I am feeling happy!", threshold=0.8)
        assert request.threshold == 0.8
    
    def test_invalid_text(self):
        """Test invalid text input."""
        with pytest.raises(ValidationError):
            PredictionRequest(text="")
        
        with pytest.raises(ValidationError):
            PredictionRequest(text=None)
    
    def test_invalid_threshold(self):
        """Test invalid threshold values."""
        with pytest.raises(ValidationError):
            PredictionRequest(text="test", threshold=1.5)
        
        with pytest.raises(ValidationError):
            PredictionRequest(text="test", threshold=-0.1)

class TestPredictionResponse:
    """Test suite for prediction response model."""
    
    def test_valid_response(self):
        """Test valid prediction response."""
        response = PredictionResponse(
            predicted_emotion="happy",
            confidence=0.95,
            probabilities={"happy": 0.95, "sad": 0.05},
            prediction_time_ms=150,
            model_version="1.0.0"
        )
        
        assert response.predicted_emotion == "happy"
        assert response.confidence == 0.95
        assert response.prediction_time_ms == 150
        assert response.model_version == "1.0.0"
    
    def test_confidence_validation(self):
        """Test confidence value validation."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_emotion="happy",
                confidence=1.5,  # Invalid confidence
                probabilities={"happy": 0.95},
                prediction_time_ms=150,
                model_version="1.0.0"
            )
```

### **Rate Limiter Testing**

```python
# tests/unit/test_api_rate_limiter.py
import pytest
import time
from unittest.mock import Mock
from src.api_rate_limiter import APIRateLimiter

class TestAPIRateLimiter:
    """Test suite for API rate limiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        return APIRateLimiter(max_requests=10, window_seconds=60)
    
    def test_initial_state(self, rate_limiter):
        """Test initial rate limiter state."""
        assert rate_limiter.max_requests == 10
        assert rate_limiter.window_seconds == 60
        assert len(rate_limiter.requests) == 0
    
    def test_allow_request(self, rate_limiter):
        """Test allowing requests within limit."""
        client_id = "test_client"
        
        # Allow multiple requests
        for i in range(10):
            assert rate_limiter.allow_request(client_id) is True
        
        # 11th request should be blocked
        assert rate_limiter.allow_request(client_id) is False
    
    def test_window_reset(self, rate_limiter):
        """Test rate limit window reset."""
        client_id = "test_client"
        
        # Use all requests
        for _ in range(10):
            rate_limiter.allow_request(client_id)
        
        # Wait for window to reset
        time.sleep(1)  # In real tests, mock time
        
        # Should allow requests again
        assert rate_limiter.allow_request(client_id) is True
    
    def test_multiple_clients(self, rate_limiter):
        """Test rate limiting for multiple clients."""
        client_1 = "client_1"
        client_2 = "client_2"
        
        # Each client should have their own limit
        for _ in range(10):
            assert rate_limiter.allow_request(client_1) is True
            assert rate_limiter.allow_request(client_2) is True
        
        # Both should be blocked
        assert rate_limiter.allow_request(client_1) is False
        assert rate_limiter.allow_request(client_2) is False
    
    def test_cleanup_old_requests(self, rate_limiter):
        """Test cleanup of old requests."""
        client_id = "test_client"
        
        # Add old requests
        old_time = time.time() - 120  # 2 minutes ago
        rate_limiter.requests[client_id] = [old_time] * 5
        
        # Cleanup should remove old requests
        rate_limiter._cleanup_old_requests(client_id)
        
        assert len(rate_limiter.requests[client_id]) == 0
```

---

## 🔗 **Integration Testing**

### **API Endpoint Testing**

```python
# tests/integration/test_api_endpoints.py
import pytest
import requests
from fastapi.testclient import TestClient
from src.unified_ai_api import app

class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_request(self):
        """Sample prediction request."""
        return {
            "text": "I am feeling happy today!",
            "threshold": 0.5
        }
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime" in data
    
    def test_predict_endpoint_success(self, client, sample_request):
        """Test successful prediction endpoint."""
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_emotion" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "prediction_time_ms" in data
        assert "model_version" in data
        
        # Verify data types
        assert isinstance(data["predicted_emotion"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["prediction_time_ms"], (int, float))
    
    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input."""
        # Empty text
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
        
        # Missing text
        response = client.post("/predict", json={"threshold": 0.5})
        assert response.status_code == 422
        
        # Invalid threshold
        response = client.post("/predict", json={
            "text": "test",
            "threshold": 1.5
        })
        assert response.status_code == 422
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "server_metrics" in data
        assert "emotion_distribution" in data
        
        # Verify server metrics structure
        server_metrics = data["server_metrics"]
        assert "uptime_seconds" in server_metrics
        assert "total_requests" in server_metrics
        assert "successful_requests" in server_metrics
        assert "failed_requests" in server_metrics
        assert "success_rate" in server_metrics
        assert "average_response_time_ms" in server_metrics
    
    def test_rate_limiting(self, client, sample_request):
        """Test API rate limiting."""
        # Make multiple requests quickly
        responses = []
        for _ in range(15):  # Exceed rate limit
            response = client.post("/predict", json=sample_request)
            responses.append(response)
        
        # Check that some requests were rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes  # Too Many Requests
    
    def test_concurrent_requests(self, client, sample_request):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/predict", json=sample_request)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests were processed
        assert len(results) == 10
        assert len(errors) == 0
        assert all(code in [200, 429] for code in results)  # Success or rate limited
```

### **Database Integration Testing**

```python
# tests/integration/test_database.py
import pytest
import sqlite3
from pathlib import Path
from src.data.database import DatabaseManager

class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def db_manager(self, temp_dir):
        """Create database manager with temporary database."""
        db_path = temp_dir / "test.db"
        return DatabaseManager(db_path)
    
    @pytest.fixture
    def sample_prediction_data(self):
        """Sample prediction data for testing."""
        return {
            "text": "I am feeling happy today!",
            "predicted_emotion": "happy",
            "confidence": 0.95,
            "prediction_time_ms": 150,
            "model_version": "1.0.0"
        }
    
    def test_database_initialization(self, db_manager):
        """Test database initialization."""
        db_manager.initialize()
        
        # Check if tables exist
        tables = db_manager.get_tables()
        assert "predictions" in tables
        assert "metrics" in tables
    
    def test_prediction_storage(self, db_manager, sample_prediction_data):
        """Test storing prediction data."""
        db_manager.initialize()
        
        # Store prediction
        prediction_id = db_manager.store_prediction(sample_prediction_data)
        assert prediction_id is not None
        
        # Retrieve prediction
        stored_prediction = db_manager.get_prediction(prediction_id)
        assert stored_prediction is not None
        assert stored_prediction["text"] == sample_prediction_data["text"]
        assert stored_prediction["predicted_emotion"] == sample_prediction_data["predicted_emotion"]
    
    def test_metrics_storage(self, db_manager):
        """Test storing metrics data."""
        db_manager.initialize()
        
        metrics_data = {
            "timestamp": "2024-01-01T00:00:00Z",
            "total_requests": 100,
            "successful_requests": 95,
            "failed_requests": 5,
            "success_rate": 0.95,
            "average_response_time_ms": 150.5
        }
        
        # Store metrics
        metrics_id = db_manager.store_metrics(metrics_data)
        assert metrics_id is not None
        
        # Retrieve metrics
        stored_metrics = db_manager.get_metrics(metrics_id)
        assert stored_metrics is not None
        assert stored_metrics["total_requests"] == 100
        assert stored_metrics["success_rate"] == 0.95
    
    def test_data_retrieval(self, db_manager, sample_prediction_data):
        """Test retrieving stored data."""
        db_manager.initialize()
        
        # Store multiple predictions
        prediction_ids = []
        for i in range(5):
            data = sample_prediction_data.copy()
            data["text"] = f"Test text {i}"
            prediction_id = db_manager.store_prediction(data)
            prediction_ids.append(prediction_id)
        
        # Retrieve all predictions
        all_predictions = db_manager.get_all_predictions()
        assert len(all_predictions) == 5
        
        # Retrieve predictions by emotion
        happy_predictions = db_manager.get_predictions_by_emotion("happy")
        assert len(happy_predictions) == 5
    
    def test_database_cleanup(self, db_manager, sample_prediction_data):
        """Test database cleanup operations."""
        db_manager.initialize()
        
        # Store some data
        for i in range(10):
            data = sample_prediction_data.copy()
            data["text"] = f"Test text {i}"
            db_manager.store_prediction(data)
        
        # Clean up old data
        deleted_count = db_manager.cleanup_old_data(days=0)  # Delete all
        assert deleted_count == 10
        
        # Verify data is gone
        remaining_predictions = db_manager.get_all_predictions()
        assert len(remaining_predictions) == 0
```

---

## 🔄 **End-to-End Testing**

### **Complete Workflow Testing**

```python
# tests/e2e/test_complete_workflows.py
import pytest
import requests
import time
from pathlib import Path

class TestCompleteWorkflows:
    """End-to-end tests for complete workflows."""
    
    @pytest.fixture
    def api_url(self):
        """API base URL."""
        return "http://localhost:8000"
    
    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        return [
            "I am feeling happy today!",
            "I feel sad about the news",
            "I am excited for the party",
            "I feel anxious about the presentation",
            "I am grateful for your help"
        ]
    
    def test_complete_prediction_workflow(self, api_url, sample_dataset):
        """Test complete prediction workflow."""
        results = []
        
        # Make predictions for all samples
        for text in sample_dataset:
            response = requests.post(
                f"{api_url}/predict",
                json={"text": text, "threshold": 0.5}
            )
            assert response.status_code == 200
            
            result = response.json()
            results.append(result)
            
            # Verify result structure
            assert "predicted_emotion" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert "prediction_time_ms" in result
            assert "model_version" in result
        
        # Verify all predictions are different emotions
        emotions = [r["predicted_emotion"] for r in results]
        assert len(set(emotions)) > 1  # Should have variety
        
        # Check confidence levels
        confidences = [r["confidence"] for r in results]
        assert all(0 <= conf <= 1 for conf in confidences)
        assert any(conf > 0.8 for conf in confidences)  # Some high confidence
    
    def test_metrics_collection_workflow(self, api_url, sample_dataset):
        """Test metrics collection workflow."""
        # Make some requests
        for text in sample_dataset:
            requests.post(f"{api_url}/predict", json={"text": text})
        
        # Wait for metrics to update
        time.sleep(2)
        
        # Get metrics
        response = requests.get(f"{api_url}/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # Verify metrics structure
        assert "server_metrics" in metrics
        assert "emotion_distribution" in metrics
        
        server_metrics = metrics["server_metrics"]
        assert server_metrics["total_requests"] >= len(sample_dataset)
        assert server_metrics["success_rate"] > 0.8
        assert server_metrics["average_response_time_ms"] > 0
    
    def test_rate_limiting_workflow(self, api_url):
        """Test rate limiting workflow."""
        # Make requests quickly to trigger rate limiting
        responses = []
        for i in range(20):
            response = requests.post(
                f"{api_url}/predict",
                json={"text": f"Test text {i}"}
            )
            responses.append(response)
        
        # Check rate limiting behavior
        status_codes = [r.status_code for r in responses]
        assert 200 in status_codes  # Some successful
        assert 429 in status_codes  # Some rate limited
        
        # Wait and try again
        time.sleep(2)
        response = requests.post(
            f"{api_url}/predict",
            json={"text": "Test after wait"}
        )
        assert response.status_code == 200
    
    def test_error_handling_workflow(self, api_url):
        """Test error handling workflow."""
        # Test various error conditions
        error_cases = [
            {"text": ""},  # Empty text
            {"text": "a" * 10000},  # Very long text
            {"threshold": 1.5},  # Invalid threshold
            {},  # Missing text
        ]
        
        for error_case in error_cases:
            response = requests.post(f"{api_url}/predict", json=error_case)
            assert response.status_code in [422, 400]  # Validation error
    
    def test_health_monitoring_workflow(self, api_url):
        """Test health monitoring workflow."""
        # Check health endpoint
        response = requests.get(f"{api_url}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "uptime" in health_data
        
        # Make some requests and check health again
        for i in range(5):
            requests.post(f"{api_url}/predict", json={"text": f"Health test {i}"})
        
        response = requests.get(f"{api_url}/health")
        assert response.status_code == 200
        
        # Health should still be good
        health_data = response.json()
        assert health_data["status"] == "healthy"
    
    def test_model_version_workflow(self, api_url, sample_dataset):
        """Test model version tracking workflow."""
        versions = set()
        
        # Make predictions and collect model versions
        for text in sample_dataset:
            response = requests.post(
                f"{api_url}/predict",
                json={"text": text}
            )
            assert response.status_code == 200
            
            result = response.json()
            versions.add(result["model_version"])
        
        # Should have consistent model version
        assert len(versions) == 1
        assert list(versions)[0] is not None
```

---

## ⚡ **Performance Testing**

### **Load Testing**

```python
# tests/performance/test_load.py
import pytest
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class TestLoadPerformance:
    """Performance and load testing."""
    
    @pytest.fixture
    def api_url(self):
        """API base URL."""
        return "http://localhost:8000"
    
    def test_single_request_performance(self, api_url):
        """Test single request performance."""
        start_time = time.time()
        
        response = requests.post(
            f"{api_url}/predict",
            json={"text": "I am feeling happy today!"}
        )
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert response_time < 1000  # Should respond within 1 second
        
        result = response.json()
        assert result["prediction_time_ms"] < 500  # Model inference < 500ms
    
    def test_concurrent_requests(self, api_url):
        """Test concurrent request handling."""
        def make_request(request_id):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": f"Concurrent test {request_id}"}
                )
                return {"id": request_id, "status": response.status_code, "time": time.time()}
            except Exception as e:
                return {"id": request_id, "error": str(e)}
        
        # Make 50 concurrent requests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r.get("status") == 200]
        failed_requests = [r for r in results if "error" in r]
        rate_limited_requests = [r for r in results if r.get("status") == 429]
        
        # Performance assertions
        assert len(successful_requests) >= 30  # At least 60% success
        assert total_time < 30  # Complete within 30 seconds
        assert len(failed_requests) < 5  # Few failures
        
        print(f"Performance Results:")
        print(f"  Total requests: 50")
        print(f"  Successful: {len(successful_requests)}")
        print(f"  Rate limited: {len(rate_limited_requests)}")
        print(f"  Failed: {len(failed_requests)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests per second: {50/total_time:.2f}")
    
    def test_sustained_load(self, api_url):
        """Test sustained load over time."""
        def make_request():
            response = requests.post(
                f"{api_url}/predict",
                json={"text": "Sustained load test"}
            )
            return response.status_code
        
        # Make requests for 60 seconds
        start_time = time.time()
        request_count = 0
        successful_count = 0
        
        while time.time() - start_time < 60:
            status_code = make_request()
            request_count += 1
            if status_code == 200:
                successful_count += 1
            time.sleep(0.1)  # 10 requests per second
        
        # Calculate metrics
        total_time = time.time() - start_time
        success_rate = successful_count / request_count
        requests_per_second = request_count / total_time
        
        # Performance assertions
        assert success_rate > 0.8  # 80% success rate
        assert requests_per_second > 5  # At least 5 RPS
        assert total_time >= 60  # Ran for full duration
        
        print(f"Sustained Load Results:")
        print(f"  Total requests: {request_count}")
        print(f"  Successful: {successful_count}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Requests per second: {requests_per_second:.2f}")
        print(f"  Duration: {total_time:.2f}s")
    
    def test_memory_usage(self, api_url):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for i in range(100):
            requests.post(
                f"{api_url}/predict",
                json={"text": f"Memory test {i}"}
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly
        assert memory_increase < 100  # Less than 100MB increase
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
```

---

## 📊 **Test Coverage and Reporting**

### **Coverage Configuration**

```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */__pycache__/*
    */migrations/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod
```

### **Coverage Commands**

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Generate coverage badge
coverage-badge -o coverage-badge.svg

# Check coverage threshold
pytest --cov=src --cov-fail-under=80
```

### **Test Reporting**

```python
# tests/conftest.py
import pytest
import json
from datetime import datetime

def pytest_sessionfinish(session, exitstatus):
    """Generate test report after session."""
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": session.testscollected,
        "passed": len(session.testscollected) - len(session.testsfailed),
        "failed": len(session.testsfailed),
        "duration": session.duration,
        "exit_status": exitstatus
    }
    
    with open("test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
```

---

## 🔧 **Test Configuration and Fixtures**

### **Global Test Configuration**

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def mock_model():
    """Mock emotion detection model."""
    model = Mock()
    model.predict.return_value = {
        "predicted_emotion": "happy",
        "confidence": 0.95,
        "probabilities": {"happy": 0.95, "sad": 0.05},
        "prediction_time_ms": 150,
        "model_version": "1.0.0"
    }
    return model

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I am feeling happy today!",
        "I feel sad about the news",
        "I am excited for the party",
        "I feel anxious about the presentation",
        "I am grateful for your help"
    ]

@pytest.fixture
def api_client():
    """Create API test client."""
    from fastapi.testclient import TestClient
    from src.unified_ai_api import app
    return TestClient(app)

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = Mock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.expire.return_value = True
    return redis_mock
```

---

## 🚨 **Troubleshooting Tests**

### **Common Test Issues**

**Import Errors:**
```bash
# Ensure test environment is set up
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Install test dependencies
pip install -r requirements-test.txt
```

**Test Failures:**
```bash
# Run with verbose output
pytest -v --tb=long

# Run specific failing test
pytest tests/unit/test_specific.py::test_function -v -s

# Debug with pdb
pytest --pdb tests/unit/test_specific.py::test_function
```

**Performance Issues:**
```bash
# Run performance tests separately
pytest tests/performance/ -v

# Profile slow tests
pytest --durations=10 tests/
```

---

## 📞 **Support & Resources**

- **Test Documentation**: [Complete Test Guide](Testing-Framework-Guide)
- **Coverage Reports**: [Coverage Dashboard](coverage/index.html)
- **Performance Benchmarks**: [Performance Results](performance/results.md)
- **GitHub Issues**: [Report Test Issues](https://github.com/your-org/SAMO--DL/issues)

---

**Ready to write comprehensive tests?** Start with the [Quick Start](#-quick-start-5-minutes) section and build robust test coverage! 🧪✅ 