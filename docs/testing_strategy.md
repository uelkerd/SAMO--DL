# SAMO Deep Learning - Testing Strategy & Test Cases

## 📋 Overview

This document outlines the comprehensive testing strategy for the SAMO Deep Learning project, covering unit testing, integration testing, performance testing, and data validation procedures. Following this strategy ensures code quality, model reliability, and system robustness.

## 🧪 Testing Pyramid

Our testing approach follows the testing pyramid principle, with a strong foundation of unit tests, complemented by integration tests, and topped with end-to-end tests:

```
    /\
   /  \
  /E2E \
 /------\
/  INT   \
/----------\
/    UNIT    \
--------------
```

| Test Type | Target Coverage | Purpose | Run Frequency |
|-----------|-----------------|---------|---------------|
| Unit Tests | 80%+ | Verify individual functions/classes | Every commit |
| Integration Tests | 60%+ | Verify component interactions | Every PR |
| End-to-End Tests | Key workflows | Verify complete system | Daily |
| Performance Tests | Critical paths | Verify system meets SLAs | Weekly |

## 🧩 Unit Testing

### Component Coverage

| Component | Test Coverage | Key Test Focus |
|-----------|---------------|---------------|
| Data Preprocessing | 85% | Data cleaning, transformation, validation |
| Feature Engineering | 80% | Feature extraction, normalization |
| Model Architecture | 75% | Layer configuration, forward pass |
| Training Pipeline | 70% | Loss calculation, optimization steps |
| Inference | 90% | Prediction accuracy, edge cases |
| API Endpoints | 85% | Request/response handling, error cases |

### Example Unit Test: Emotion Detection Model

```python
# tests/unit/test_emotion_detection.py
import torch
import pytest
from models.emotion_detection.bert_classifier import BERTEmotionClassifier

class TestBERTEmotionClassifier:
    @pytest.fixture
    def model(self):
        model, _ = BERTEmotionClassifier.create_bert_emotion_classifier()
        return model
    
    def test_model_initialization(self, model):
        """Test that model initializes with correct parameters."""
        assert model.num_labels == 28
        assert model.model_name == "bert-base-uncased"
        assert isinstance(model.temperature, torch.nn.Parameter)
        
    def test_forward_pass(self, model):
        """Test model forward pass with dummy inputs."""
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check output shape (batch_size, num_labels)
        assert outputs.shape == (batch_size, 28)
        
    def test_temperature_scaling(self, model):
        """Test temperature scaling affects outputs correctly."""
        input_ids = torch.randint(0, 30522, (1, 128))
        attention_mask = torch.ones(1, 128)
        
        # Get outputs with default temperature
        outputs_default = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Set new temperature
        model.set_temperature(2.0)
        assert model.temperature.item() == 2.0
        
        # Get outputs with new temperature
        outputs_scaled = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Scaled outputs should be "softer" (closer to 0.5) than default
        assert torch.mean(torch.abs(outputs_scaled - 0.5)) < torch.mean(torch.abs(outputs_default - 0.5))
```

### Mocking Strategy

For tests that require external dependencies (databases, APIs, file systems), we use mocking:

```python
# Example of mocking database interactions
@pytest.fixture
def mock_db_connection(monkeypatch):
    class MockConnection:
        def execute_query(self, query):
            if "SELECT" in query:
                return [{"id": 1, "text": "Sample text", "label": "joy"}]
            return {"affected_rows": 1}
    
    monkeypatch.setattr("src.data.database.get_connection", lambda: MockConnection())
    return MockConnection()

def test_data_retrieval(mock_db_connection):
    from src.data.loaders import DataLoader
    
    loader = DataLoader()
    data = loader.load_from_database("emotions")
    
    assert len(data) == 1
    assert data[0]["label"] == "joy"
```

## 🔄 Integration Testing

### Key Integration Points

| Integration Point | Test Focus | Success Criteria |
|-------------------|------------|------------------|
| Data Pipeline → Model Training | Data loading, preprocessing, training | Model converges, metrics improve |
| Model → API | Model loading, inference, response formatting | Correct predictions, proper error handling |
| API → Database | Query execution, data persistence | Data integrity, query performance |
| Frontend → API | Request validation, response handling | Correct rendering, error handling |

### Example Integration Test: Data Pipeline to Model Training

```python
# tests/integration/test_training_pipeline.py
import pytest
import os
import torch
from models.emotion_detection.training_pipeline import EmotionDetectionTrainer
from models.emotion_detection.dataset_loader import GoEmotionsDataLoader

@pytest.mark.integration
def test_training_pipeline_integration():
    """Test the full training pipeline with a small dataset."""
    # Use a small subset of data for quick testing
    os.environ["SAMO_ENV"] = "testing"
    
    # Initialize components
    data_loader = GoEmotionsDataLoader(dev_mode=True)
    trainer = EmotionDetectionTrainer(
        batch_size=4,
        learning_rate=5e-5,
        num_epochs=1,
        dev_mode=True
    )
    
    # Run the pipeline
    train_dataset, val_dataset, _ = data_loader.prepare_datasets()
    model, optimizer = trainer.initialize_model_and_optimizer()
    
    # Train for 1 epoch
    train_loss = trainer.train_epoch(model, optimizer, train_dataset)
    
    # Evaluate
    val_metrics = trainer.evaluate(model, val_dataset)
    
    # Assertions
    assert train_loss > 0  # Training occurred
    assert "f1_score" in val_metrics
    assert val_metrics["f1_score"] > 0
```

### API Integration Tests

```python
# tests/integration/test_api_endpoints.py
import pytest
import requests
import json
from src.unified_ai_api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.mark.integration
def test_emotion_detection_endpoint(client):
    """Test the emotion detection API endpoint."""
    response = client.post(
        '/api/v1/emotions/analyze',
        data=json.dumps({'text': 'I feel so happy today!'}),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'emotions' in data
    assert isinstance(data['emotions'], list)
    assert len(data['emotions']) > 0
    
    # Check format of each emotion
    for emotion in data['emotions']:
        assert 'label' in emotion
        assert 'score' in emotion
        assert 0 <= emotion['score'] <= 1
```

## 🌐 End-to-End Testing

### Critical User Journeys

| Journey | Test Steps | Success Criteria |
|---------|------------|------------------|
| Text Emotion Analysis | Upload text → Process → View results | Correct emotions detected, confidence scores displayed |
| Voice Emotion Analysis | Upload audio → Transcribe → Analyze → View results | Accurate transcription, emotion detection matches content |
| Text Summarization | Upload document → Process → View summary | Summary captures key points, appropriate length |
| Batch Processing | Upload multiple files → Process → Download results | All files processed correctly, results properly formatted |

### Example E2E Test: Text Emotion Analysis

```python
# tests/e2e/test_complete_workflows.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@pytest.mark.e2e
class TestEmotionAnalysisWorkflow:
    @pytest.fixture
    def browser(self):
        driver = webdriver.Chrome()
        driver.implicitly_wait(10)
        yield driver
        driver.quit()
    
    def test_text_emotion_analysis(self, browser):
        """Test the complete text emotion analysis workflow."""
        # Navigate to application
        browser.get("https://app.samo.ai")
        
        # Log in
        browser.find_element(By.ID, "email").send_keys("test@example.com")
        browser.find_element(By.ID, "password").send_keys("password123")
        browser.find_element(By.ID, "login-button").click()
        
        # Navigate to emotion analysis
        browser.find_element(By.ID, "emotion-analysis-link").click()
        
        # Enter text
        text_area = browser.find_element(By.ID, "text-input")
        text_area.clear()
        text_area.send_keys("I'm feeling really excited about this project!")
        
        # Submit for analysis
        browser.find_element(By.ID, "analyze-button").click()
        
        # Wait for results
        WebDriverWait(browser, 30).until(
            EC.presence_of_element_located((By.ID, "results-container"))
        )
        
        # Verify results
        results = browser.find_element(By.ID, "results-container")
        emotion_items = results.find_elements(By.CLASS_NAME, "emotion-item")
        
        # There should be at least one emotion detected
        assert len(emotion_items) > 0
        
        # "Excitement" should be among the top emotions
        emotion_texts = [item.text.lower() for item in emotion_items]
        assert any("excite" in text for text in emotion_texts)
        
        # Check confidence scores are displayed
        confidence_elements = results.find_elements(By.CLASS_NAME, "confidence-score")
        assert len(confidence_elements) > 0
        for elem in confidence_elements:
            score_text = elem.text
            assert "%" in score_text
```

## 🚀 Performance Testing

### Performance Test Cases

| Test Case | Description | Success Criteria |
|-----------|-------------|------------------|
| API Response Time | Measure response time under various loads | p95 < 300ms at 100 RPS |
| Model Inference Latency | Measure time to generate predictions | p95 < 100ms per request |
| Batch Processing Throughput | Measure documents processed per minute | > 1000 documents/minute |
| Memory Usage | Monitor memory during sustained load | < 4GB per instance |
| Database Query Performance | Measure query execution time | p95 < 50ms for common queries |

### Example Performance Test: API Load Testing

```python
# tests/performance/test_api_load.py
import time
import statistics
import concurrent.futures
import requests
import pytest

@pytest.mark.performance
def test_api_response_time():
    """Test API response time under load."""
    base_url = "https://api.samo.ai/v1/emotions/analyze"
    headers = {"Authorization": "Bearer test_token"}
    payload = {"text": "I'm feeling really happy about this achievement!"}
    
    # Number of concurrent requests
    num_requests = 100
    response_times = []
    
    def make_request():
        start_time = time.time()
        response = requests.post(base_url, json=payload, headers=headers)
        end_time = time.time()
        
        assert response.status_code == 200
        return end_time - start_time
    
    # Execute concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_request = {executor.submit(make_request): i for i in range(num_requests)}
        for future in concurrent.futures.as_completed(future_to_request):
            response_time = future.result()
            response_times.append(response_time)
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times)
    p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
    
    print(f"Average response time: {avg_response_time:.3f}s")
    print(f"95th percentile response time: {p95_response_time:.3f}s")
    
    # Assertions
    assert avg_response_time < 0.2  # Average < 200ms
    assert p95_response_time < 0.3  # 95th percentile < 300ms
```

## 📊 Data Quality Testing

### Data Validation Test Cases

| Test Case | Description | Success Criteria |
|-----------|-------------|------------------|
| Schema Validation | Verify data conforms to expected schema | 100% compliance |
| Completeness Check | Verify required fields are present | < 1% missing values |
| Range Validation | Verify values fall within expected ranges | 100% compliance |
| Consistency Check | Verify related data points are consistent | < 0.1% inconsistencies |
| Duplicate Detection | Identify and flag duplicate records | < 0.5% duplicates |

### Example Data Quality Test: Dataset Validation

```python
# tests/unit/test_data_validation.py
import pytest
from src.data.validation import DataValidator
from models.emotion_detection.dataset_loader import GoEmotionsDataLoader

def test_dataset_validation():
    """Test validation of the GoEmotions dataset."""
    # Load dataset
    loader = GoEmotionsDataLoader()
    dataset = loader.download_dataset()
    
    # Create validator
    validator = DataValidator()
    
    # Define validation rules
    rules = {
        "text": {
            "type": "string",
            "required": True,
            "min_length": 1
        },
        "labels": {
            "type": "list",
            "required": True
        }
    }
    
    # Validate dataset
    validation_results = validator.validate_dataset(dataset["train"], rules)
    
    # Check results
    assert validation_results["valid_percentage"] > 99
    assert validation_results["missing_fields_percentage"] < 1
    assert validation_results["type_errors_percentage"] < 0.5
```

## 🔄 Continuous Integration Testing

### CI Pipeline Test Stages

| Stage | Tests Run | Trigger | Success Criteria |
|-------|-----------|---------|------------------|
| Pre-commit | Linting, formatting | Local commit | All checks pass |
| Quick Tests | Unit tests | Push to branch | All tests pass |
| Full Tests | Unit + Integration tests | PR creation | All tests pass, coverage thresholds met |
| Performance Tests | Load tests, benchmarks | Scheduled (nightly) | Performance within thresholds |
| Security Tests | Vulnerability scans | PR to main | No high/critical vulnerabilities |

### CircleCI Configuration

```yaml
# .circleci/config.yml (excerpt)
version: 2.1

jobs:
  unit-tests:
    docker:
      - image: cimg/python:3.10
    steps:
      - checkout
      - restore_cache:
          keys:
            - v1-dependencies-{{ checksum "pyproject.toml" }}
      - run:
          name: Install dependencies
          command: |
            python -m pip install --upgrade pip
            pip install -e .
      - save_cache:
          paths:
            - ~/.cache/pip
          key: v1-dependencies-{{ checksum "pyproject.toml" }}
      - run:
          name: Run unit tests
          command: |
            pytest tests/unit/ --cov=src --cov-report=xml
      - store_test_results:
          path: test-results
      - store_artifacts:
          path: coverage.xml

  integration-tests:
    docker:
      - image: cimg/python:3.10
      - image: cimg/postgres:14
        environment:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: samo_test
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            python -m pip install --upgrade pip
            pip install -e .
      - run:
          name: Run integration tests
          command: |
            pytest tests/integration/ -v

  model-validation:
    docker:
      - image: cimg/python:3.10
    resource_class: large
    steps:
      - checkout
      - run:
          name: Install dependencies
          command: |
            python -m pip install --upgrade pip
            pip install -e .
      - run:
          name: Run model tests
          command: |
            python scripts/ci/bert_model_test.py
      - run:
          name: Model Calibration Test
          command: |
            python scripts/ci/model_calibration_test.py
      - run:
          name: Model Compression Test
          command: |
            python scripts/compress_model.py --input-model test_checkpoints/best_model.pt --output-model test_checkpoints/compressed_model.pt
      - run:
          name: ONNX Conversion Test
          command: |
            python scripts/convert_to_onnx.py --input-model test_checkpoints/best_model.pt --output-model test_checkpoints/model.onnx

workflows:
  version: 2
  build-test-deploy:
    jobs:
      - unit-tests
      - integration-tests:
          requires:
            - unit-tests
      - model-validation:
          requires:
            - unit-tests
```

## 🐛 Debugging and Troubleshooting

### Common Test Failures and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Model test failures | Incompatible model initialization | Check model constructor parameters, device handling |
| Data loading errors | Missing cache, permission issues | Clear cache, check file permissions |
| CI pipeline timeouts | Long-running tests, resource constraints | Use dev_mode, optimize test execution |
| Flaky tests | Race conditions, external dependencies | Add retries, improve mocking, isolate tests |
| Memory errors | Large model/dataset, memory leaks | Use smaller batches, profile memory usage |

### Debugging Tools

1. **pytest-xdist**: Parallel test execution
   ```bash
   pytest -n 4 tests/  # Run tests on 4 cores
   ```

2. **pytest-cov**: Coverage reporting
   ```bash
   pytest --cov=src --cov-report=html tests/
   ```

3. **pytest-profiling**: Performance profiling
   ```bash
   pytest --profile tests/unit/test_model.py
   ```

4. **pytest-timeout**: Prevent hanging tests
   ```bash
   pytest --timeout=300 tests/integration/
   ```

## 📈 Test Coverage Goals

| Component | Current Coverage | Target Coverage | Priority Areas |
|-----------|------------------|-----------------|----------------|
| Data Processing | 82% | 90% | Error handling, edge cases |
| Model Core | 75% | 85% | Training logic, custom layers |
| API Layer | 88% | 90% | Authentication, rate limiting |
| Database | 70% | 80% | Transaction handling, migrations |
| Frontend | 65% | 75% | State management, error displays |

### Coverage Improvement Strategy

1. **Identify gaps**: Run coverage reports to find untested code
2. **Prioritize critical paths**: Focus on user-facing functionality first
3. **Add test cases incrementally**: Target 2-3% coverage increase per sprint
4. **Refactor for testability**: Improve code design to facilitate testing

## 🔒 Security Testing

### Security Test Cases

| Test Case | Description | Tools |
|-----------|-------------|-------|
| Dependency Scanning | Check for vulnerable dependencies | safety, npm audit |
| Secret Detection | Detect hardcoded secrets | detect-secrets |
| SAST | Static code analysis for security issues | bandit, semgrep |
| API Security Testing | Test for OWASP Top 10 vulnerabilities | OWASP ZAP |
| Container Scanning | Check container images for vulnerabilities | trivy |

### Example Security Test: Dependency Scanning

```bash
# Run as part of CI pipeline
safety check --full-report
```

## 🔄 Test Automation Best Practices

1. **Test isolation**: Each test should be independent and not rely on other tests
2. **Deterministic tests**: Tests should produce the same result on every run
3. **Fast execution**: Optimize tests to run quickly to enable frequent runs
4. **Readable tests**: Use clear naming and structure for easy understanding
5. **Maintainable fixtures**: Use fixtures for test setup and teardown
6. **Continuous feedback**: Run tests automatically on code changes

## 📝 Test Documentation Standards

### Test Case Documentation Template

```markdown
## Test Case: [ID] - [Brief Description]

### Objective
[What this test aims to verify]

### Preconditions
- [Required setup/state before test execution]

### Test Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Expected Results
- [Expected outcome after test execution]

### Data Requirements
- [Test data needed]

### Special Instructions
- [Any special considerations or notes]
```

### Example: Emotion Detection Model Test Case

```markdown
## Test Case: ED-001 - Multi-label Emotion Detection Accuracy

### Objective
Verify that the emotion detection model correctly identifies multiple emotions in a single text input.

### Preconditions
- Model is trained on GoEmotions dataset
- Model checkpoint is available at models/checkpoints/best_model.pt

### Test Steps
1. Load the trained model
2. Process the text: "I'm both excited and nervous about the upcoming presentation"
3. Get predicted emotion labels with confidence scores
4. Compare predictions with expected emotions

### Expected Results
- Both "excitement" and "nervousness" emotions should be detected
- Confidence scores for both emotions should be above 0.5
- No unrelated emotions should have high confidence scores

### Data Requirements
- Test text with multiple emotions
- Trained model checkpoint

### Special Instructions
- This test requires GPU for optimal performance
- If run on CPU, expect slower inference times
```

## 🚀 Next Steps for Testing Improvement

1. **Automated UI Testing**: Implement Cypress for frontend testing
2. **Property-based Testing**: Add hypothesis for edge case discovery
3. **Chaos Testing**: Test system resilience under failure conditions
4. **Continuous Benchmarking**: Track performance metrics over time
5. **Test Data Management**: Improve test data generation and management 