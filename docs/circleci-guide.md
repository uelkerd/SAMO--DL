# CircleCI Pipeline Guide - SAMO Deep Learning

## Overview

The SAMO Deep Learning project uses a comprehensive 3-stage CircleCI pipeline designed to ensure code quality, security, and performance while supporting AI/ML workloads.

## Pipeline Architecture

### 🚀 Stage 1: Fast Feedback (<5 minutes)

- **Linting & Formatting**: Ruff code quality checks
- **Unit Tests**: Core functionality testing with mocking
- **Parallel Execution**: Both jobs run simultaneously for quick feedback

### 🔍 Stage 2: Integration & Security (<15 minutes)  

- **Security Scanning**: Bandit + Safety vulnerability detection
- **Integration Tests**: API endpoint and service integration testing
- **Model Validation**: AI model loading and basic inference testing
- **Dependency Checks**: Vulnerability scanning of dependencies

### 🎯 Stage 3: Comprehensive Testing (<30 minutes)

- **End-to-End Tests**: Complete workflow testing
- **Performance Benchmarks**: Response time and throughput validation
- **GPU Compatibility**: CUDA-enabled model testing (when available)
- **Docker Build & Deploy**: Production image creation and deployment

## Quality Gates

### ✅ Code Quality Requirements

- **Ruff Linting**: All linting errors must be resolved
- **Code Formatting**: Code must pass Ruff formatting checks
- **Type Checking**: MyPy type checking must pass (warnings allowed)

### 📊 Test Coverage Requirements

- **Minimum Coverage**: 70% test coverage required
- **Unit Tests**: Must pass with comprehensive assertions
- **Integration Tests**: API endpoints must respond correctly
- **E2E Tests**: Complete workflows must function end-to-end

### 🔒 Security Requirements

- **Bandit Scan**: Security vulnerabilities must be addressed
- **Dependency Safety**: Known vulnerabilities flagged and documented
- **Secrets Detection**: No hardcoded secrets in code

### ⚡ Performance Requirements

- **API Response Time**: <2 seconds in CI environment (<500ms target for production)
- **Model Loading**: AI models must initialize within reasonable time
- **Throughput**: System must handle concurrent requests

## Environment Configuration

### Required Environment Variables

```bash
# CircleCI Project Settings > Environment Variables
PYTHONPATH=/home/circleci/samo-dl/src
TOKENIZERS_PARALLELISM=false
TESTING=1

# Optional: For production deployment
SAMO_API_KEY=your-api-key
DEPLOYMENT_ENV=staging|production
SLACK_WEBHOOK_URL=your-slack-webhook (for notifications)
```

### Resource Classes

- **Standard Jobs**: `large` (4 CPU, 8GB RAM)
- **GPU Jobs**: `gpu.nvidia.medium` (2 GPU, 8 CPU, 15GB RAM)
- **Performance Tests**: `xlarge` (8 CPU, 16GB RAM) if needed

## Branch Strategy

### Automatic Pipeline Triggers

- **All Branches**: Runs stages 1-2 (fast feedback + security)
- **Main Branch**: Runs complete pipeline including deployment
- **Feature Branches**: Runs all tests except deployment
- **GPU Branches**: Branches matching `/^feature\/gpu-.*/` run GPU tests

### Manual Triggers

- **Nightly Benchmarks**: Scheduled performance testing (2 AM UTC)
- **Manual Approval**: Required for production deployment

## Local Development Setup

### Prerequisites

```bash
# Install Python 3.12+
python --version  # Should be 3.12+

# Install project in development mode
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests Locally

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# End-to-end tests (slower)
pytest tests/e2e/ -v

# All tests with coverage
pytest --cov=src --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Run specific test categories
pytest -m integration
pytest -m e2e
pytest -m gpu  # (if GPU available)
```

### Code Quality Checks

```bash
# Linting
ruff check src/ tests/ scripts/

# Formatting
ruff format src/ tests/ scripts/

# Type checking
mypy src/ --ignore-missing-imports

# Security scan
bandit -r src/

# Dependency vulnerabilities
safety check
```

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. **Test Failures**

**Symptom**: Unit tests failing with import errors

```
ModuleNotFoundError: No module named 'src.models'
```

**Solution**:

```bash
# Check PYTHONPATH is set correctly
export PYTHONPATH=/path/to/samo-dl/src

# Or install in editable mode
pip install -e .
```

**Symptom**: Integration tests timing out

```
FAILED tests/integration/test_api_endpoints.py::TestAPIEndpoints::test_performance_requirements
```

**Solution**: Check if models are loading correctly and optimize for CI environment:

```python
# Use smaller models or mocking in CI
if os.getenv("TESTING"):
    model = MockModel()  # Faster for CI
else:
    model = RealModel()  # Full model for production
```

#### 2. **Memory Issues**

**Symptom**: Out of memory errors during model loading

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

- Use CPU-only models in CI: `torch.device("cpu")`
- Implement model caching between test runs
- Use smaller batch sizes for testing
- Mock heavy models in unit tests

#### 3. **Dependency Conflicts**

**Symptom**: Package installation failures

```
ERROR: pip's dependency resolver does not currently support...
```

**Solutions**:

```bash
# Clear pip cache
pip cache purge

# Install specific versions
pip install "torch==2.0.0" "transformers==4.30.0"

# Use conda for complex ML dependencies
conda env create -f environment.yml
```

#### 4. **Performance Test Failures**

**Symptom**: API response times exceed thresholds

```
AssertionError: API response too slow: 3.45s
```

**Solutions**:

- Check if models are properly cached
- Verify CircleCI resource class is sufficient
- Optimize model loading with lazy initialization
- Use async processing for heavy operations

#### 5. **Docker Build Issues**

**Symptom**: Docker build failing in CI

```
Error: failed to solve: process "/bin/sh -c pip install -e ." did not complete
```

**Solutions**:

```dockerfile
# Use multi-stage builds to reduce size
FROM python:3.12-slim as base
# ... build stage ...
FROM base as production
# ... production stage ...

# Add proper error handling
RUN pip install --no-cache-dir -e . || \
    (echo "Build failed" && cat /tmp/pip-*.log && exit 1)
```

### GPU Testing Issues

**Symptom**: GPU tests skipped even when GPU available

```
SKIPPED [1] tests/conftest.py:xx: CUDA not available
```

**Solutions**:

- Verify CircleCI GPU resource class is selected
- Check CUDA drivers are installed
- Enable GPU in CircleCI project settings

### CircleCI Configuration Issues

**Symptom**: Workflow not running

```
This workflow was not run because it is not triggered by this event
```

**Solutions**:

- Check branch filters in `.circleci/config.yml`
- Verify workflow triggers are correct
- Check if branch naming conventions match filters

## Performance Optimization

### Model Loading Optimization

```python
# Use model caching
@lru_cache(maxsize=1)
def load_model(model_name: str):
    return torch.load(f"models/{model_name}")

# Lazy loading
class ModelManager:
    def __init__(self):
        self._models = {}
    
    def get_model(self, name):
        if name not in self._models:
            self._models[name] = load_model(name)
        return self._models[name]
```

### Test Parallelization

```bash
# Run tests in parallel
pytest -n auto  # Auto-detect CPU cores
pytest -n 4     # Use 4 processes

# Distribute tests by duration
pytest --dist=loadscope
```

### Caching Strategies

```yaml
# In .circleci/config.yml
- save_cache:
    key: deps-v1-{{ checksum "pyproject.toml" }}
    paths:
      - ~/.cache/pip
      - ~/.cache/huggingface
      - data/cache
      - models/*/cache
```

## Monitoring & Alerts

### Slack Integration

Add to CircleCI environment variables:

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Performance Monitoring

- Response time trends tracked in artifacts
- Model loading time benchmarks
- Test execution duration monitoring

### Failed Build Notifications

- Automatic Slack notifications for failures
- Email alerts for main branch issues
- GitHub status checks integration

## Best Practices

### 1. **Test Organization**

- Keep unit tests fast (<1s each)
- Use mocking for external dependencies
- Separate slow tests with `@pytest.mark.slow`
- Group related tests in classes

### 2. **CI/CD Efficiency**

- Use Docker layer caching
- Cache dependencies between runs
- Parallelize independent jobs
- Skip unnecessary tests on documentation changes

### 3. **Model Management**

- Version control model configurations
- Use model registries for large models
- Implement fallback models for CI
- Cache model artifacts

### 4. **Security Best Practices**

- Never commit API keys or secrets
- Use environment variables for configuration
- Regularly update dependencies
- Scan for vulnerabilities continuously

## Migration Guide

### From Existing CI Systems

**From GitHub Actions**:

1. Convert workflow files to CircleCI config
2. Update environment variable names
3. Adjust resource classes and parallelism
4. Test branch filters and triggers

**From Jenkins**:

1. Replace Jenkinsfile with `.circleci/config.yml`
2. Convert pipeline steps to CircleCI jobs
3. Update artifact storage paths
4. Migrate environment configurations

### Deployment Integration

**Staging Environment**:

- Automatic deployment on main branch
- Health checks after deployment
- Rollback on failure

**Production Environment**:

- Manual approval required
- Blue-green deployment strategy
- Comprehensive monitoring

## Support & Resources

- **CircleCI Documentation**: <https://circleci.com/docs/>
- **SAMO DL Team**: Contact via Slack #samo-dl-dev
- **Pipeline Status**: <https://app.circleci.com/pipelines/github/samo-ai/samo-dl>
- **Performance Dashboards**: Internal monitoring links

For additional support, create an issue in the repository or contact the development team.
