# SAMO Testing Environment

This directory contains comprehensive testing scripts for the SAMO Emotion Detection API.

## 🚀 Quick Start

### 1. Setup Testing Environment

Run the setup script to configure your testing environment:

```bash
./scripts/testing/setup_test_environment.sh
```

This will:
- Prompt you to select an API endpoint
- Set up environment variables
- Create a `.env` file for persistent configuration
- Test the configuration

### 2. Run Tests

After setup, you can run any of the testing scripts:

```bash
# Comprehensive API testing
python scripts/testing/test_cloud_run_api_endpoints.py

# Quick health check
python scripts/testing/check_model_health.py

# Model status verification
python scripts/testing/test_model_status.py
```

## 🔧 Configuration

### Environment Variables

The testing scripts support multiple environment variables for flexibility:

- `API_BASE_URL` - Primary API URL
- `CLOUD_RUN_API_URL` - Alternative API URL
- `MODEL_API_BASE_URL` - Model-specific API URL
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: 30)
- `RATE_LIMIT_REQUESTS` - Number of requests for rate limiting tests (default: 10)

### Command Line Arguments

All scripts support the `--base-url` argument to override the configured URL:

```bash
python scripts/testing/test_cloud_run_api_endpoints.py --base-url "https://your-custom-api.com"
```

### Available API Endpoints

1. **Minimal API (Production)**: `https://samo-emotion-api-minimal-71517823771.us-central1.run.app`
2. **Optimized API (Staging)**: `https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app`

## 📋 Test Scripts

### `test_cloud_run_api_endpoints.py`

Comprehensive API testing including:
- Health endpoint validation
- Emotion detection functionality
- Model loading verification
- Invalid input handling
- Security features (rate limiting)
- Performance metrics

**Output**: Detailed JSON report in `test_reports/cloud_run_api_test_results.json`

### `check_model_health.py`

Quick health check for:
- API availability
- Model loading status
- Basic prediction functionality

**Output**: Console status with pass/fail indicators

### `test_model_status.py`

Detailed model status verification:
- Health endpoint
- Emotions support
- Model type and service info
- Prediction endpoint

**Output**: Console summary with detailed status

## 🏗️ Architecture

### Centralized Configuration

All scripts use centralized configuration through:
- `test_config.py` - Primary configuration management
- `config.py` - Alternative configuration with additional features

### API Client

Consistent API client with:
- Automatic retry logic
- Error handling
- Request/response logging
- Authentication support

### Test Results

- **Console Output**: Real-time test progress and results
- **JSON Reports**: Detailed test results for analysis
- **Exit Codes**: Proper exit codes for CI/CD integration

## 🔒 Security Features

The testing suite includes:
- API key authentication
- Rate limiting tests
- Security header validation
- Input sanitization testing

## 🚨 Error Handling

Comprehensive error handling for:
- Network connectivity issues
- API response errors
- Configuration problems
- Invalid input scenarios

## 📊 Performance Testing

Performance metrics include:
- Response time measurements
- Success rate calculation
- Throughput analysis
- Error rate monitoring

## 🛠️ Development

### Adding New Tests

1. Create a new test script in this directory
2. Import the centralized configuration:
   ```python
   from test_config import create_test_config, create_api_client
   ```
3. Use the API client for consistent request handling
4. Follow the existing test structure and output format

### Configuration Updates

To modify configuration behavior:
1. Update `test_config.py` for core functionality
2. Update `config.py` for additional features
3. Ensure backward compatibility
4. Update this README if needed

## 📝 Troubleshooting

### Common Issues

1. **No API URL configured**: Run the setup script or set environment variables
2. **Connection errors**: Check network connectivity and API availability
3. **Authentication failures**: Verify API key configuration
4. **Timeout errors**: Increase `REQUEST_TIMEOUT` environment variable

### Debug Mode

Enable verbose output with the `--verbose` flag:

```bash
python scripts/testing/test_cloud_run_api_endpoints.py --verbose
```

## 🔄 CI/CD Integration

The testing scripts are designed for CI/CD integration:

- **Exit Codes**: 0 for success, 1 for failure
- **JSON Output**: Machine-readable test results
- **Environment Variables**: Easy configuration in CI/CD pipelines
- **Logging**: Structured logging for monitoring

Example CI/CD usage:

```yaml
# GitHub Actions example
- name: Run API Tests
  env:
    API_BASE_URL: ${{ secrets.API_BASE_URL }}
  run: |
    python scripts/testing/test_cloud_run_api_endpoints.py
    python scripts/testing/check_model_health.py
``` 