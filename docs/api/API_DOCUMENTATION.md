# SAMO-DL API Documentation

## Overview

The SAMO-DL API provides enterprise-grade emotion detection capabilities with >90% F1 score and sub-50ms latency. This document provides comprehensive information about all available endpoints, request/response formats, error handling, and usage examples.

## Base URL

```
https://samo-emotion-api-xxxxx-ew.a.run.app
```

## Authentication

Currently, the API uses IP-based rate limiting. For enterprise customers, API key authentication is available.

## Rate Limiting

- **Default**: 1000 requests per minute per IP
- **Burst**: 100 requests per second
- **Headers**: Rate limit information is included in response headers

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check the health status of the API and model

**Request**:
```bash
curl -X GET https://samo-emotion-api-xxxxx-ew.a.run.app/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime": "99.9%",
  "version": "2.0.0",
  "endpoints": ["/predict", "/health", "/metrics"],
  "timestamp": "2025-08-06T10:30:00Z"
}
```

**Status Codes**:
- `200`: Service is healthy
- `503`: Service is unhealthy or model not loaded

### 2. Emotion Detection

**Endpoint**: `POST /analyze/journal`

**Description**: Analyze text and return detected emotions with confidence scores

**Authentication**: Required (JWT Bearer token)

**Request Headers**:
```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json
```

**Request Body**:
```json
{
  "text": "I am feeling really happy today!",
  "generate_summary": true,
  "emotion_threshold": 0.1
}
```

**Request Parameters**:
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `text` | string | Yes | Text to analyze for emotions | "I'm excited about this project!" |

**Response**:
```json
[
  {
    "emotion": "joy",
    "confidence": 0.89
  },
  {
    "emotion": "excitement",
    "confidence": 0.76
  },
  {
    "emotion": "optimism",
    "confidence": 0.65
  }
]
```

**Response Format**:
| Field | Type | Description |
|-------|------|-------------|
| `emotion` | string | Detected emotion name |
| `confidence` | float | Confidence score (0.0 to 1.0) |

**Supported Emotions**:
```python
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
```

**Example Usage**:
```bash
curl -X POST https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/journal \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling really happy today!", "generate_summary": true}'
```

### 3. Voice Transcription

**Endpoint**: `POST /transcribe/voice`

**Description**: Transcribe audio files to text with detailed analysis

**Authentication**: Required (JWT Bearer token)

**Request Format**: `multipart/form-data` (NOT JSON)

**Request Headers**:
```
Authorization: Bearer YOUR_TOKEN
```

**Request Parameters**:
- `audio_file` (file, required): Audio file to transcribe
- `language` (form data, optional): Language code (e.g., "en")
- `model_size` (form data, optional): Whisper model size ("base", "small", "medium")
- `timestamp` (form data, optional): Include timestamps ("true"/"false")

**Example Usage**:
```bash
curl -X POST https://samo-unified-api-frrnetyhfa-uc.a.run.app/transcribe/voice \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio_file=@audio.wav" \
  -F "language=en" \
  -F "model_size=base"
```

**Response**:
```json
{
  "text": "Transcribed audio content",
  "language": "en",
  "confidence": 0.85,
  "duration": 15.4,
  "word_count": 12,
  "speaking_rate": 120.5,
  "audio_quality": "good"
}
```

### 4. Text Summarization

**Endpoint**: `POST /summarize/text`

**Description**: Generate summaries from text input

**Authentication**: Required (JWT Bearer token)

**Request Format**: `application/x-www-form-urlencoded` (NOT JSON)

**Request Headers**:
```
Authorization: Bearer YOUR_TOKEN
Content-Type: application/x-www-form-urlencoded
```

**Request Parameters**:
- `text` (form data, required): Text to summarize
- `model` (form data, optional): Model to use ("t5-small", "t5-base")
- `max_length` (form data, optional): Maximum summary length
- `min_length` (form data, optional): Minimum summary length

**Example Usage**:
```bash
curl -X POST https://samo-unified-api-frrnetyhfa-uc.a.run.app/summarize/text \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "text=Long text to summarize..." \
  -d "model=t5-small" \
  -d "max_length=150"
```

**Response**:
```json
{
  "summary": "Generated summary text",
  "key_emotions": ["neutral"],
  "compression_ratio": 0.7,
  "emotional_tone": "neutral"
}
```

### 5. Voice Journal Analysis

**Endpoint**: `POST /analyze/voice-journal`

**Description**: Complete pipeline - transcribe audio and analyze emotions

**Authentication**: Required (JWT Bearer token)

**Request Format**: `multipart/form-data` (NOT JSON)

**Request Parameters**:
- `audio_file` (file, required): Audio file to process
- `language` (form data, optional): Language for transcription
- `generate_summary` (form data, optional): Generate summary ("true"/"false")
- `emotion_threshold` (form data, optional): Emotion detection threshold

**Example Usage**:
```bash
curl -X POST https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/voice-journal \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio_file=@journal.wav" \
  -F "language=en" \
  -F "generate_summary=true"
```

### 6. Authentication

**Registration Endpoint**: `POST /auth/register`

**Request Body**:
```json
{
  "username": "user@example.com",
  "email": "user@example.com",
  "password": "YourPassword123!",
  "full_name": "Your Name"
}
```

**Login Endpoint**: `POST /auth/login`

**Request Body**:
```json
{
  "username": "user@example.com",
  "password": "YourPassword123!"
}
```

**Response** (both endpoints):
```json
{
  "access_token": "JWT_ACCESS_TOKEN_HERE",
  "refresh_token": "JWT_REFRESH_TOKEN_HERE",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 7. Metrics

**Endpoint**: `GET /metrics`

**Description**: Get Prometheus-formatted metrics for monitoring

**Request**:
```bash
curl -X GET https://samo-emotion-api-xxxxx-ew.a.run.app/metrics
```

**Response**:
```
# HELP samo_emotion_requests_total Total number of emotion detection requests
# TYPE samo_emotion_requests_total counter
samo_emotion_requests_total 1234

# HELP samo_emotion_request_duration_seconds Duration of emotion detection requests
# TYPE samo_emotion_request_duration_seconds histogram
samo_emotion_request_duration_seconds_bucket{le="0.01"} 100
samo_emotion_request_duration_seconds_bucket{le="0.05"} 500
samo_emotion_request_duration_seconds_bucket{le="0.1"} 1000
samo_emotion_request_duration_seconds_bucket{le="+Inf"} 1234

# HELP samo_emotion_model_loaded Model loaded status
# TYPE samo_emotion_model_loaded gauge
samo_emotion_model_loaded 1
```

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "error": "Error message description",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional error details"
  },
  "timestamp": "2025-08-06T10:30:00Z"
}
```

### HTTP Status Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `200` | Success | Request processed successfully |
| `400` | Bad Request | Invalid input format or missing required fields |
| `413` | Payload Too Large | Text exceeds maximum length (1000 characters) |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error or model loading issue |
| `503` | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Error Code | Description | HTTP Status |
|------------|-------------|-------------|
| `INVALID_INPUT` | Invalid or missing input text | 400 |
| `TEXT_TOO_LONG` | Text exceeds maximum length | 413 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `MODEL_NOT_LOADED` | Emotion detection model not available | 503 |
| `INTERNAL_ERROR` | Internal server error | 500 |

### Error Examples

**Invalid Input**:
```json
{
  "error": "Text field is required and cannot be empty",
  "code": "INVALID_INPUT",
  "details": {
    "field": "text"
  },
  "timestamp": "2025-08-06T10:30:00Z"
}
```

**Rate Limit Exceeded**:
```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "limit": 1000,
    "window": "1 minute",
    "retry_after": 30
  },
  "timestamp": "2025-08-06T10:30:00Z"
}
```

**Text Too Long**:
```json
{
  "error": "Text exceeds maximum length of 1000 characters",
  "code": "TEXT_TOO_LONG",
  "details": {
    "max_length": 1000,
    "actual_length": 1500
  },
  "timestamp": "2025-08-06T10:30:00Z"
}
```

## Response Headers

### Standard Headers

| Header | Description | Example |
|--------|-------------|---------|
| `Content-Type` | Response content type | `application/json` |
| `X-Request-ID` | Unique request identifier | `req_1234567890` |
| `X-Response-Time` | Request processing time | `45ms` |

### Rate Limiting Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-RateLimit-Limit` | Rate limit per window | `1000` |
| `X-RateLimit-Remaining` | Remaining requests in window | `999` |
| `X-RateLimit-Reset` | Time when rate limit resets | `1640995200` |
| `Retry-After` | Seconds to wait before retrying | `30` |

## Usage Examples

### Python

```python
import requests
import json

def detect_emotion(text: str) -> dict:
    """Detect emotions in text using SAMO-DL API"""
    url = "https://samo-emotion-api-xxxxx-ew.a.run.app/predict"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return {"error": "Failed to analyze emotions"}

# Example usage
emotions = detect_emotion("I'm feeling excited about this project!")
print(json.dumps(emotions, indent=2))
```

### JavaScript

```javascript
async function detectEmotion(text) {
    const url = 'https://samo-emotion-api-xxxxx-ew.a.run.app/predict';
    const options = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
    };
    
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return { error: 'Failed to analyze emotions' };
    }
}

// Example usage
const emotions = await detectEmotion("I'm feeling excited about this project!");
console.log(emotions);
```

### cURL

```bash
# Basic emotion detection
curl -X POST https://samo-emotion-api-xxxxx-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling really happy today!"}'

# Health check
curl -X GET https://samo-emotion-api-xxxxx-ew.a.run.app/health

# Get metrics
curl -X GET https://samo-emotion-api-xxxxx-ew.a.run.app/metrics
```

### Node.js with Axios

```javascript
const axios = require('axios');

async function detectEmotion(text) {
    try {
        const response = await axios.post(
            'https://samo-emotion-api-xxxxx-ew.a.run.app/predict',
            { text },
            {
                headers: { 'Content-Type': 'application/json' },
                timeout: 10000
            }
        );
        return response.data;
    } catch (error) {
        console.error('API Error:', error.message);
        return { error: 'Failed to analyze emotions' };
    }
}

// Example usage
const emotions = await detectEmotion("I'm feeling excited about this project!");
console.log(emotions);
```

## Performance Characteristics

### Latency

- **Average Response Time**: <50ms
- **95th Percentile**: <100ms
- **99th Percentile**: <200ms

### Throughput

- **Maximum Requests/Second**: 100 (burst)
- **Sustained Requests/Minute**: 1000
- **Concurrent Connections**: 1000+

### Model Performance

- **F1 Score**: >90%
- **Accuracy**: >92%
- **Precision**: >89%
- **Recall**: >91%

## Best Practices

### Input Validation

1. **Text Length**: Keep text under 1000 characters for optimal performance
2. **Content**: Avoid HTML tags, scripts, or malicious content
3. **Language**: Currently optimized for English text
4. **Encoding**: Use UTF-8 encoding

### Error Handling

1. **Always check HTTP status codes**
2. **Implement retry logic with exponential backoff**
3. **Handle rate limiting gracefully**
4. **Log errors for debugging**

### Performance Optimization

1. **Use connection pooling for high-volume requests**
2. **Implement caching for repeated text analysis**
3. **Batch requests when possible**
4. **Monitor response times and error rates**

### Security

1. **Validate all input text**
2. **Sanitize user input before sending to API**
3. **Use HTTPS for all requests**
4. **Implement proper error handling to avoid information leakage**

## Monitoring and Observability

### Health Checks

Monitor the `/health` endpoint to ensure service availability:

```bash
# Check service health
curl -f https://samo-emotion-api-xxxxx-ew.a.run.app/health || echo "Service unhealthy"
```

### Metrics Collection

Use the `/metrics` endpoint for Prometheus monitoring:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'samo-emotion-api'
    static_configs:
      - targets: ['samo-emotion-api-xxxxx-ew.a.run.app']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Key Metrics to Monitor

- **Request Rate**: `samo_emotion_requests_total`
- **Response Time**: `samo_emotion_request_duration_seconds`
- **Error Rate**: `samo_emotion_errors_total`
- **Model Status**: `samo_emotion_model_loaded`

## Support and Resources

### Documentation

- **Integration Guide**: [docs/guides/INTEGRATION_GUIDE.md](docs/guides/INTEGRATION_GUIDE.md)
- **Deployment Guide**: [docs/DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
- **Architecture Overview**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

### Examples

- **Python Examples**: [examples/python_integration.py](examples/python_integration.py)
- **JavaScript Examples**: [examples/javascript_integration.js](examples/javascript_integration.js)
- **React Component**: [examples/ReactEmotionDetector.jsx](examples/ReactEmotionDetector.jsx)

### Testing

- **API Test Suite**: [scripts/testing/](scripts/testing/)
- **Performance Benchmarks**: [scripts/testing/benchmarks.py](scripts/testing/benchmarks.py)

### Support

- **GitHub Issues**: [https://github.com/uelkerd/SAMO--DL/issues](https://github.com/uelkerd/SAMO--DL/issues)
- **Documentation**: [https://uelkerd.github.io/SAMO--DL/](https://uelkerd.github.io/SAMO--DL/)

## Changelog

### Version 2.0.0 (Current)
- **Performance**: 2.3x speedup with ONNX optimization
- **Accuracy**: >90% F1 score
- **Features**: Enhanced error handling and monitoring
- **Security**: Improved input validation and rate limiting

### Version 1.0.0
- **Initial Release**: Basic emotion detection API
- **Features**: Core emotion detection functionality
- **Performance**: Baseline PyTorch implementation

## License

This API is part of the SAMO-DL project. See the main project repository for licensing information. 