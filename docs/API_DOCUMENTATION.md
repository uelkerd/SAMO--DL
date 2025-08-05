# SAMO Emotion Detection API Documentation

## Overview

The SAMO Emotion Detection API is a production-ready service that analyzes text input and predicts emotional states. The system supports 12 different emotions with high accuracy and confidence levels.

### Key Features

- **High Accuracy**: 100% basic accuracy, 93.75% real-world accuracy
- **12 Emotions**: anxious, calm, content, excited, frustrated, grateful, happy, hopeful, overwhelmed, proud, sad, tired
- **Real-time Processing**: Average response time < 100ms
- **Batch Processing**: Efficient batch predictions
- **Rate Limiting**: 100 requests per minute per IP
- **Comprehensive Monitoring**: Real-time metrics and logging
- **Production Ready**: Robust error handling and validation

## Base URL

```
http://localhost:8000 (Local Development)
https://your-production-domain.com (Production)
```

## Authentication

Currently, the API does not require authentication for local development. For production deployment, consider implementing API keys or OAuth2.

## Endpoints

### 1. Health Check

**GET** `/health`

Check the health status of the API and get basic metrics.

#### Response

```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model_version": "2.0",
  "emotions": ["anxious", "calm", "content", "excited", "frustrated", "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"],
  "uptime_seconds": 1234.5,
  "metrics": {
    "total_requests": 150,
    "successful_requests": 145,
    "failed_requests": 5,
    "average_response_time_ms": 65.2
  }
}
```

#### Example

```bash
curl -X GET http://localhost:8000/health
```

### 2. Single Prediction

**POST** `/predict`

Analyze a single text input and predict the emotional state.

#### Request Body

```json
{
  "text": "I am feeling happy today!"
}
```

#### Response

```json
{
  "text": "I am feeling happy today!",
  "predicted_emotion": "happy",
  "confidence": 0.964,
  "prediction_time_ms": 25.3,
  "probabilities": {
    "anxious": 0.001,
    "calm": 0.002,
    "content": 0.004,
    "excited": 0.004,
    "frustrated": 0.002,
    "grateful": 0.005,
    "happy": 0.964,
    "hopeful": 0.004,
    "overwhelmed": 0.001,
    "proud": 0.002,
    "sad": 0.008,
    "tired": 0.002
  },
  "model_version": "2.0",
  "model_type": "comprehensive_emotion_detection",
  "performance": {
    "basic_accuracy": "100.00%",
    "real_world_accuracy": "93.75%",
    "average_confidence": "83.9%"
  }
}
```

#### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

### 3. Batch Prediction

**POST** `/predict_batch`

Analyze multiple text inputs in a single request for improved efficiency.

#### Request Body

```json
{
  "texts": [
    "I am feeling happy today!",
    "I feel sad about the news",
    "I am excited for the party"
  ]
}
```

#### Response

```json
{
  "predictions": [
    {
      "text": "I am feeling happy today!",
      "predicted_emotion": "happy",
      "confidence": 0.964,
      "prediction_time_ms": 25.3,
      "probabilities": { ... },
      "model_version": "2.0",
      "model_type": "comprehensive_emotion_detection",
      "performance": { ... }
    },
    {
      "text": "I feel sad about the news",
      "predicted_emotion": "sad",
      "confidence": 0.965,
      "prediction_time_ms": 22.1,
      "probabilities": { ... },
      "model_version": "2.0",
      "model_type": "comprehensive_emotion_detection",
      "performance": { ... }
    },
    {
      "text": "I am excited for the party",
      "predicted_emotion": "excited",
      "confidence": 0.968,
      "prediction_time_ms": 21.4,
      "probabilities": { ... },
      "model_version": "2.0",
      "model_type": "comprehensive_emotion_detection",
      "performance": { ... }
    }
  ],
  "count": 3,
  "batch_processing_time_ms": 68.8
}
```

#### Example

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am happy", "I feel sad", "I am excited"]}'
```

### 4. Metrics

**GET** `/metrics`

Get detailed server metrics and performance statistics.

#### Response

```json
{
  "server_metrics": {
    "uptime_seconds": 1234.5,
    "total_requests": 150,
    "successful_requests": 145,
    "failed_requests": 5,
    "success_rate": "96.67%",
    "average_response_time_ms": 65.2,
    "requests_per_minute": 7.3
  },
  "emotion_distribution": {
    "happy": 45,
    "sad": 23,
    "excited": 18,
    "anxious": 12,
    "calm": 8,
    "grateful": 7,
    "frustrated": 6,
    "overwhelmed": 5,
    "proud": 4,
    "hopeful": 3,
    "content": 2,
    "tired": 1
  },
  "error_counts": {
    "missing_text": 2,
    "empty_text": 2,
    "prediction_error": 1
  },
  "rate_limiting": {
    "window_seconds": 60,
    "max_requests": 100
  }
}
```

#### Example

```bash
curl -X GET http://localhost:8000/metrics
```

### 5. API Documentation

**GET** `/`

Get comprehensive API documentation and usage examples.

#### Response

```json
{
  "message": "Comprehensive Emotion Detection API",
  "version": "2.0",
  "endpoints": {
    "GET /": "This documentation",
    "GET /health": "Health check with basic metrics",
    "GET /metrics": "Detailed server metrics",
    "POST /predict": "Single prediction (send {\"text\": \"your text\"})",
    "POST /predict_batch": "Batch prediction (send {\"texts\": [\"text1\", \"text2\"]})"
  },
  "model_info": {
    "emotions": ["anxious", "calm", "content", "excited", "frustrated", "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"],
    "performance": {
      "basic_accuracy": "100.00%",
      "real_world_accuracy": "93.75%",
      "average_confidence": "83.9%"
    }
  },
  "features": {
    "rate_limiting": "100 requests per 60 seconds",
    "monitoring": "Comprehensive metrics and logging",
    "batch_processing": "Efficient batch predictions",
    "error_handling": "Robust error handling and reporting"
  },
  "example_usage": {
    "single_prediction": {
      "url": "POST /predict",
      "body": "{\"text\": \"I am feeling happy today!\"}"
    },
    "batch_prediction": {
      "url": "POST /predict_batch",
      "body": "{\"texts\": [\"I am happy\", \"I feel sad\", \"I am excited\"]}"
    }
  }
}
```

#### Example

```bash
curl -X GET http://localhost:8000/
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request format or missing required fields
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

### Error Response Format

```json
{
  "error": "Error description",
  "message": "Additional error details (for rate limiting)"
}
```

### Common Errors

#### Missing Text

```json
{
  "error": "No text provided"
}
```

#### Empty Text

```json
{
  "error": "Empty text provided"
}
```

#### Rate Limit Exceeded

```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per 60 seconds"
}
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Limit**: 100 requests per minute per IP address
- **Window**: 60 seconds
- **Response**: HTTP 429 with error message when exceeded

## Performance

### Response Times

- **Single Prediction**: ~25-70ms average
- **Batch Prediction**: ~20-30ms per text
- **Health Check**: ~5-10ms
- **Metrics**: ~5-10ms

### Throughput

- **Concurrent Requests**: Supports multiple concurrent requests
- **Batch Processing**: Recommended for multiple predictions
- **CPU Usage**: Optimized for both CPU and GPU inference

## Supported Emotions

The model can detect 12 different emotional states:

1. **anxious** - Worry, nervousness, concern
2. **calm** - Peaceful, relaxed, tranquil
3. **content** - Satisfied, pleased, fulfilled
4. **excited** - Enthusiastic, thrilled, eager
5. **frustrated** - Annoyed, irritated, exasperated
6. **grateful** - Thankful, appreciative, indebted
7. **happy** - Joyful, cheerful, delighted
8. **hopeful** - Optimistic, confident, positive
9. **overwhelmed** - Stressed, burdened, swamped
10. **proud** - Accomplished, satisfied, confident
11. **sad** - Unhappy, sorrowful, down
12. **tired** - Exhausted, weary, fatigued

## Model Performance

### Accuracy Metrics

- **Basic Accuracy**: 100.00% (on validation set)
- **Real-world Accuracy**: 93.75% (on diverse test data)
- **Average Confidence**: 83.9% (across all predictions)

### Model Details

- **Architecture**: BERT-based transformer
- **Version**: 2.0
- **Training Data**: Go Emotions dataset + custom annotations
- **Fine-tuning**: Domain adaptation with focal loss
- **Optimization**: Class weighting and data augmentation

## Usage Examples

### Python

```python
import requests
import json

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I am feeling happy today!"},
    headers={"Content-Type": "application/json"}
)
result = response.json()
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
texts = ["I am happy", "I feel sad", "I am excited"]
response = requests.post(
    "http://localhost:8000/predict_batch",
    json={"texts": texts},
    headers={"Content-Type": "application/json"}
)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['text']} → {pred['predicted_emotion']}")

# Get metrics
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()
print(f"Success rate: {metrics['server_metrics']['success_rate']}")
```

### JavaScript

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'I am feeling happy today!'
    })
});
const result = await response.json();
console.log(`Emotion: ${result.predicted_emotion}`);
console.log(`Confidence: ${result.confidence}`);

// Batch prediction
const texts = ['I am happy', 'I feel sad', 'I am excited'];
const batchResponse = await fetch('http://localhost:8000/predict_batch', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ texts })
});
const batchResults = await batchResponse.json();
batchResults.predictions.forEach(pred => {
    console.log(`${pred.text} → ${pred.predicted_emotion}`);
});
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I am happy", "I feel sad", "I am excited"]}'

# Health check
curl -X GET http://localhost:8000/health

# Metrics
curl -X GET http://localhost:8000/metrics
```

## Monitoring and Logging

### Log Files

- **API Logs**: `api_server.log` (in local_deployment directory)
- **Format**: Structured JSON with timestamps
- **Level**: INFO, WARNING, ERROR

### Metrics Available

- Request counts (total, successful, failed)
- Response times (average, min, max)
- Emotion distribution
- Error counts by type
- Rate limiting statistics
- Uptime and performance metrics

### Health Monitoring

Use the `/health` endpoint for:
- Service health checks
- Load balancer health checks
- Monitoring system integration
- Basic performance metrics

## Best Practices

### Performance

1. **Use Batch Predictions**: For multiple texts, use `/predict_batch` instead of multiple `/predict` calls
2. **Handle Rate Limits**: Implement exponential backoff for 429 responses
3. **Monitor Response Times**: Track performance using the `/metrics` endpoint

### Error Handling

1. **Validate Input**: Ensure text is not empty and properly formatted
2. **Handle Network Errors**: Implement retry logic for transient failures
3. **Check Status Codes**: Always verify HTTP status codes before processing responses

### Security

1. **Input Validation**: Sanitize text input to prevent injection attacks
2. **Rate Limiting**: Respect rate limits to avoid being blocked
3. **HTTPS**: Use HTTPS in production environments

## Troubleshooting

### Common Issues

#### Server Not Starting

```bash
# Check if port 8000 is available
lsof -i :8000

# Check Python environment
python --version
pip list | grep flask
```

#### Model Loading Errors

```bash
# Check model files
ls -la local_deployment/model/

# Check dependencies
pip install -r local_deployment/requirements.txt
```

#### Performance Issues

```bash
# Check system resources
top
htop

# Check API metrics
curl -s http://localhost:8000/metrics | python -m json.tool
```

### Debug Mode

For debugging, you can enable Flask debug mode by modifying `api_server.py`:

```python
app.run(host='0.0.0.0', port=8000, debug=True)
```

**Note**: Debug mode should not be used in production.

## Support

For issues and questions:

1. Check the logs: `tail -f local_deployment/api_server.log`
2. Review metrics: `curl http://localhost:8000/metrics`
3. Test endpoints: Use the provided test scripts
4. Check documentation: `curl http://localhost:8000/`

## Version History

### Version 2.0 (Current)
- Enhanced monitoring and logging
- Rate limiting implementation
- Comprehensive error handling
- Performance optimizations
- Batch processing improvements
- Real-time metrics endpoint

### Version 1.0
- Basic emotion detection
- Single prediction endpoint
- Simple health check
- Local deployment only 