# 📚 API Reference

*Complete API documentation for SAMO Brain's emotion detection and AI analysis services.*

## 🔑 Authentication

### API Key Authentication

```bash
# Include API key in Authorization header
curl -X POST "https://api.samobrain.com/predict" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great today!"}'
```

### JWT Token Authentication

```bash
# Include JWT token in Authorization header
curl -X POST "https://api.samobrain.com/predict" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great today!"}'
```

## 🧠 Core Endpoints

### POST /predict

Analyze emotion from a single text input.

**Request:**
```json
{
  "text": "I am feeling great today!"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "predicted_emotion": "happy",
    "confidence": 0.8942,
    "probabilities": {
      "happy": 0.8942,
      "excited": 0.0451,
      "calm": 0.0321,
      "content": 0.0156,
      "grateful": 0.0089,
      "hopeful": 0.0041
    },
    "prediction_time_ms": 45.2
  },
  "processing_time_ms": 67.8,
  "cached": false,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

**Error Responses:**
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Text input is required",
    "details": {
      "field": "text",
      "constraint": "required"
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### POST /predict_batch

Analyze emotions from multiple text inputs efficiently.

**Request:**
```json
{
  "texts": [
    "I am feeling great today!",
    "This is so frustrating!",
    "I'm really excited about this project"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "predictions": [
      {
        "text": "I am feeling great today!",
        "predicted_emotion": "happy",
        "confidence": 0.8942,
        "probabilities": {
          "happy": 0.8942,
          "excited": 0.0451,
          "calm": 0.0321,
          "content": 0.0156,
          "grateful": 0.0089,
          "hopeful": 0.0041
        }
      },
      {
        "text": "This is so frustrating!",
        "predicted_emotion": "frustrated",
        "confidence": 0.9234,
        "probabilities": {
          "frustrated": 0.9234,
          "anxious": 0.0456,
          "overwhelmed": 0.0210,
          "sad": 0.0100
        }
      },
      {
        "text": "I'm really excited about this project",
        "predicted_emotion": "excited",
        "confidence": 0.8765,
        "probabilities": {
          "excited": 0.8765,
          "happy": 0.0987,
          "hopeful": 0.0156,
          "proud": 0.0092
        }
      }
    ],
    "batch_processing_time_ms": 89.3
  },
  "processing_time_ms": 112.5,
  "cached": false,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### POST /analyze

Comprehensive text analysis with multiple AI services.

**Request:**
```json
{
  "text": "I am feeling great today and accomplished so much!",
  "services": ["emotion", "summarization", "sentiment"],
  "options": {
    "emotion": {
      "include_probabilities": true,
      "confidence_threshold": 0.7
    },
    "summarization": {
      "max_length": 100,
      "style": "concise"
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "input_text": "I am feeling great today and accomplished so much!",
    "services_used": ["emotion", "summarization", "sentiment"],
    "results": {
      "emotion": {
        "predicted_emotion": "happy",
        "confidence": 0.8942,
        "probabilities": {
          "happy": 0.8942,
          "excited": 0.0451,
          "calm": 0.0321,
          "content": 0.0156,
          "grateful": 0.0089,
          "hopeful": 0.0041
        }
      },
      "summarization": {
        "summary": "User expresses positive feelings about their accomplishments.",
        "key_points": ["feeling great", "accomplished much"],
        "summary_length": 65
      },
      "sentiment": {
        "overall_sentiment": "positive",
        "sentiment_score": 0.85,
        "confidence": 0.92
      }
    },
    "processing_time_ms": 156.7
  },
  "processing_time_ms": 189.2,
  "cached": false,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## 🎤 Voice Processing Endpoints

### POST /voice/analyze

Analyze emotion from audio input.

**Request:**
```bash
curl -X POST "https://api.samobrain.com/voice/analyze" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@voice_sample.wav" \
  -F "options={\"language\": \"en\", \"include_transcript\": true}"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "audio_processing": {
      "transcript": "I am feeling great today!",
      "confidence": 0.95,
      "language": "en",
      "duration_seconds": 2.3
    },
    "emotion_analysis": {
      "predicted_emotion": "happy",
      "confidence": 0.8942,
      "probabilities": {
        "happy": 0.8942,
        "excited": 0.0451,
        "calm": 0.0321,
        "content": 0.0156,
        "grateful": 0.0089,
        "hopeful": 0.0041
      }
    },
    "processing_time_ms": 234.5
  },
  "processing_time_ms": 289.1,
  "cached": false,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### POST /voice/transcribe

Convert audio to text with emotion analysis.

**Request:**
```bash
curl -X POST "https://api.samobrain.com/voice/transcribe" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@voice_sample.wav" \
  -F "options={\"language\": \"en\", \"timestamps\": true}"
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "transcript": "I am feeling great today!",
    "segments": [
      {
        "start": 0.0,
        "end": 2.3,
        "text": "I am feeling great today!",
        "confidence": 0.95
      }
    ],
    "language": "en",
    "duration_seconds": 2.3,
    "word_count": 6,
    "processing_time_ms": 156.7
  },
  "processing_time_ms": 189.2,
  "cached": false,
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## 📊 Monitoring Endpoints

### GET /health

Check API health and system status.

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12.3,
      "timestamp": "2024-01-15T10:30:45.123Z"
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2.1,
      "timestamp": "2024-01-15T10:30:45.123Z"
    },
    "model": {
      "status": "healthy",
      "response_time_ms": 45.6,
      "timestamp": "2024-01-15T10:30:45.123Z"
    },
    "api": {
      "status": "healthy",
      "response_time_ms": 8.9,
      "timestamp": "2024-01-15T10:30:45.123Z"
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### GET /metrics

Get detailed system metrics and performance data.

**Response:**
```json
{
  "status": "success",
  "data": {
    "system_metrics": {
      "uptime_seconds": 86400,
      "memory_usage_mb": 512.3,
      "cpu_usage_percent": 23.4,
      "active_connections": 45
    },
    "request_metrics": {
      "total_requests": 15420,
      "requests_per_minute": 12.3,
      "average_response_time_ms": 67.8,
      "error_rate_percent": 0.5
    },
    "emotion_metrics": {
      "total_predictions": 12340,
      "predictions_per_minute": 10.2,
      "average_confidence": 0.85,
      "emotion_distribution": {
        "happy": 0.25,
        "sad": 0.15,
        "excited": 0.20,
        "calm": 0.18,
        "frustrated": 0.12,
        "anxious": 0.08,
        "grateful": 0.02
      }
    },
    "cache_metrics": {
      "cache_hit_rate_percent": 78.5,
      "cache_size_mb": 256.7,
      "cache_evictions": 45
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### GET /metrics/prometheus

Get Prometheus-compatible metrics.

**Response:**
```
# HELP samo_brain_requests_total Total number of requests
# TYPE samo_brain_requests_total counter
samo_brain_requests_total{endpoint="/predict",method="POST",status_code="200"} 12340
samo_brain_requests_total{endpoint="/predict",method="POST",status_code="400"} 23
samo_brain_requests_total{endpoint="/predict",method="POST",status_code="429"} 12

# HELP samo_brain_request_duration_seconds Request duration in seconds
# TYPE samo_brain_request_duration_seconds histogram
samo_brain_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.1"} 8900
samo_brain_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="0.5"} 12340
samo_brain_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="1.0"} 12340
samo_brain_request_duration_seconds_bucket{endpoint="/predict",method="POST",le="+Inf"} 12340

# HELP samo_brain_emotion_predictions_total Total emotion predictions
# TYPE samo_brain_emotion_predictions_total counter
samo_brain_emotion_predictions_total{emotion="happy",confidence_bucket="0.9-1.0"} 4567
samo_brain_emotion_predictions_total{emotion="sad",confidence_bucket="0.8-0.9"} 2345
samo_brain_emotion_predictions_total{emotion="excited",confidence_bucket="0.7-0.8"} 3456

# HELP samo_brain_active_connections Number of active connections
# TYPE samo_brain_active_connections gauge
samo_brain_active_connections 45

# HELP samo_brain_model_memory_bytes Memory usage of AI models in bytes
# TYPE samo_brain_model_memory_bytes gauge
samo_brain_model_memory_bytes 536870912
```

## 🔧 Configuration Endpoints

### GET /config

Get current API configuration.

**Response:**
```json
{
  "status": "success",
  "data": {
    "api_version": "1.0.0",
    "supported_emotions": [
      "happy", "sad", "excited", "calm", "frustrated", "anxious",
      "grateful", "hopeful", "overwhelmed", "proud", "content", "tired"
    ],
    "rate_limits": {
      "standard": {
        "requests_per_minute": 100,
        "requests_per_hour": 1000,
        "requests_per_day": 10000
      },
      "premium": {
        "requests_per_minute": 500,
        "requests_per_hour": 5000,
        "requests_per_day": 50000
      }
    },
    "supported_languages": ["en", "es", "fr", "de", "it"],
    "max_text_length": 10000,
    "max_batch_size": 100,
    "supported_audio_formats": ["wav", "mp3", "m4a", "flac"],
    "max_audio_duration_seconds": 300
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### PUT /config/rate_limit

Update rate limit configuration (Admin only).

**Request:**
```json
{
  "tier": "premium",
  "user_id": "user_123",
  "limits": {
    "requests_per_minute": 500,
    "requests_per_hour": 5000,
    "requests_per_day": 50000
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "message": "Rate limit updated successfully",
    "user_id": "user_123",
    "new_limits": {
      "requests_per_minute": 500,
      "requests_per_hour": 5000,
      "requests_per_day": 50000
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## 📈 Analytics Endpoints

### GET /analytics/emotions

Get emotion analytics and trends.

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `user_id`: Filter by user ID
- `emotion`: Filter by specific emotion
- `group_by`: Group by hour, day, week, month

**Response:**
```json
{
  "status": "success",
  "data": {
    "period": {
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-01-15T23:59:59Z"
    },
    "summary": {
      "total_predictions": 15420,
      "unique_users": 1234,
      "average_confidence": 0.85
    },
    "emotion_distribution": {
      "happy": {
        "count": 3855,
        "percentage": 25.0,
        "average_confidence": 0.87
      },
      "sad": {
        "count": 2313,
        "percentage": 15.0,
        "average_confidence": 0.82
      },
      "excited": {
        "count": 3084,
        "percentage": 20.0,
        "average_confidence": 0.89
      }
    },
    "trends": {
      "daily": [
        {
          "date": "2024-01-01",
          "total_predictions": 1023,
          "emotion_distribution": {
            "happy": 256,
            "sad": 154,
            "excited": 205
          }
        }
      ]
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### GET /analytics/performance

Get performance analytics.

**Response:**
```json
{
  "status": "success",
  "data": {
    "response_times": {
      "average_ms": 67.8,
      "p95_ms": 120.5,
      "p99_ms": 234.1,
      "min_ms": 12.3,
      "max_ms": 456.7
    },
    "throughput": {
      "requests_per_second": 12.3,
      "requests_per_minute": 738,
      "requests_per_hour": 44280
    },
    "error_rates": {
      "overall_percent": 0.5,
      "by_endpoint": {
        "/predict": 0.3,
        "/predict_batch": 0.7,
        "/voice/analyze": 1.2
      },
      "by_error_type": {
        "validation_error": 0.2,
        "rate_limit_exceeded": 0.1,
        "internal_error": 0.2
      }
    },
    "cache_performance": {
      "hit_rate_percent": 78.5,
      "miss_rate_percent": 21.5,
      "average_cache_time_ms": 2.1
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## 🚨 Error Handling

### Error Response Format

All error responses follow this standard format:

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional_error_details",
      "suggestion": "How to fix the error"
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid or missing authentication |
| `AUTHORIZATION_FAILED` | 403 | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `TEXT_TOO_LONG` | 400 | Text exceeds maximum length |
| `BATCH_TOO_LARGE` | 400 | Batch size exceeds limit |
| `AUDIO_TOO_LONG` | 400 | Audio duration exceeds limit |
| `UNSUPPORTED_FORMAT` | 400 | Unsupported file format |
| `MODEL_ERROR` | 500 | AI model processing error |
| `DATABASE_ERROR` | 500 | Database connection error |
| `CACHE_ERROR` | 500 | Cache system error |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Rate Limiting

When rate limits are exceeded:

```json
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later.",
    "details": {
      "limit": "100 requests per minute",
      "reset_time": "2024-01-15T10:31:00Z",
      "retry_after_seconds": 15
    }
  },
  "timestamp": "2024-01-15T10:30:45.123Z"
}
```

## 📝 Request/Response Examples

### Python Examples

```python
import requests
import json

class SAMOBrainClient:
    def __init__(self, api_key, base_url="https://api.samobrain.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"ApiKey {api_key}",
            "Content-Type": "application/json"
        }

    def predict_emotion(self, text):
        """Predict emotion from text."""
        url = f"{self.base_url}/predict"
        payload = {"text": text}

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

    def predict_batch(self, texts):
        """Predict emotions from multiple texts."""
        url = f"{self.base_url}/predict_batch"
        payload = {"texts": texts}

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()

    def analyze_voice(self, audio_file_path):
        """Analyze emotion from audio file."""
        url = f"{self.base_url}/voice/analyze"

        with open(audio_file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            response = requests.post(url, headers={"Authorization": f"ApiKey {self.api_key}"}, files=files)

        response.raise_for_status()
        return response.json()

# Usage example
client = SAMOBrainClient("your_api_key_here")

# Single prediction
result = client.predict_emotion("I am feeling great today!")
print(f"Predicted emotion: {result['data']['predicted_emotion']}")

# Batch prediction
texts = ["I am happy!", "I am sad.", "I am excited!"]
batch_result = client.predict_batch(texts)
for prediction in batch_result['data']['predictions']:
    print(f"Text: {prediction['text']} -> Emotion: {prediction['predicted_emotion']}")
```

### JavaScript Examples

```javascript
class SAMOBrainClient {
    constructor(apiKey, baseUrl = 'https://api.samobrain.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }

    async predictEmotion(text) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Authorization': `ApiKey ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async predictBatch(texts) {
        const response = await fetch(`${this.baseUrl}/predict_batch`, {
            method: 'POST',
            headers: {
                'Authorization': `ApiKey ${this.apiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async analyzeVoice(audioFile) {
        const formData = new FormData();
        formData.append('audio', audioFile);

        const response = await fetch(`${this.baseUrl}/voice/analyze`, {
            method: 'POST',
            headers: {
                'Authorization': `ApiKey ${this.apiKey}`
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
}

// Usage example
const client = new SAMOBrainClient('your_api_key_here');

// Single prediction
client.predictEmotion("I am feeling great today!")
    .then(result => {
        console.log(`Predicted emotion: ${result.data.predicted_emotion}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });

// Batch prediction
const texts = ["I am happy!", "I am sad.", "I am excited!"];
client.predictBatch(texts)
    .then(result => {
        result.data.predictions.forEach(prediction => {
            console.log(`Text: ${prediction.text} -> Emotion: ${prediction.predicted_emotion}`);
        });
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

### cURL Examples

```bash
# Single emotion prediction
curl -X POST "https://api.samobrain.com/predict" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling great today!"}'

# Batch emotion prediction
curl -X POST "https://api.samobrain.com/predict_batch" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I am feeling great today!",
      "This is so frustrating!",
      "I am really excited about this project"
    ]
  }'

# Voice analysis
curl -X POST "https://api.samobrain.com/voice/analyze" \
  -H "Authorization: ApiKey YOUR_API_KEY" \
  -F "audio=@voice_sample.wav"

# Health check
curl -X GET "https://api.samobrain.com/health" \
  -H "Authorization: ApiKey YOUR_API_KEY"

# Get metrics
curl -X GET "https://api.samobrain.com/metrics" \
  -H "Authorization: ApiKey YOUR_API_KEY"
```

## 🔄 WebSocket API

### WebSocket Connection

```javascript
// Connect to WebSocket API
const ws = new WebSocket('wss://api.samobrain.com/ws');

// Authenticate connection
ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'auth',
        api_key: 'your_api_key_here'
    }));
};

// Handle incoming messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'auth_success':
            console.log('WebSocket authenticated successfully');
            break;
        case 'emotion_prediction':
            console.log('Emotion prediction:', data.data);
            break;
        case 'error':
            console.error('WebSocket error:', data.error);
            break;
    }
};

// Send emotion prediction request
ws.send(JSON.stringify({
    type: 'predict_emotion',
    text: 'I am feeling great today!'
}));

// Send batch prediction request
ws.send(JSON.stringify({
    type: 'predict_batch',
    texts: ['I am happy!', 'I am sad.', 'I am excited!']
}));
```

---

*This API reference provides comprehensive documentation for all SAMO Brain endpoints, including authentication, request/response formats, error handling, and practical examples in multiple programming languages.*
