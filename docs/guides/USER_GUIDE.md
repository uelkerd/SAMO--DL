# SAMO Emotion Detection - User Guide

## Welcome to SAMO Emotion Detection

The SAMO Emotion Detection system is a powerful AI tool that analyzes text and identifies emotional states with high accuracy. This guide will help you get started and make the most of the system.

## Quick Start

### 1. Start the API Server

```bash
# Navigate to the project directory
cd SAMO--DL/local_deployment

# Start the server
python api_server.py
```

You should see output like:
```
🔧 Loading emotion detection model...
✅ Model loaded successfully
🌐 Starting enhanced local API server...
🚀 Server starting on http://localhost:8000
```

### 2. Test the System

Open a new terminal and test the API:

```bash
# Health check
curl http://localhost:8000/health

# Test emotion detection
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

### 3. Run Comprehensive Tests

```bash
# Run the test suite
python test_api.py
```

## Understanding the System

### What Emotions Can It Detect?

The system recognizes 12 different emotional states:

| Emotion | Description | Example |
|---------|-------------|---------|
| **anxious** | Worry, nervousness, concern | "I'm worried about the test tomorrow" |
| **calm** | Peaceful, relaxed, tranquil | "I feel peaceful and relaxed" |
| **content** | Satisfied, pleased, fulfilled | "I'm satisfied with how things are going" |
| **excited** | Enthusiastic, thrilled, eager | "I'm so excited for the concert!" |
| **frustrated** | Annoyed, irritated, exasperated | "This is so frustrating, nothing works" |
| **grateful** | Thankful, appreciative, indebted | "I'm grateful for your help" |
| **happy** | Joyful, cheerful, delighted | "I'm feeling really happy today!" |
| **hopeful** | Optimistic, confident, positive | "I'm hopeful about the future" |
| **overwhelmed** | Stressed, burdened, swamped | "I feel overwhelmed with all this work" |
| **proud** | Accomplished, satisfied, confident | "I'm proud of what I've achieved" |
| **sad** | Unhappy, sorrowful, down | "I feel sad about the news" |
| **tired** | Exhausted, weary, fatigued | "I'm so tired after that long day" |

### How Accurate Is It?

- **Basic Accuracy**: 100% (on validation data)
- **Real-world Accuracy**: 93.75% (on diverse test data)
- **Average Confidence**: 83.9% (across all predictions)

### How Fast Is It?

- **Single Prediction**: ~25-70ms average
- **Batch Processing**: ~20-30ms per text
- **Health Check**: ~5-10ms

## Using the API

### Single Prediction

Analyze one piece of text at a time:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

**Response:**
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
  "model_type": "comprehensive_emotion_detection"
}
```

### Batch Prediction

Analyze multiple texts efficiently:

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I am feeling happy today!",
      "I feel sad about the news",
      "I am excited for the party"
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "I am feeling happy today!",
      "predicted_emotion": "happy",
      "confidence": 0.964
    },
    {
      "text": "I feel sad about the news",
      "predicted_emotion": "sad",
      "confidence": 0.965
    },
    {
      "text": "I am excited for the party",
      "predicted_emotion": "excited",
      "confidence": 0.968
    }
  ],
  "count": 3,
  "batch_processing_time_ms": 68.8
}
```

## Programming Examples

### Python

```python
import requests
import json

# Single prediction
def analyze_emotion(text):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        result = response.json()
        return {
            'emotion': result['predicted_emotion'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        }
    else:
        return {'error': f"Request failed: {response.status_code}"}

# Example usage
text = "I am feeling happy today!"
result = analyze_emotion(text)
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
def analyze_emotions_batch(texts):
    response = requests.post(
        "http://localhost:8000/predict_batch",
        json={"texts": texts},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        results = response.json()
        return results['predictions']
    else:
        return {'error': f"Request failed: {response.status_code}"}

# Example batch usage
texts = ["I am happy", "I feel sad", "I am excited"]
results = analyze_emotions_batch(texts)
for i, result in enumerate(results):
    print(f"{i+1}. {result['text']} → {result['predicted_emotion']}")
```

### JavaScript

```javascript
// Single prediction
async function analyzeEmotion(text) {
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });

        if (response.ok) {
            const result = await response.json();
            return {
                emotion: result.predicted_emotion,
                confidence: result.confidence,
                probabilities: result.probabilities
            };
        } else {
            throw new Error(`Request failed: ${response.status}`);
        }
    } catch (error) {
        console.error('Error:', error);
        return { error: error.message };
    }
}

// Example usage
const text = "I am feeling happy today!";
analyzeEmotion(text).then(result => {
    console.log(`Emotion: ${result.emotion}`);
    console.log(`Confidence: ${result.confidence.toFixed(3)}`);
});

// Batch prediction
async function analyzeEmotionsBatch(texts) {
    try {
        const response = await fetch('http://localhost:8000/predict_batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ texts })
        });

        if (response.ok) {
            const results = await response.json();
            return results.predictions;
        } else {
            throw new Error(`Request failed: ${response.status}`);
        }
    } catch (error) {
        console.error('Error:', error);
        return { error: error.message };
    }
}

// Example batch usage
const texts = ['I am happy', 'I feel sad', 'I am excited'];
analyzeEmotionsBatch(texts).then(results => {
    results.forEach((result, index) => {
        console.log(`${index + 1}. ${result.text} → ${result.predicted_emotion}`);
    });
});
```

### Node.js

```javascript
const axios = require('axios');

// Single prediction
async function analyzeEmotion(text) {
    try {
        const response = await axios.post('http://localhost:8000/predict', {
            text: text
        }, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        return {
            emotion: response.data.predicted_emotion,
            confidence: response.data.confidence,
            probabilities: response.data.probabilities
        };
    } catch (error) {
        console.error('Error:', error.message);
        return { error: error.message };
    }
}

// Batch prediction
async function analyzeEmotionsBatch(texts) {
    try {
        const response = await axios.post('http://localhost:8000/predict_batch', {
            texts: texts
        }, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        return response.data.predictions;
    } catch (error) {
        console.error('Error:', error.message);
        return { error: error.message };
    }
}

// Example usage
const text = "I am feeling happy today!";
analyzeEmotion(text).then(result => {
    console.log(`Emotion: ${result.emotion}`);
    console.log(`Confidence: ${result.confidence.toFixed(3)}`);
});
```

## Monitoring and Metrics

### Check System Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model_version": "2.0",
  "uptime_seconds": 1234.5,
  "metrics": {
    "total_requests": 150,
    "successful_requests": 145,
    "failed_requests": 5,
    "average_response_time_ms": 65.2
  }
}
```

### Get Detailed Metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**
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
    "anxious": 12
  },
  "error_counts": {
    "missing_text": 2,
    "empty_text": 2,
    "prediction_error": 1
  }
}
```

## Best Practices

### 1. Use Batch Processing for Multiple Texts

Instead of making multiple single requests:

```python
# ❌ Inefficient
for text in texts:
    result = analyze_emotion(text)

# ✅ Efficient
results = analyze_emotions_batch(texts)
```

### 2. Handle Rate Limits

The API limits requests to 100 per minute per IP. Implement exponential backoff:

```python
import time
import random

def analyze_emotion_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 429:  # Rate limited
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            elif response.status_code == 200:
                return response.json()
            else:
                return {'error': f"Request failed: {response.status_code}"}

        except Exception as e:
            if attempt == max_retries - 1:
                return {'error': str(e)}
            time.sleep(1)

    return {'error': 'Max retries exceeded'}
```

### 3. Validate Input

Always validate text input before sending:

```python
def validate_text(text):
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"

    if len(text.strip()) == 0:
        return False, "Text cannot be empty or whitespace only"

    if len(text) > 1000:  # Adjust limit as needed
        return False, "Text too long (max 1000 characters)"

    return True, "Valid"

# Usage
text = "I am feeling happy today!"
is_valid, message = validate_text(text)
if is_valid:
    result = analyze_emotion(text)
else:
    print(f"Invalid input: {message}")
```

### 4. Monitor Performance

Track response times and success rates:

```python
import time
from collections import defaultdict

class EmotionAnalyzer:
    def __init__(self):
        self.stats = defaultdict(list)

    def analyze_with_monitoring(self, text):
        start_time = time.time()

        try:
            result = analyze_emotion(text)
            response_time = (time.time() - start_time) * 1000

            self.stats['response_times'].append(response_time)
            self.stats['success_count'] += 1

            return result

        except Exception as e:
            self.stats['error_count'] += 1
            raise e

    def get_stats(self):
        if self.stats['response_times']:
            avg_time = sum(self.stats['response_times']) / len(self.stats['response_times'])
            return {
                'avg_response_time_ms': avg_time,
                'success_count': self.stats['success_count'],
                'error_count': self.stats['error_count']
            }
        return {'error': 'No data available'}
```

## Error Handling

### Common Error Responses

**Missing Text (400)**
```json
{
  "error": "No text provided"
}
```

**Empty Text (400)**
```json
{
  "error": "Empty text provided"
}
```

**Rate Limit Exceeded (429)**
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 100 requests per 60 seconds"
}
```

**Server Error (500)**
```json
{
  "error": "Internal server error"
}
```

### Error Handling Example

```python
def safe_analyze_emotion(text):
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10  # 10 second timeout
        )

        if response.status_code == 200:
            return {'success': True, 'data': response.json()}
        elif response.status_code == 400:
            return {'success': False, 'error': 'Invalid input', 'details': response.json()}
        elif response.status_code == 429:
            return {'success': False, 'error': 'Rate limited', 'details': response.json()}
        elif response.status_code == 500:
            return {'success': False, 'error': 'Server error', 'details': response.json()}
        else:
            return {'success': False, 'error': f'Unexpected status: {response.status_code}'}

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timeout'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Connection failed'}
    except Exception as e:
        return {'success': False, 'error': f'Unexpected error: {str(e)}'}

# Usage
result = safe_analyze_emotion("I am feeling happy today!")
if result['success']:
    print(f"Emotion: {result['data']['predicted_emotion']}")
else:
    print(f"Error: {result['error']}")
```

## Advanced Usage

### Custom Confidence Thresholds

```python
def analyze_emotion_with_threshold(text, confidence_threshold=0.8):
    result = analyze_emotion(text)

    if result['confidence'] >= confidence_threshold:
        return {
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'reliable': True
        }
    else:
        return {
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'reliable': False,
            'message': f'Low confidence ({result["confidence"]:.3f} < {confidence_threshold})'
        }
```

### Emotion Trend Analysis

```python
def analyze_emotion_trend(texts):
    """Analyze emotional trend across multiple texts."""
    results = analyze_emotions_batch(texts)

    emotion_counts = defaultdict(int)
    total_confidence = 0

    for result in results:
        emotion_counts[result['predicted_emotion']] += 1
        total_confidence += result['confidence']

    avg_confidence = total_confidence / len(results) if results else 0

    # Find dominant emotion
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

    return {
        'dominant_emotion': dominant_emotion,
        'emotion_distribution': dict(emotion_counts),
        'average_confidence': avg_confidence,
        'text_count': len(texts)
    }

# Example usage
texts = [
    "I'm feeling great today!",
    "This is amazing news!",
    "I'm so excited about this!",
    "I feel really happy right now"
]

trend = analyze_emotion_trend(texts)
print(f"Dominant emotion: {trend['dominant_emotion']}")
print(f"Average confidence: {trend['average_confidence']:.3f}")
```

## Troubleshooting

### Common Issues

**1. Server won't start**
```bash
# Check if port 8000 is available
lsof -i :8000

# Check Python environment
python --version
pip list | grep flask
```

**2. Model loading errors**
```bash
# Check model files
ls -la local_deployment/model/

# Reinstall dependencies
pip install -r requirements.txt
```

**3. High response times**
```bash
# Check system resources
htop

# Check API metrics
curl http://localhost:8000/metrics
```

**4. Rate limiting issues**
```bash
# Check current rate limit status
curl http://localhost:8000/metrics | grep rate_limiting

# Wait and retry
sleep 60  # Wait 1 minute
```

### Getting Help

1. **Check the logs**
   ```bash
   tail -f local_deployment/api_server.log
   ```

2. **Run diagnostics**
   ```bash
   python test_api.py
   ```

3. **Check system status**
   ```bash
   curl http://localhost:8000/health
   ```

## Performance Tips

### 1. Optimize Text Length

- **Too short**: "happy" (may be ambiguous)
- **Good**: "I am feeling happy today!" (clear and specific)
- **Too long**: Very long texts may be truncated

### 2. Use Appropriate Language

- **Clear**: "I am feeling anxious about the presentation"
- **Ambiguous**: "I feel something" (unclear emotion)

### 3. Batch Processing

For multiple texts, always use batch processing:

```python
# Process 100 texts efficiently
texts = ["Text 1", "Text 2", ..., "Text 100"]
results = analyze_emotions_batch(texts)
```

### 4. Cache Results

For repeated analysis of the same text:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_analyze_emotion(text):
    return analyze_emotion(text)
```

## Conclusion

The SAMO Emotion Detection system provides powerful, accurate emotion analysis with high performance and reliability. By following the best practices outlined in this guide, you can effectively integrate emotion detection into your applications.

For additional support:
- Check the API documentation: `curl http://localhost:8000/`
- Review the deployment guide for production setup
- Monitor system metrics for optimal performance

Happy emotion analyzing! 🎉
