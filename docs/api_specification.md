# SAMO Deep Learning - API Specification

## 📋 Overview

This document provides comprehensive API specifications for the SAMO Deep Learning system. It serves as the definitive reference for Web Development teams and other services integrating with our AI capabilities.

## 🔑 Authentication & Security

### Authentication Methods

The SAMO API supports two authentication methods:

#### 1. API Key Authentication (Recommended)

```bash
curl -X POST "https://api.samo.ai/v1/emotions/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"text": "I feel so happy about this achievement!"}'
```

#### 2. JWT Bearer Token

```bash
curl -X POST "https://api.samo.ai/v1/emotions/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{"text": "I feel so happy about this achievement!"}'
```

### Rate Limiting

- **Standard Tier**: 100 requests/minute
- **Premium Tier**: 1000 requests/minute
- **Enterprise Tier**: Custom limits

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

## 📊 API Endpoints

### Emotion Analysis

#### `POST /v1/emotions/analyze`

Analyzes text for emotional content and returns detailed emotion classifications.

**Request Schema:**

```json
{
  "text": "string (required, 1-5000 characters)",
  "options": {
    "threshold": "float (optional, 0.0-1.0, default: 0.6)",
    "top_k": "integer (optional, default: 3)",
    "include_probabilities": "boolean (optional, default: true)",
    "temperature": "float (optional, 0.1-2.0, default: 1.0)"
  },
  "user_id": "string (optional, for personalization)",
  "session_id": "string (optional, for conversation context)"
}
```

**Response Schema:**

```json
{
  "request_id": "string (UUID for tracking)",
  "timestamp": "string (ISO 8601 format)",
  "emotions": {
    "primary": "string (primary emotion category)",
    "secondary": ["string (additional emotions)"],
    "scores": {
      "joy": 0.92,
      "sadness": 0.03,
      "anger": 0.01,
      "fear": 0.02,
      "surprise": 0.15,
      // All 28 emotion categories included
    }
  },
  "intensity": 0.85,
  "metadata": {
    "processing_time_ms": 156,
    "model_version": "bert-emotion-v2.1",
    "confidence": 0.94
  }
}
```

**Example:**

```bash
curl -X POST "https://api.samo.ai/v1/emotions/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "text": "I just got promoted at work! I can't believe it, I've been working so hard for this!",
    "options": {
      "threshold": 0.5,
      "top_k": 3
    }
  }'
```

**Response:**

```json
{
  "request_id": "f7cdf982-1a5e-4d11-b8d7-e93e7c24d2d4",
  "timestamp": "2025-07-24T08:12:34.567Z",
  "emotions": {
    "primary": "joy",
    "secondary": ["surprise", "pride"],
    "scores": {
      "joy": 0.92,
      "surprise": 0.78,
      "pride": 0.65,
      "gratitude": 0.45,
      "optimism": 0.38,
      "relief": 0.22,
      "love": 0.18,
      "admiration": 0.15,
      "approval": 0.12,
      "caring": 0.09,
      "excitement": 0.08,
      "amusement": 0.05,
      "realization": 0.04,
      "sadness": 0.03,
      "fear": 0.02,
      "nervousness": 0.02,
      "anger": 0.01,
      "annoyance": 0.01,
      "disappointment": 0.01,
      "embarrassment": 0.01,
      "grief": 0.01,
      "remorse": 0.01,
      "confusion": 0.01,
      "curiosity": 0.01,
      "disgust": 0.01,
      "desire": 0.01,
      "disapproval": 0.01,
      "neutral": 0.01
    }
  },
  "intensity": 0.85,
  "metadata": {
    "processing_time_ms": 156,
    "model_version": "bert-emotion-v2.1",
    "confidence": 0.94
  }
}
```

### Text Summarization

#### `POST /v1/summarize`

Generates concise summaries of longer text passages.

**Request Schema:**

```json
{
  "text": "string (required, 100-50000 characters)",
  "options": {
    "max_length": "integer (optional, default: 150)",
    "min_length": "integer (optional, default: 50)",
    "style": "string (optional, enum: ['concise', 'detailed', 'bullets'], default: 'concise')",
    "focus": "string (optional, enum: ['general', 'emotional', 'action_items'], default: 'general')"
  }
}
```

**Response Schema:**

```json
{
  "request_id": "string (UUID for tracking)",
  "timestamp": "string (ISO 8601 format)",
  "summary": "string (generated summary)",
  "metadata": {
    "processing_time_ms": 345,
    "model_version": "t5-summarizer-v1.2",
    "original_length": 1250,
    "summary_length": 142,
    "compression_ratio": 0.11
  }
}
```

**Example:**

```bash
curl -X POST "https://api.samo.ai/v1/summarize" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "text": "Today was a challenging day at work. The project deadline was moved up by two weeks, which means our team will need to work overtime to meet the new timeline. I had a productive conversation with my manager about resource allocation, and we agreed to bring in two additional developers from another team. Despite the pressure, I feel confident that we can deliver quality work on time. The team morale is surprisingly good, with everyone committed to making this work. I'm planning to organize a team dinner next week to show my appreciation for their dedication.",
    "options": {
      "max_length": 100,
      "style": "concise",
      "focus": "emotional"
    }
  }'
```

**Response:**

```json
{
  "request_id": "a1b2c3d4-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
  "timestamp": "2025-07-24T08:15:45.123Z",
  "summary": "Feeling confident despite project deadline being moved up. Team morale is good with everyone committed. Planning appreciation dinner for team's dedication.",
  "metadata": {
    "processing_time_ms": 287,
    "model_version": "t5-summarizer-v1.2",
    "original_length": 521,
    "summary_length": 98,
    "compression_ratio": 0.19
  }
}
```

### Voice Transcription

#### `POST /v1/voice/transcribe`

Transcribes audio files to text and optionally performs emotion analysis.

**Request Schema (multipart/form-data):**

```
file: Binary audio file (required, formats: mp3, wav, m4a, max 15MB)
language: string (optional, ISO 639-1 code, default: "en")
analyze_emotions: boolean (optional, default: false)
timestamp_granularity: string (optional, enum: ["none", "sentence", "word"], default: "sentence")
```

**Response Schema:**

```json
{
  "request_id": "string (UUID for tracking)",
  "timestamp": "string (ISO 8601 format)",
  "transcription": "string (full transcription text)",
  "segments": [
    {
      "text": "string (segment text)",
      "start_time": 0.0,
      "end_time": 4.2,
      "confidence": 0.98
    }
  ],
  "emotions": {
    // Only included if analyze_emotions=true
    // Same structure as emotion analysis response
  },
  "metadata": {
    "processing_time_ms": 2156,
    "model_version": "whisper-large-v2",
    "audio_duration_seconds": 45.3,
    "language_detected": "en"
  }
}
```

**Example:**

```bash
curl -X POST "https://api.samo.ai/v1/voice/transcribe" \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@recording.mp3" \
  -F "language=en" \
  -F "analyze_emotions=true"
```

**Response:**

```json
{
  "request_id": "c7d8e9f0-1a2b-3c4d-5e6f-7g8h9i0j1k2l",
  "timestamp": "2025-07-24T08:20:12.789Z",
  "transcription": "I'm really excited about our new project. The team has been working hard and I think we're making great progress. I'm a bit concerned about the timeline, but I believe we can make it work.",
  "segments": [
    {
      "text": "I'm really excited about our new project.",
      "start_time": 0.0,
      "end_time": 2.4,
      "confidence": 0.98
    },
    {
      "text": "The team has been working hard and I think we're making great progress.",
      "start_time": 2.4,
      "end_time": 6.1,
      "confidence": 0.97
    },
    {
      "text": "I'm a bit concerned about the timeline, but I believe we can make it work.",
      "start_time": 6.2,
      "end_time": 9.8,
      "confidence": 0.95
    }
  ],
  "emotions": {
    "primary": "optimism",
    "secondary": ["excitement", "concern"],
    "scores": {
      "optimism": 0.82,
      "excitement": 0.76,
      "concern": 0.45,
      "joy": 0.38,
      // Additional emotions omitted for brevity
    }
  },
  "metadata": {
    "processing_time_ms": 1856,
    "model_version": "whisper-large-v2",
    "audio_duration_seconds": 9.8,
    "language_detected": "en"
  }
}
```

### Unified AI API

#### `POST /v1/analyze`

Comprehensive endpoint that performs multiple analyses on text input.

**Request Schema:**

```json
{
  "text": "string (required, 1-10000 characters)",
  "analyses": ["emotions", "summary", "topics", "sentiment", "entities"],
  "options": {
    "emotions": {
      "threshold": 0.6,
      "top_k": 3
    },
    "summary": {
      "max_length": 100,
      "style": "concise"
    },
    "topics": {
      "max_topics": 5
    },
    "sentiment": {
      "detailed": true
    },
    "entities": {
      "types": ["person", "organization", "location", "date"]
    }
  }
}
```

**Response Schema:**

```json
{
  "request_id": "string (UUID for tracking)",
  "timestamp": "string (ISO 8601 format)",
  "analyses": {
    "emotions": {
      // Emotion analysis results
    },
    "summary": {
      // Summary results
    },
    "topics": {
      // Topic analysis results
    },
    "sentiment": {
      // Sentiment analysis results
    },
    "entities": {
      // Entity extraction results
    }
  },
  "metadata": {
    "processing_time_ms": 478,
    "models_used": ["bert-emotion-v2.1", "t5-summarizer-v1.2"]
  }
}
```

## 📝 Error Handling

### Error Response Format

All API errors follow this consistent format:

```json
{
  "error": {
    "code": "string (error code)",
    "message": "string (human-readable message)",
    "details": {
      // Additional error context
    },
    "request_id": "string (for support reference)"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `authentication_error` | 401 | Invalid API key or token | Check your authentication credentials |
| `permission_denied` | 403 | Insufficient permissions | Upgrade your plan or request access |
| `rate_limit_exceeded` | 429 | Too many requests | Reduce request frequency or upgrade plan |
| `invalid_request` | 400 | Malformed request | Check request format against schema |
| `text_too_long` | 400 | Input text exceeds limits | Reduce input length |
| `text_too_short` | 400 | Input text too short for analysis | Provide more text |
| `unsupported_language` | 400 | Language not supported | Check supported languages |
| `model_error` | 500 | Model inference failed | Retry or contact support |
| `service_unavailable` | 503 | Service temporarily unavailable | Retry after a short delay |

### Error Examples

**Invalid API Key:**

```json
{
  "error": {
    "code": "authentication_error",
    "message": "Invalid API key provided",
    "details": {
      "hint": "Ensure you're using the correct API key for this environment"
    },
    "request_id": "d1e2f3g4-5h6i-7j8k-9l0m-1n2o3p4q5r6s"
  }
}
```

**Rate Limit Exceeded:**

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit of 100 requests per minute exceeded",
    "details": {
      "rate_limit": 100,
      "retry_after_seconds": 45
    },
    "request_id": "s6r5q4p3-o2n1-m0l9-k8j7-i6h5g4f3e2d1"
  }
}
```

## 🔄 Webhooks

### Webhook Events

For long-running processes or asynchronous notifications, SAMO provides webhooks:

| Event Type | Description |
|------------|-------------|
| `analysis.completed` | Analysis job has completed |
| `model.updated` | Model has been updated |
| `error.occurred` | Error occurred during processing |

### Webhook Payload

```json
{
  "event_type": "string (event type)",
  "timestamp": "string (ISO 8601 format)",
  "request_id": "string (original request ID)",
  "data": {
    // Event-specific data
  }
}
```

### Webhook Configuration

```bash
curl -X POST "https://api.samo.ai/v1/webhooks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "url": "https://your-server.com/webhook-endpoint",
    "events": ["analysis.completed", "error.occurred"],
    "secret": "your_webhook_signing_secret"
  }'
```

## 🧩 SDK Integration

### Python SDK

```python
from samo_client import SamoAI

# Initialize client
samo = SamoAI(api_key="your_api_key_here")

# Analyze emotions
emotions = samo.emotions.analyze(
    text="I'm feeling really excited about this new opportunity!",
    threshold=0.5,
    top_k=3
)

print(f"Primary emotion: {emotions.primary}")
print(f"Secondary emotions: {emotions.secondary}")
print(f"Confidence: {emotions.metadata.confidence}")

# Generate summary
summary = samo.summarize(
    text="Long text to summarize...",
    max_length=100,
    style="concise"
)

print(f"Summary: {summary.text}")
```

### JavaScript SDK

```javascript
import { SamoAI } from 'samo-ai';

// Initialize client
const samo = new SamoAI('your_api_key_here');

// Analyze emotions
samo.emotions.analyze({
  text: "I'm feeling really excited about this new opportunity!",
  options: {
    threshold: 0.5,
    top_k: 3
  }
})
.then(result => {
  console.log(`Primary emotion: ${result.emotions.primary}`);
  console.log(`Secondary emotions: ${result.emotions.secondary.join(', ')}`);
  console.log(`Confidence: ${result.metadata.confidence}`);
})
.catch(error => {
  console.error(`Error: ${error.message}`);
});

// Generate summary
samo.summarize({
  text: "Long text to summarize...",
  options: {
    max_length: 100,
    style: "concise"
  }
})
.then(result => {
  console.log(`Summary: ${result.summary}`);
})
.catch(error => {
  console.error(`Error: ${error.message}`);
});
```

## 🔌 Integration Patterns

### Web Application Integration

```javascript
// React component example
function EmotionAnalyzer() {
  const [text, setText] = useState('');
  const [emotions, setEmotions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeEmotions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('https://api.samo.ai/v1/emotions/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': process.env.REACT_APP_SAMO_API_KEY
        },
        body: JSON.stringify({ text })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error.message);
      }
      
      const data = await response.json();
      setEmotions(data.emotions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <textarea 
        value={text} 
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to analyze emotions..."
      />
      <button onClick={analyzeEmotions} disabled={loading || !text}>
        {loading ? 'Analyzing...' : 'Analyze Emotions'}
      </button>
      
      {error && <div className="error">{error}</div>}
      
      {emotions && (
        <div className="results">
          <h3>Primary Emotion: {emotions.primary}</h3>
          <h4>Secondary Emotions:</h4>
          <ul>
            {emotions.secondary.map(emotion => (
              <li key={emotion}>{emotion}</li>
            ))}
          </ul>
          <div className="emotion-chart">
            {/* Visualization of emotion scores */}
          </div>
        </div>
      )}
    </div>
  );
}
```

### Mobile App Integration

```swift
// Swift example for iOS
import SamoSDK

class EmotionAnalysisViewController: UIViewController {
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var analyzeButton: UIButton!
    @IBOutlet weak var resultsView: UIView!
    
    private let samoClient = SamoClient(apiKey: "your_api_key_here")
    
    @IBAction func analyzeButtonTapped(_ sender: UIButton) {
        guard let text = textView.text, !text.isEmpty else { return }
        
        analyzeButton.isEnabled = false
        analyzeButton.setTitle("Analyzing...", for: .normal)
        
        samoClient.emotions.analyze(text: text) { [weak self] result in
            DispatchQueue.main.async {
                self?.analyzeButton.isEnabled = true
                self?.analyzeButton.setTitle("Analyze", for: .normal)
                
                switch result {
                case .success(let emotions):
                    self?.displayResults(emotions)
                case .failure(let error):
                    self?.showError(error.localizedDescription)
                }
            }
        }
    }
    
    private func displayResults(_ emotions: EmotionAnalysis) {
        // Display emotion results in UI
    }
    
    private func showError(_ message: String) {
        let alert = UIAlertController(
            title: "Analysis Error",
            message: message,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}
```

### Batch Processing

For large-scale processing, use the batch API endpoints:

```bash
curl -X POST "https://api.samo.ai/v1/batch/emotions/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "items": [
      {"id": "entry1", "text": "I feel happy today"},
      {"id": "entry2", "text": "This makes me angry"},
      {"id": "entry3", "text": "I'm not sure how I feel"}
    ],
    "options": {
      "threshold": 0.5
    }
  }'
```

Response:

```json
{
  "batch_id": "batch_12345",
  "status": "processing",
  "total_items": 3,
  "estimated_completion_time": "2025-07-24T08:25:00Z",
  "webhook_url": "https://your-server.com/webhook-endpoint"
}
```

## 📈 Performance Considerations

### Response Times

| Endpoint | Typical Response Time | p95 Response Time |
|----------|------------------------|-------------------|
| `/v1/emotions/analyze` | 200-300ms | <500ms |
| `/v1/summarize` | 500-800ms | <1.5s |
| `/v1/voice/transcribe` | 1-3s | <5s per minute of audio |

### Batch Processing

For processing multiple items, use batch endpoints to reduce overhead:
- Up to 100 items per batch request
- Asynchronous processing with webhook notifications
- Significant performance improvement over individual requests

### Caching Strategy

- Responses include `ETag` headers for client-side caching
- Identical requests within 24 hours may return cached results
- Include `Cache-Control: no-cache` header to force fresh analysis

## 🔒 Security Best Practices

### API Key Management

- Store API keys securely in environment variables or secrets management systems
- Never expose API keys in client-side code
- Rotate API keys periodically (recommended: every 90 days)
- Use different API keys for development and production environments

### Data Privacy

- All data is encrypted in transit (TLS 1.3)
- Data retention policies:
  - Request logs: 30 days
  - Analysis results: 7 days
  - Raw text/audio: Not stored after processing
- GDPR and CCPA compliance built-in

## 🚀 Getting Started

### 1. Sign Up for API Access

Visit [https://samo.ai/signup](https://samo.ai/signup) to create an account and get your API key.

### 2. Install SDK (Optional)

```bash
# Python
pip install samo-ai

# JavaScript
npm install samo-ai
```

### 3. Make Your First API Call

```bash
curl -X POST "https://api.samo.ai/v1/emotions/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{"text": "I am really excited to try this API!"}'
```

### 4. Explore Documentation

- [API Reference](https://samo.ai/docs/api)
- [SDK Documentation](https://samo.ai/docs/sdk)
- [Integration Examples](https://samo.ai/docs/examples)
- [FAQ](https://samo.ai/docs/faq)

## 📞 Support

- Email: api-support@samo.ai
- Developer Community: [https://community.samo.ai](https://community.samo.ai)
- Status Page: [https://status.samo.ai](https://status.samo.ai) 