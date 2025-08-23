# 🔧 Backend Integration Guide

Welcome, Backend Developers! This guide will help you integrate SAMO Brain's AI capabilities into your backend systems quickly and efficiently.

## 🚀 **Quick Start (5 minutes)**

### **1. Test the API**
```bash
# Health check
curl http://localhost:8000/health

# Test emotion detection
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

### **2. Expected Response**
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

---

## 📋 **API Endpoints Reference**

### **Base URL**
```
Local Development: http://localhost:8000
Production: https://api.samo.ai/v1
```

### **Available Endpoints**

| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/health` | GET | Health check with metrics | <10ms |
| `/metrics` | GET | Detailed server metrics | <10ms |
| `/predict` | POST | Single emotion prediction | <100ms |
| `/predict_batch` | POST | Batch emotion predictions | <100ms per text |
| `/` | GET | API documentation | <10ms |

---

## 🔌 **Integration Examples**

### **Python (FastAPI/Flask)**

```python
import requests
import json
from typing import Dict, List, Optional

class SAMOBrainClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check API health and get basic metrics."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotion for a single text."""
        payload = {"text": text}
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def analyze_emotions_batch(self, texts: List[str]) -> Dict:
        """Analyze emotions for multiple texts efficiently."""
        payload = {"texts": texts}
        response = self.session.post(
            f"{self.base_url}/predict_batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def get_metrics(self) -> Dict:
        """Get detailed server metrics."""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()

# Usage example
client = SAMOBrainClient()

# Health check
health = client.health_check()
print(f"API Status: {health['status']}")

# Single prediction
result = client.analyze_emotion("I am feeling excited about this project!")
print(f"Emotion: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
texts = ["I am happy", "I feel sad", "I am excited"]
batch_results = client.analyze_emotions_batch(texts)
for pred in batch_results['predictions']:
    print(f"{pred['text']} → {pred['predicted_emotion']}")
```

### **Node.js (Express)**

```javascript
const axios = require('axios');

class SAMOBrainClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }

    async analyzeEmotion(text) {
        try {
            const response = await this.client.post('/predict', { text });
            return response.data;
        } catch (error) {
            throw new Error(`Emotion analysis failed: ${error.message}`);
        }
    }

    async analyzeEmotionsBatch(texts) {
        try {
            const response = await this.client.post('/predict_batch', { texts });
            return response.data;
        } catch (error) {
            throw new Error(`Batch analysis failed: ${error.message}`);
        }
    }

    async getMetrics() {
        try {
            const response = await this.client.get('/metrics');
            return response.data;
        } catch (error) {
            throw new Error(`Metrics retrieval failed: ${error.message}`);
        }
    }
}

// Usage example
async function main() {
    const client = new SAMOBrainClient();

    try {
        // Health check
        const health = await client.healthCheck();
        console.log(`API Status: ${health.status}`);

        // Single prediction
        const result = await client.analyzeEmotion("I am feeling excited about this project!");
        console.log(`Emotion: ${result.predicted_emotion}`);
        console.log(`Confidence: ${result.confidence.toFixed(3)}`);

        // Batch prediction
        const texts = ["I am happy", "I feel sad", "I am excited"];
        const batchResults = await client.analyzeEmotionsBatch(texts);
        batchResults.predictions.forEach(pred => {
            console.log(`${pred.text} → ${pred.predicted_emotion}`);
        });

    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

### **Java (Spring Boot)**

```java
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.Map;

public class SAMOBrainClient {
    private final String baseUrl;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    public SAMOBrainClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    public Map<String, Object> healthCheck() {
        String url = baseUrl + "/health";
        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
        return response.getBody();
    }

    public Map<String, Object> analyzeEmotion(String text) {
        String url = baseUrl + "/predict";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, String> payload = Map.of("text", text);
        HttpEntity<Map<String, String>> request = new HttpEntity<>(payload, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
        return response.getBody();
    }

    public Map<String, Object> analyzeEmotionsBatch(List<String> texts) {
        String url = baseUrl + "/predict_batch";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        Map<String, List<String>> payload = Map.of("texts", texts);
        HttpEntity<Map<String, List<String>>> request = new HttpEntity<>(payload, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(url, request, Map.class);
        return response.getBody();
    }

    public Map<String, Object> getMetrics() {
        String url = baseUrl + "/metrics";
        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
        return response.getBody();
    }
}

// Usage example
public class Main {
    public static void main(String[] args) {
        SAMOBrainClient client = new SAMOBrainClient("http://localhost:8000");

        try {
            // Health check
            Map<String, Object> health = client.healthCheck();
            System.out.println("API Status: " + health.get("status"));

            // Single prediction
            Map<String, Object> result = client.analyzeEmotion("I am feeling excited about this project!");
            System.out.println("Emotion: " + result.get("predicted_emotion"));
            System.out.println("Confidence: " + result.get("confidence"));

            // Batch prediction
            List<String> texts = List.of("I am happy", "I feel sad", "I am excited");
            Map<String, Object> batchResults = client.analyzeEmotionsBatch(texts);
            List<Map<String, Object>> predictions = (List<Map<String, Object>>) batchResults.get("predictions");

            for (Map<String, Object> pred : predictions) {
                System.out.println(pred.get("text") + " → " + pred.get("predicted_emotion"));
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
```

---

## 🔒 **Authentication & Security**

### **Rate Limiting**
- **Limit**: 100 requests per minute per IP address
- **Window**: 60 seconds (sliding window)
- **Response**: HTTP 429 when exceeded

### **Error Handling**
```python
import requests
from requests.exceptions import RequestException

def safe_analyze_emotion(text: str, client: SAMOBrainClient) -> Dict:
    """Safe emotion analysis with comprehensive error handling."""
    try:
        return client.analyze_emotion(text)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limit exceeded", "retry_after": 60}
        elif e.response.status_code == 400:
            return {"error": "Invalid request", "details": e.response.json()}
        elif e.response.status_code == 500:
            return {"error": "Server error", "retry": True}
        else:
            return {"error": f"HTTP error: {e.response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout", "retry": True}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed", "retry": True}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Usage with retry logic
def analyze_with_retry(text: str, max_retries: int = 3) -> Dict:
    client = SAMOBrainClient()

    for attempt in range(max_retries):
        result = safe_analyze_emotion(text, client)

        if "error" not in result:
            return result

        if result.get("retry") and attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
            continue

        return result
```

---

## 📊 **Performance Optimization**

### **Batch Processing**
```python
# ❌ Inefficient - Multiple API calls
for text in texts:
    result = client.analyze_emotion(text)

# ✅ Efficient - Single batch call
results = client.analyze_emotions_batch(texts)
```

### **Caching Strategy**
```python
from functools import lru_cache
import hashlib

class CachedSAMOBrainClient(SAMOBrainClient):
    def __init__(self, base_url: str = "http://localhost:8000", cache_size: int = 1000):
        super().__init__(base_url)
        self.cache_size = cache_size

    @lru_cache(maxsize=1000)
    def analyze_emotion_cached(self, text_hash: str) -> Dict:
        """Cache emotion analysis results."""
        # In production, use Redis or similar for distributed caching
        return super().analyze_emotion(text_hash)

    def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotion with caching."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.analyze_emotion_cached(text_hash)
```

### **Connection Pooling**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedSAMOBrainClient(SAMOBrainClient):
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(base_url)

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(
                total=3,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504]
            )
        )

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
```

---

## 🔍 **Monitoring & Observability**

### **Health Monitoring**
```python
import time
from typing import Dict, List

class SAMOBrainMonitor:
    def __init__(self, client: SAMOBrainClient):
        self.client = client
        self.metrics = []

    def check_health(self) -> Dict:
        """Comprehensive health check."""
        start_time = time.time()

        try:
            health = self.client.health_check()
            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy" if health["status"] == "healthy" else "unhealthy",
                "response_time_ms": response_time,
                "uptime_seconds": health.get("uptime_seconds", 0),
                "success_rate": health.get("metrics", {}).get("success_rate", "0%"),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000,
                "timestamp": time.time()
            }

    def monitor_performance(self, duration_minutes: int = 5) -> Dict:
        """Monitor performance over time."""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        while time.time() < end_time:
            health = self.check_health()
            self.metrics.append(health)
            time.sleep(30)  # Check every 30 seconds

        return self.analyze_metrics()

    def analyze_metrics(self) -> Dict:
        """Analyze collected metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}

        response_times = [m["response_time_ms"] for m in self.metrics if "response_time_ms" in m]
        success_count = sum(1 for m in self.metrics if m["status"] == "healthy")

        return {
            "total_checks": len(self.metrics),
            "success_rate": success_count / len(self.metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0
        }
```

---

## 🧪 **Testing Integration**

### **Unit Tests**
```python
import unittest
from unittest.mock import Mock, patch

class TestSAMOBrainIntegration(unittest.TestCase):
    def setUp(self):
        self.client = SAMOBrainClient("http://localhost:8000")

    def test_health_check(self):
        """Test health check endpoint."""
        health = self.client.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue("model_status" in health)

    def test_emotion_analysis(self):
        """Test emotion analysis endpoint."""
        result = self.client.analyze_emotion("I am feeling happy today!")
        self.assertEqual(result["predicted_emotion"], "happy")
        self.assertGreater(result["confidence"], 0.5)
        self.assertTrue("probabilities" in result)

    def test_batch_analysis(self):
        """Test batch analysis endpoint."""
        texts = ["I am happy", "I feel sad", "I am excited"]
        results = self.client.analyze_emotions_batch(texts)
        self.assertEqual(len(results["predictions"]), 3)
        self.assertEqual(results["count"], 3)

    @patch('requests.Session.post')
    def test_error_handling(self, mock_post):
        """Test error handling."""
        mock_post.side_effect = requests.exceptions.ConnectionError()

        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.analyze_emotion("test")

if __name__ == '__main__':
    unittest.main()
```

### **Integration Tests**
```python
import pytest
import requests

@pytest.fixture
def client():
    return SAMOBrainClient("http://localhost:8000")

def test_full_integration(client):
    """Test complete integration workflow."""
    # 1. Health check
    health = client.health_check()
    assert health["status"] == "healthy"

    # 2. Single prediction
    result = client.analyze_emotion("I am feeling grateful for this opportunity!")
    assert result["predicted_emotion"] in ["grateful", "happy", "content"]
    assert result["confidence"] > 0.3

    # 3. Batch prediction
    texts = [
        "I am feeling anxious about the presentation",
        "I am excited to start this new project",
        "I feel overwhelmed with all the work"
    ]
    batch_results = client.analyze_emotions_batch(texts)
    assert len(batch_results["predictions"]) == 3

    # 4. Metrics check
    metrics = client.get_metrics()
    assert "server_metrics" in metrics
    assert metrics["server_metrics"]["success_rate"] > "90%"
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Connection Refused**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start server if needed
cd SAMO--DL/local_deployment
python api_server.py
```

#### **2. Rate Limiting**
```python
# Implement exponential backoff
import time
import random

def analyze_with_backoff(text: str, max_retries: int = 3) -> Dict:
    client = SAMOBrainClient()

    for attempt in range(max_retries):
        try:
            return client.analyze_emotion(text)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            raise
```

#### **3. High Response Times**
```python
# Check server metrics
metrics = client.get_metrics()
print(f"Average response time: {metrics['server_metrics']['average_response_time_ms']}ms")

# Use batch processing for multiple requests
results = client.analyze_emotions_batch(texts)  # More efficient
```

---

## 📞 **Support & Resources**

- **API Documentation**: [Complete API Reference](API-Reference)
- **Error Codes**: [Error Handling Guide](Error-Handling-Guide)
- **Performance**: [Performance Optimization Guide](Performance-Guide)
- **GitHub Issues**: [Report Issues](https://github.com/your-org/SAMO--DL/issues)

---

**Ready to integrate?** Start with the [Quick Start](#-quick-start-5-minutes) section above, and you'll be up and running in minutes! 🚀
