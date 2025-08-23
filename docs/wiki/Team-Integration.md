# 🤝 Team Integration Playbook

Welcome to the SAMO Brain Integration Playbook! This guide is the single source of truth for Backend, Frontend, Data Science, and UX teams to integrate with SAMO's AI capabilities.

**Current Status**: ✅ **LIVE & PRODUCTION-READY**

---

## 🚀 Quick Start (All Teams)

The core emotion detection API is live and ready for integration.

**Live API Endpoint**: `https://samo-emotion-api-minimal-71517823771.us-central1.run.app`

### Test Your Connection (2 minutes)

You can test the API with a simple `curl` command. No authentication is needed for the basic prediction endpoint.

```bash
# Health Check
curl https://samo-emotion-api-minimal-71517823771.us-central1.run.app/health

# Emotion Prediction
curl -X POST https://samo-emotion-api-minimal-71517823771.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling incredibly excited about this integration!"}'
```

**Expected Response:**
```json
{
  "primary_emotion": {
    "emotion": "excited",
    "confidence": 0.95
  },
  "all_emotions": [
    {"emotion": "excited", "confidence": 0.95},
    {"emotion": "happy", "confidence": 0.88}
  ],
  "inference_time": 0.08,
  "model_type": "roberta_single_label",
  "text_length": 58
}
```

---

## 🔧 Backend Integration Guide

**Objective**: Integrate emotion analysis into your backend services.

### Key Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check with basic metrics. |
| `/predict` | `POST` | Analyze a single text for emotions. |
| `/predict_batch` | `POST` | Analyze multiple texts in one call for efficiency. |
| `/metrics` | `GET` | Get detailed server and model metrics. |

### Python (FastAPI/Flask) Example

```python
import requests
from typing import Dict, List, Optional

class SAMO_API_Client:
    def __init__(self, base_url: str = "https://samo-emotion-api-minimal-71517823771.us-central1.run.app"):
        self.base_url = base_url
        self.session = requests.Session()

    def analyze_emotion(self, text: str) -> Optional[Dict]:
        """Analyzes emotion for a single text, with error handling."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=10 # 10-second timeout
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def analyze_emotions_batch(self, texts: List[str]) -> Optional[List[Dict]]:
        """Analyzes emotions for multiple texts efficiently."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict_batch",
                json={"texts": texts},
                headers={"Content-Type": "application/json"},
                timeout=30 # Longer timeout for batches
            )
            response.raise_for_status()
            return response.json().get("predictions")
        except requests.exceptions.RequestException as e:
            print(f"API Batch Error: {e}")
            return None

# --- Usage Example ---
client = SAMO_API_Client()

# Single prediction
result = client.analyze_emotion("I am feeling excited about this project!")
if result:
    print(f"Primary Emotion: {result['primary_emotion']['emotion']}")

# Batch prediction
texts_to_analyze = ["I am happy", "I feel sad", "I am excited"]
batch_results = client.analyze_emotions_batch(texts_to_analyze)
if batch_results:
    for res in batch_results:
        print(f"'{res['text']}' -> {res['predicted_emotion']}")
```

### Best Practices
1.  **Use Batching**: For analyzing multiple user comments or entries, the `/predict_batch` endpoint is significantly more performant.
2.  **Implement Timeouts & Retries**: Use a timeout for requests and implement exponential backoff for retries on `5xx` server errors.
3.  **Connection Pooling**: For high-throughput services, use `requests.Session` to reuse TCP connections.

---

## 🎨 Frontend & UX/UI Integration Guide

**Objective**: Create beautiful, responsive, and emotionally-aware user interfaces.

### Emotion Color Palette & Icons

Use this palette to create a consistent emotional language in the UI.

```javascript
const EMOTION_STYLES = {
  happy:      { color: '#FFD54F', icon: '😄' },
  sad:        { color: '#7986CB', icon: '😢' },
  excited:    { color: '#FFA726', icon: '🤩' },
  anxious:    { color: '#FF6B6B', icon: '😰' },
  frustrated: { color: '#FF7043', icon: '😤' },
  calm:       { color: '#4ECDC4', icon: '😌' },
  grateful:   { color: '#66BB6A', icon: '🙏' },
  hopeful:    { color: '#81C784', icon: '🤗' },
  overwhelmed:{ color: '#9575CD', icon: '😵' },
  proud:      { color: '#4DB6AC', icon: '😎' },
  content:    { color: '#45B7D1', icon: '😊' },
  tired:      { color: '#A1887F', icon: '😴' }
};
```

### React Component Example

Here is a simple, reusable React component to analyze and display emotions.

```jsx
import React, { useState, useCallback } from 'react';
import { debounce } from 'lodash';

const EmotionAnalyzer = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeAPI = async (inputText) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('https://samo-emotion-api-minimal-71517823771.us-central1.run.app/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const debouncedAnalyze = useCallback(debounce(analyzeAPI, 500), []);

  const handleChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    if (newText.trim().length > 10) {
      debouncedAnalyze(newText);
    }
  };

  return (
    <div>
      <textarea
        value={text}
        onChange={handleChange}
        placeholder="How are you feeling today?"
        rows={4}
      />
      {loading && <p>Analyzing...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {result && (
        <div>
          <h3>Primary Emotion: {result.primary_emotion.emotion}</h3>
          <p>Confidence: {(result.primary_emotion.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
};

export default EmotionAnalyzer;
```

### UX Best Practices
1.  **Subtle Feedback**: Use emotion data to create subtle UI changes (e.g., changing a border color, showing a gentle animation). Avoid drastic, jarring changes.
2.  **Debounce Input**: Don't call the API on every keystroke. Analyze text after the user pauses to create a smoother experience and reduce API calls.
3.  **Provide Control**: Always give users control over emotion-aware features and be transparent about what you're analyzing.

---

## 📊 Data Science Integration Guide

**Objective**: Leverage emotion data for analytics, model monitoring, and research.

### Accessing Performance Metrics

The `/metrics` endpoint provides real-time server and model performance data.

```python
import requests
import pandas as pd

def get_api_metrics():
    """Fetches and formats the latest API metrics."""
    try:
        response = requests.get("https://samo-emotion-api-minimal-71517823771.us-central1.run.app/metrics")
        response.raise_for_status()
        metrics = response.json()

        # Flatten metrics for easy analysis in Pandas
        server_metrics = metrics.get('server_metrics', {})
        emotion_dist = metrics.get('emotion_distribution', {})

        flat_data = {**server_metrics, **emotion_dist}
        return pd.DataFrame([flat_data])

    except requests.exceptions.RequestException as e:
        print(f"Could not fetch metrics: {e}")
        return None

# --- Usage Example ---
metrics_df = get_api_metrics()
if metrics_df is not None:
    print("Latest API Metrics:")
    print(metrics_df[['success_rate', 'average_response_time_ms', 'happy', 'sad', 'excited']].T)
```

### Analyzing Emotion Trends

You can build powerful analytics by collecting prediction results over time.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_emotion_trends(collected_data: pd.DataFrame):
    """
    Analyzes and visualizes emotion trends from collected prediction data.
    Assumes `collected_data` is a DataFrame with 'timestamp' and 'primary_emotion' columns.
    """
    df = collected_data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Analyze distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df['primary_emotion'], order=df['primary_emotion'].value_counts().index, palette='viridis')
    plt.title('Overall Emotion Distribution')
    plt.xlabel('Count')
    plt.show()

    # Analyze trends over time
    daily_trends = df.resample('D')['primary_emotion'].value_counts().unstack().fillna(0)
    daily_trends.plot(kind='line', figsize=(14, 7), title='Daily Emotion Trends')
    plt.ylabel('Number of Mentions')
    plt.show()

# --- Usage Example ---
# Assuming `results_df` is a DataFrame of collected results
# analyze_emotion_trends(results_df)
```

### Detecting Model Drift

A key responsibility is to monitor for model drift (when the model's performance degrades on new, real-world data).

```python
class ModelDriftDetector:
    def __init__(self, baseline_metrics: Dict):
        # Baseline metrics from a "golden" dataset
        self.baseline = baseline_metrics
        self.drift_threshold = 0.10 # 10% change

    def check_for_drift(self, current_metrics: Dict) -> Dict:
        """Compares current metrics to the baseline to detect drift."""
        drift_report = {}

        # 1. Confidence Drift
        conf_change = (current_metrics['avg_confidence'] - self.baseline['avg_confidence']) / self.baseline['avg_confidence']
        if abs(conf_change) > self.drift_threshold:
            drift_report['confidence_drift'] = {
                "change": f"{conf_change:.2%}",
                "alert": "Confidence has drifted significantly."
            }

        # 2. Distribution Drift (using a simple metric for example)
        # A more robust method like KL divergence is recommended in production
        baseline_dist = self.baseline['emotion_distribution']
        current_dist = current_metrics['emotion_distribution']

        # Check if a top emotion has changed
        if baseline_dist.index[0] != current_dist.index[0]:
             drift_report['distribution_drift'] = {
                "change": f"Top emotion changed from {baseline_dist.index[0]} to {current_dist.index[0]}",
                "alert": "Emotion distribution has shifted."
            }

        return drift_report

# --- Usage Example ---
# baseline_stats = {'avg_confidence': 0.85, 'emotion_distribution': ...}
# current_stats = {'avg_confidence': 0.75, 'emotion_distribution': ...}

# detector = ModelDriftDetector(baseline_stats)
# drift = detector.check_for_drift(current_stats)
# if drift:
#     print("Drift Detected!", drift)
```

### Key Responsibilities
1.  **Monitor Performance**: Regularly query the `/metrics` endpoint and track key performance indicators.
2.  **Analyze Trends**: Collect prediction data to analyze long-term emotional trends and user behavior.
3.  **Detect Drift**: Implement a robust drift detection system to know when the model needs retraining.
4.  **Provide Feedback**: Your analysis is crucial for guiding the next iteration of model development.
