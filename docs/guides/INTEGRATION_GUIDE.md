# SAMO-DL Integration Guide

## Overview

This guide provides comprehensive integration instructions for backend, frontend, UX, and data science teams to integrate with the SAMO-DL emotion detection API.

## Quick Start

### API Base URL
```
https://samo-emotion-api-xxxxx-ew.a.run.app
```

### Basic Usage
```bash
curl -X POST https://samo-emotion-api-xxxxx-ew.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling really happy today!"}'
```

**Response:**
```json
[
  {
    "emotion": "joy",
    "confidence": 0.89
  },
  {
    "emotion": "excitement",
    "confidence": 0.76
  }
]
```

---

## Backend Integration

### Python (Flask/Django/FastAPI)

#### 1. Basic Integration
```python
import requests

def detect_emotion(text: str) -> dict:
    """Integrate with SAMO Emotion API"""
    try:
        response = requests.post(
            "https://samo-emotion-api-xxxxx-ew.a.run.app/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return {"error": "Failed to analyze emotions"}

# Example usage
emotions = detect_emotion("I'm feeling excited about this project!")
print(emotions)  # [{"emotion": "excitement", "confidence": 0.92}]
```

#### 2. Flask Integration
```python
from flask import Flask, request, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_user_feedback():
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Call SAMO-DL API
    emotions = detect_emotion(user_text)

    return jsonify({
        "user_text": user_text,
        "emotions": emotions,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

#### 3. Django Integration
```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests

@csrf_exempt
def analyze_emotion(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')

        if not text:
            return JsonResponse({"error": "No text provided"}, status=400)

        # Call SAMO-DL API
        emotions = detect_emotion(text)

        return JsonResponse({
            "text": text,
            "emotions": emotions
        })

    return JsonResponse({"error": "Method not allowed"}, status=405)
```

### Node.js (Express)

#### 1. Basic Integration
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
console.log(emotions); // [{emotion: "excitement", confidence: 0.92}]
```

#### 2. Express.js Middleware
```javascript
const express = require('express');
const app = express();

app.use(express.json());

// Middleware for emotion analysis
const emotionAnalysis = async (req, res, next) => {
    const text = req.body.text;

    if (!text) {
        return res.status(400).json({ error: 'No text provided' });
    }

    try {
        const emotions = await detectEmotion(text);
        req.emotions = emotions;
        next();
    } catch (error) {
        res.status(500).json({ error: 'Emotion analysis failed' });
    }
};

// Route using emotion analysis
app.post('/feedback', emotionAnalysis, (req, res) => {
    res.json({
        message: 'Feedback received',
        emotions: req.emotions,
        timestamp: new Date().toISOString()
    });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

---

## Frontend Integration

### React

#### 1. Custom Hook
```javascript
import { useState } from 'react';

const useEmotionDetection = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [emotions, setEmotions] = useState([]);

    const analyzeEmotion = async (text) => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                'https://samo-emotion-api-xxxxx-ew.a.run.app/predict',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setEmotions(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return { analyzeEmotion, emotions, loading, error };
};
```

#### 2. React Component
```jsx
import React, { useState } from 'react';
import { useEmotionDetection } from './useEmotionDetection';

const EmotionAnalyzer = () => {
    const [text, setText] = useState('');
    const { analyzeEmotion, emotions, loading, error } = useEmotionDetection();

    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim()) {
            analyzeEmotion(text);
        }
    };

    return (
        <div className="emotion-analyzer">
            <form onSubmit={handleSubmit}>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text to analyze emotions..."
                    className="form-control mb-3"
                    rows="4"
                />
                <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={loading || !text.trim()}
                >
                    {loading ? 'Analyzing...' : 'Analyze Emotions'}
                </button>
            </form>

            {error && (
                <div className="alert alert-danger mt-3">
                    Error: {error}
                </div>
            )}

            {emotions.length > 0 && (
                <div className="mt-3">
                    <h6>Detected Emotions:</h6>
                    {emotions.map((emotion, index) => (
                        <div key={index} className="card mb-2">
                            <div className="card-body">
                                <h6 className="card-title">{emotion.emotion}</h6>
                                <div className="progress">
                                    <div
                                        className="progress-bar"
                                        style={{width: `${emotion.confidence * 100}%`}}
                                    >
                                        {Math.round(emotion.confidence * 100)}%
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default EmotionAnalyzer;
```

### Vue.js

#### 1. Vue Composable
```javascript
// composables/useEmotionDetection.js
import { ref } from 'vue';

export function useEmotionDetection() {
    const loading = ref(false);
    const error = ref(null);
    const emotions = ref([]);

    const analyzeEmotion = async (text) => {
        loading.value = true;
        error.value = null;

        try {
            const response = await fetch(
                'https://samo-emotion-api-xxxxx-ew.a.run.app/predict',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            emotions.value = data;
        } catch (err) {
            error.value = err.message;
        } finally {
            loading.value = false;
        }
    };

    return { analyzeEmotion, emotions, loading, error };
}
```

#### 2. Vue Component
```vue
<template>
    <div class="emotion-analyzer">
        <form @submit.prevent="handleSubmit">
            <textarea
                v-model="text"
                placeholder="Enter text to analyze emotions..."
                class="form-control mb-3"
                rows="4"
            ></textarea>
            <button
                type="submit"
                class="btn btn-primary"
                :disabled="loading || !text.trim()"
            >
                {{ loading ? 'Analyzing...' : 'Analyze Emotions' }}
            </button>
        </form>

        <div v-if="error" class="alert alert-danger mt-3">
            Error: {{ error }}
        </div>

        <div v-if="emotions.length > 0" class="mt-3">
            <h6>Detected Emotions:</h6>
            <div
                v-for="(emotion, index) in emotions"
                :key="index"
                class="card mb-2"
            >
                <div class="card-body">
                    <h6 class="card-title">{{ emotion.emotion }}</h6>
                    <div class="progress">
                        <div
                            class="progress-bar"
                            :style="{width: `${emotion.confidence * 100}%`}"
                        >
                            {{ Math.round(emotion.confidence * 100) }}%
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import { ref } from 'vue';
import { useEmotionDetection } from '@/composables/useEmotionDetection';

export default {
    name: 'EmotionAnalyzer',
    setup() {
        const text = ref('');
        const { analyzeEmotion, emotions, loading, error } = useEmotionDetection();

        const handleSubmit = () => {
            if (text.value.trim()) {
                analyzeEmotion(text.value);
            }
        };

        return {
            text,
            analyzeEmotion,
            emotions,
            loading,
            error,
            handleSubmit
        };
    }
};
</script>
```

### Angular

#### 1. Angular Service
```typescript
// services/emotion.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Emotion {
    emotion: string;
    confidence: number;
}

@Injectable({
    providedIn: 'root'
})
export class EmotionService {
    private apiUrl = 'https://samo-emotion-api-xxxxx-ew.a.run.app';

    constructor(private http: HttpClient) {}

    analyzeEmotion(text: string): Observable<Emotion[]> {
        return this.http.post<Emotion[]>(`${this.apiUrl}/predict`, { text });
    }
}
```

#### 2. Angular Component
```typescript
// components/emotion-analyzer.component.ts
import { Component } from '@angular/core';
import { EmotionService, Emotion } from '../services/emotion.service';

@Component({
    selector: 'app-emotion-analyzer',
    template: `
        <div class="emotion-analyzer">
            <form (ngSubmit)="handleSubmit()">
                <textarea
                    [(ngModel)]="text"
                    name="text"
                    placeholder="Enter text to analyze emotions..."
                    class="form-control mb-3"
                    rows="4"
                ></textarea>
                <button
                    type="submit"
                    class="btn btn-primary"
                    [disabled]="loading || !text.trim()"
                >
                    {{ loading ? 'Analyzing...' : 'Analyze Emotions' }}
                </button>
            </form>

            <div *ngIf="error" class="alert alert-danger mt-3">
                Error: {{ error }}
            </div>

            <div *ngIf="emotions.length > 0" class="mt-3">
                <h6>Detected Emotions:</h6>
                <div *ngFor="let emotion of emotions; let i = index" class="card mb-2">
                    <div class="card-body">
                        <h6 class="card-title">{{ emotion.emotion }}</h6>
                        <div class="progress">
                            <div
                                class="progress-bar"
                                [style.width.%]="emotion.confidence * 100"
                            >
                                {{ (emotion.confidence * 100) | number:'1.0-0' }}%
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `
})
export class EmotionAnalyzerComponent {
    text = '';
    emotions: Emotion[] = [];
    loading = false;
    error: string | null = null;

    constructor(private emotionService: EmotionService) {}

    handleSubmit() {
        if (!this.text.trim()) return;

        this.loading = true;
        this.error = null;

        this.emotionService.analyzeEmotion(this.text).subscribe({
            next: (emotions) => {
                this.emotions = emotions;
                this.loading = false;
            },
            error: (err) => {
                this.error = err.message;
                this.loading = false;
            }
        });
    }
}
```

---

## UX Integration

### Emotion-Based UI Adaptation

```javascript
// Emotion-aware UI adaptation
class EmotionAwareUI {
    constructor() {
        this.currentEmotion = null;
        this.emotionHistory = [];
    }

    async analyzeUserInput(text) {
        try {
            const response = await fetch(
                'https://samo-emotion-api-xxxxx-ew.a.run.app/predict',
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                }
            );

            const emotions = await response.json();
            this.currentEmotion = emotions[0]?.emotion;
            this.emotionHistory.push({
                emotion: this.currentEmotion,
                timestamp: new Date(),
                text: text
            });

            this.adaptUI();
        } catch (error) {
            console.error('Emotion analysis failed:', error);
        }
    }

    adaptUI() {
        const body = document.body;

        // Remove existing emotion classes
        body.classList.remove('emotion-joy', 'emotion-sadness', 'emotion-anger', 'emotion-fear');

        // Add emotion-specific styling
        if (this.currentEmotion) {
            body.classList.add(`emotion-${this.currentEmotion}`);
        }

        // Update UI elements based on emotion
        this.updateColorScheme();
        this.updateContent();
        this.updateInteractions();
    }

    updateColorScheme() {
        const colorSchemes = {
            joy: { primary: '#10b981', secondary: '#34d399' },
            sadness: { primary: '#3b82f6', secondary: '#60a5fa' },
            anger: { primary: '#ef4444', secondary: '#f87171' },
            fear: { primary: '#f59e0b', secondary: '#fbbf24' }
        };

        const scheme = colorSchemes[this.currentEmotion] || colorSchemes.joy;
        document.documentElement.style.setProperty('--primary-color', scheme.primary);
        document.documentElement.style.setProperty('--secondary-color', scheme.secondary);
    }

    updateContent() {
        const contentAdaptations = {
            joy: {
                greeting: "Great to see you're happy! 😊",
                suggestions: ["Share your joy", "Celebrate this moment"]
            },
            sadness: {
                greeting: "I understand you're feeling down. 💙",
                suggestions: ["Take a break", "Talk to someone"]
            },
            anger: {
                greeting: "I sense you're frustrated. 🔥",
                suggestions: ["Take deep breaths", "Step back for a moment"]
            },
            fear: {
                greeting: "It's okay to feel anxious. 🤗",
                suggestions: ["Breathe slowly", "You're safe here"]
            }
        };

        const adaptation = contentAdaptations[this.currentEmotion] || contentAdaptations.joy;

        // Update UI elements
        const greetingElement = document.getElementById('greeting');
        if (greetingElement) {
            greetingElement.textContent = adaptation.greeting;
        }
    }

    updateInteractions() {
        // Adjust interaction patterns based on emotion
        const interactionPatterns = {
            joy: { animationSpeed: 'fast', soundEnabled: true },
            sadness: { animationSpeed: 'slow', soundEnabled: false },
            anger: { animationSpeed: 'fast', soundEnabled: false },
            fear: { animationSpeed: 'slow', soundEnabled: true }
        };

        const pattern = interactionPatterns[this.currentEmotion] || interactionPatterns.joy;

        // Apply interaction changes
        document.body.style.setProperty('--animation-speed', pattern.animationSpeed);
        if (window.soundManager) {
            window.soundManager.enabled = pattern.soundEnabled;
        }
    }
}

// Usage
const emotionUI = new EmotionAwareUI();

// Analyze user input and adapt UI
document.getElementById('user-input').addEventListener('input', (e) => {
    if (e.target.value.length > 10) {
        emotionUI.analyzeUserInput(e.target.value);
    }
});
```

---

## Data Science Integration

### Data Collection and Analysis

```python
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import logging

class EmotionDataCollector:
    def __init__(self, api_url="https://samo-emotion-api-xxxxx-ew.a.run.app"):
        self.api_url = api_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def analyze_batch(self, texts: list) -> pd.DataFrame:
        """Analyze a batch of texts and return structured results"""
        results = []

        for i, text in enumerate(texts):
            try:
                response = self.session.post(
                    f"{self.api_url}/predict",
                    json={"text": text},
                    timeout=30
                )
                response.raise_for_status()

                emotions = response.json()
                results.append({
                    'text': text,
                    'emotions': emotions,
                    'timestamp': datetime.now(),
                    'status': 'success'
                })

            except Exception as e:
                self.logger.error(f"Error analyzing text {i}: {e}")
                results.append({
                    'text': text,
                    'emotions': [],
                    'timestamp': datetime.now(),
                    'status': 'error',
                    'error': str(e)
                })

        return pd.DataFrame(results)

    def extract_emotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract emotion features from API responses"""
        features = []

        for _, row in df.iterrows():
            if row['status'] == 'success' and row['emotions']:
                # Get top emotion
                top_emotion = max(row['emotions'], key=lambda x: x['confidence'])

                # Create feature vector
                emotion_features = {
                    'text': row['text'],
                    'primary_emotion': top_emotion['emotion'],
                    'primary_confidence': top_emotion['confidence'],
                    'emotion_count': len(row['emotions']),
                    'timestamp': row['timestamp']
                }

                # Add individual emotion confidences
                for emotion in row['emotions']:
                    emotion_features[f"conf_{emotion['emotion']}"] = emotion['confidence']

                features.append(emotion_features)

        return pd.DataFrame(features)

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate performance and quality metrics"""
        metrics = {
            'total_requests': len(df),
            'successful_requests': len(df[df['status'] == 'success']),
            'error_rate': len(df[df['status'] == 'error']) / len(df),
            'avg_confidence': df[df['status'] == 'success']['primary_confidence'].mean(),
            'emotion_distribution': df[df['status'] == 'success']['primary_emotion'].value_counts().to_dict()
        }

        return metrics

class ModelMonitor:
    def __init__(self):
        self.performance_history = []

    def track_performance(self, metrics: dict):
        """Track model performance over time"""
        metrics['timestamp'] = datetime.now()
        self.performance_history.append(metrics)

    def detect_drift(self, window_size: int = 100) -> dict:
        """Detect performance drift"""
        if len(self.performance_history) < window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        recent_metrics = self.performance_history[-window_size:]
        historical_metrics = self.performance_history[:-window_size]

        # Calculate drift indicators
        recent_avg_conf = np.mean([m['avg_confidence'] for m in recent_metrics])
        historical_avg_conf = np.mean([m['avg_confidence'] for m in historical_metrics])

        confidence_drift = abs(recent_avg_conf - historical_avg_conf) / historical_avg_conf

        drift_detected = confidence_drift > 0.1  # 10% threshold

        return {
            'drift_detected': drift_detected,
            'confidence_drift': confidence_drift,
            'recent_avg_confidence': recent_avg_conf,
            'historical_avg_confidence': historical_avg_conf
        }

    def generate_report(self) -> str:
        """Generate performance report"""
        if not self.performance_history:
            return "No performance data available"

        latest = self.performance_history[-1]
        drift = self.detect_drift()

        report = f"""
        SAMO-DL Performance Report
        ==========================

        Latest Metrics:
        - Total Requests: {latest['total_requests']}
        - Success Rate: {(1 - latest['error_rate']) * 100:.2f}%
        - Average Confidence: {latest['avg_confidence']:.3f}

        Drift Analysis:
        - Drift Detected: {drift['drift_detected']}
        - Confidence Drift: {drift['confidence_drift']:.3f}

        Top Emotions:
        {chr(10).join([f"- {emotion}: {count}" for emotion, count in list(latest['emotion_distribution'].items())[:5]])}
        """

        return report

# Usage example
if __name__ == "__main__":
    # Initialize components
    collector = EmotionDataCollector()
    monitor = ModelMonitor()

    # Sample texts for analysis
    sample_texts = [
        "I'm feeling really happy today!",
        "This is frustrating and annoying.",
        "I love this new feature!",
        "I'm scared about the upcoming changes.",
        "What a wonderful surprise!"
    ]

    # Collect data
    results_df = collector.analyze_batch(sample_texts)
    features_df = collector.extract_emotion_features(results_df)

    # Calculate metrics
    metrics = collector.calculate_metrics(results_df)
    monitor.track_performance(metrics)

    # Generate report
    report = monitor.generate_report()
    print(report)

    # Save results
    features_df.to_csv('emotion_analysis_results.csv', index=False)
    print("Results saved to emotion_analysis_results.csv")
```

---

## Error Handling and Best Practices

### Comprehensive Error Handling

```python
def safe_emotion_detection(text: str) -> dict:
    """Safe emotion detection with comprehensive error handling"""
    if not text or len(text.strip()) == 0:
        return {"error": "Empty text provided"}

    if len(text) > 1000:
        return {"error": "Text too long (max 1000 characters)"}

    try:
        emotions = detect_emotion(text)
        if "error" in emotions:
            return emotions

        # Validate response format
        if not isinstance(emotions, list):
            return {"error": "Invalid response format"}

        return {"success": True, "emotions": emotions}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
```

### Rate Limiting and Retry Logic

```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """Decorator for retrying failed API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=1)
def robust_emotion_detection(text: str) -> dict:
    """Robust emotion detection with retry logic"""
    return detect_emotion(text)
```

---

## Performance Optimization

### Batch Processing

```python
async def batch_emotion_analysis(texts: list, batch_size: int = 10) -> list:
    """Process multiple texts in batches for better performance"""
    import asyncio
    import aiohttp

    async def analyze_batch(batch_texts):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for text in batch_texts:
                task = session.post(
                    'https://samo-emotion-api-xxxxx-ew.a.run.app/predict',
                    json={'text': text},
                    headers={'Content-Type': 'application/json'}
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses

    # Process in batches
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = await analyze_batch(batch)
        results.extend(batch_results)

    return results
```

---

## Testing and Validation

### Unit Tests

```python
import unittest
from unittest.mock import patch, Mock

class TestEmotionDetection(unittest.TestCase):

    @patch('requests.post')
    def test_successful_emotion_detection(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = [
            {"emotion": "joy", "confidence": 0.89}
        ]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = detect_emotion("I'm happy!")

        self.assertEqual(result[0]['emotion'], 'joy')
        self.assertEqual(result[0]['confidence'], 0.89)

    @patch('requests.post')
    def test_api_error_handling(self, mock_post):
        # Mock API error
        mock_post.side_effect = Exception("API Error")

        result = detect_emotion("Test text")

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Failed to analyze emotions')

if __name__ == '__main__':
    unittest.main()
```

---

## Security Considerations

### Input Validation

```python
import re
from typing import Optional

def validate_text_input(text: str) -> Optional[str]:
    """Validate and sanitize text input"""
    if not text or not isinstance(text, str):
        return "Text must be a non-empty string"

    if len(text) > 1000:
        return "Text too long (maximum 1000 characters)"

    # Check for potentially malicious content
    suspicious_patterns = [
        r'<script.*?>.*?</script>',  # XSS attempts
        r'javascript:',              # JavaScript injection
        r'data:text/html',           # Data URI attacks
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "Text contains potentially malicious content"

    return None  # Validation passed

def secure_emotion_detection(text: str) -> dict:
    """Secure emotion detection with input validation"""
    validation_error = validate_text_input(text)
    if validation_error:
        return {"error": validation_error}

    return detect_emotion(text)
```

---

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_emotion_analysis(text: str, emotions: list, duration_ms: float):
    """Log emotion analysis with structured data"""
    log_data = {
        'event': 'emotion_analysis',
        'text_length': len(text),
        'emotions_detected': len(emotions),
        'primary_emotion': emotions[0]['emotion'] if emotions else None,
        'primary_confidence': emotions[0]['confidence'] if emotions else None,
        'duration_ms': duration_ms,
        'timestamp': datetime.now().isoformat()
    }

    logger.info(
        "Emotion analysis completed",
        extra={'log_data': json.dumps(log_data)}
    )
```

---

## Conclusion

This integration guide provides comprehensive instructions for integrating with the SAMO-DL emotion detection API across different platforms and use cases. The examples demonstrate best practices for error handling, performance optimization, security, and monitoring.

For additional support or questions, please refer to the main project documentation or contact the development team.
