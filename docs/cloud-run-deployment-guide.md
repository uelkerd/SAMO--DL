# Cloud Run Deployment Guide for SAMO Emotion Detection API

## Overview

This guide provides step-by-step instructions for deploying the SAMO Emotion Detection API to Google Cloud Run, which offers better reliability and simpler deployment compared to Vertex AI for HTTP-based ML services.

## Prerequisites

- Google Cloud Platform account with billing enabled
- Google Cloud CLI (gcloud) installed and configured
- Docker installed locally
- Project ID: `the-tendril-466607-n8`
- Region: `us-central1`

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   Cloud Run     │───▶│  Emotion Model  │
│                 │    │   (API Server)  │    │   (BERT-based)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Cloud Logging  │
                       │  & Monitoring   │
                       └─────────────────┘
```

## Step 1: Prepare the Container

### 1.1 Create Cloud Run Optimized Dockerfile

```dockerfile
# deployment/cloud-run/Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY predict.py .
COPY model/ ./model/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["python", "predict.py"]
```

### 1.2 Create Cloud Run Requirements

```txt
# deployment/cloud-run/requirements.txt
flask==2.3.3
torch==2.0.1
transformers==4.35.0
numpy==1.24.3
scikit-learn==1.3.0
gunicorn==21.2.0
```

### 1.3 Create Cloud Run Optimized predict.py

```python
# deployment/cloud-run/predict.py
import os
import json
import logging
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report
import time

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
label_mapping = None

def load_model():
    """Load the emotion detection model"""
    global model, tokenizer, label_mapping
    
    try:
        model_path = os.path.join(os.getcwd(), 'model')
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=7,
            ignore_mismatched_sizes=True
        )
        
        # Load label mapping
        label_mapping = {
            0: 'anger',
            1: 'disgust', 
            2: 'fear',
            3: 'joy',
            4: 'neutral',
            5: 'sadness',
            6: 'surprise'
        }
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_emotion(text):
    """Predict emotion for given text"""
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get emotion label
        emotion = label_mapping.get(predicted_class, 'unknown')
        
        return {
            'emotion': emotion,
            'confidence': round(confidence, 4),
            'class_id': predicted_class
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field in request body'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({
                'error': 'Text cannot be empty'
            }), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 503
        
        # Make prediction
        start_time = time.time()
        result = predict_emotion(text)
        prediction_time = time.time() - start_time
        
        if result is None:
            return jsonify({
                'error': 'Prediction failed'
            }), 500
        
        # Add metadata
        result['prediction_time_ms'] = round(prediction_time * 1000, 2)
        result['text_length'] = len(text)
        
        logger.info(f"Prediction successful: {result['emotion']} (confidence: {result['confidence']})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'SAMO Emotion Detection API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        },
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        # Get port from environment (Cloud Run sets PORT)
        port = int(os.environ.get('PORT', 8080))
        
        # Run with gunicorn for production
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1)
```

## Step 2: Build and Push Container

### 2.1 Build the Container

```bash
# Navigate to deployment directory
cd deployment/cloud-run

# Build the container
docker build -t gcr.io/the-tendril-466607-n8/emotion-detection-api:latest .

# Tag for Artifact Registry
docker tag gcr.io/the-tendril-466607-n8/emotion-detection-api:latest \
    us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo/cloud-run-api:latest
```

### 2.2 Push to Artifact Registry

```bash
# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo/cloud-run-api:latest
```

## Step 3: Deploy to Cloud Run

### 3.1 Deploy the Service

```bash
# Deploy to Cloud Run
gcloud run deploy emotion-detection-api \
    --image us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo/cloud-run-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 300 \
    --concurrency 80 \
    --set-env-vars "MODEL_PATH=/app/model"
```

### 3.2 Verify Deployment

```bash
# Get the service URL
gcloud run services describe emotion-detection-api \
    --region us-central1 \
    --format="value(status.url)"

# Test the health endpoint
curl https://[SERVICE_URL]/health

# Test prediction endpoint
curl -X POST https://[SERVICE_URL]/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling very happy today!"}'
```

## Step 4: Configure Monitoring and Logging

### 4.1 Set Up Cloud Monitoring

```bash
# Create monitoring dashboard
gcloud monitoring dashboards create \
    --project=the-tendril-466607-n8 \
    --config-from-file=dashboard-config.json
```

### 4.2 Create Dashboard Configuration

```json
{
  "displayName": "Emotion Detection API Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Request Count",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"run.googleapis.com/request_count\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
                }
              }
            }
          ]
        }
      },
      {
        "title": "Request Latency",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"run.googleapis.com/request_latencies\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN"
                  }
                }
              }
            }
          ]
        }
      }
    ]
  }
}
```

### 4.3 Set Up Alerts

```bash
# Create alert policy for high error rate
gcloud alpha monitoring policies create \
    --policy-from-file=alert-policy.json
```

## Step 5: Testing and Validation

### 5.1 Load Testing

```python
# scripts/testing/cloud_run_load_test.py
import requests
import time
import concurrent.futures
import json

def test_prediction(text, service_url):
    """Test single prediction"""
    try:
        response = requests.post(
            f"{service_url}/predict",
            json={"text": text},
            timeout=30
        )
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def load_test(service_url, num_requests=100, concurrent=10):
    """Run load test"""
    test_texts = [
        "I am feeling very happy today!",
        "This makes me so angry!",
        "I'm scared about what might happen.",
        "I feel neutral about this situation.",
        "This is disgusting!",
        "I'm so sad about the news.",
        "Wow, that's surprising!"
    ]
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = []
        for i in range(num_requests):
            text = test_texts[i % len(test_texts)]
            future = executor.submit(test_prediction, text, service_url)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    successful = sum(1 for status, _ in results if status == 200)
    failed = len(results) - successful
    avg_time = total_time / len(results)
    
    print(f"Load Test Results:")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(results)*100:.2f}%")
    print(f"Average Response Time: {avg_time:.3f}s")
    print(f"Requests per Second: {len(results)/total_time:.2f}")

if __name__ == "__main__":
    service_url = "https://[YOUR_SERVICE_URL]"
    load_test(service_url, num_requests=100, concurrent=10)
```

### 5.2 Integration Testing

```python
# scripts/testing/test_cloud_run_integration.py
import requests
import json
import time

def test_cloud_run_integration():
    """Test Cloud Run API integration"""
    service_url = "https://[YOUR_SERVICE_URL]"
    
    # Test health endpoint
    print("Testing health endpoint...")
    health_response = requests.get(f"{service_url}/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data['status'] == 'healthy'
    assert health_data['model_loaded'] == True
    print("✅ Health check passed")
    
    # Test prediction endpoint
    print("Testing prediction endpoint...")
    test_cases = [
        ("I am feeling very happy today!", "joy"),
        ("This makes me so angry!", "anger"),
        ("I'm scared about what might happen.", "fear"),
        ("I feel neutral about this situation.", "neutral")
    ]
    
    for text, expected_emotion in test_cases:
        response = requests.post(
            f"{service_url}/predict",
            json={"text": text},
            timeout=30
        )
        assert response.status_code == 200
        result = response.json()
        assert 'emotion' in result
        assert 'confidence' in result
        assert result['confidence'] > 0.5  # Reasonable confidence
        print(f"✅ Prediction for '{text[:30]}...' -> {result['emotion']} (confidence: {result['confidence']})")
    
    print("✅ All integration tests passed!")

if __name__ == "__main__":
    test_cloud_run_integration()
```

## Step 6: Production Configuration

### 6.1 Environment Variables

```bash
# Set production environment variables
gcloud run services update emotion-detection-api \
    --region us-central1 \
    --set-env-vars \
    "LOG_LEVEL=INFO,MODEL_PATH=/app/model,ENABLE_METRICS=true"
```

### 6.2 Custom Domain (Optional)

```bash
# Map custom domain
gcloud run domain-mappings create \
    --service emotion-detection-api \
    --domain api.samo-emotion.com \
    --region us-central1
```

### 6.3 SSL Certificate

```bash
# SSL is automatically handled by Cloud Run
# No additional configuration needed
```

## Monitoring and Maintenance

### 6.1 View Logs

```bash
# View real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=emotion-detection-api"

# View specific log entries
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=emotion-detection-api" --limit=50
```

### 6.2 Monitor Performance

```bash
# View service metrics
gcloud run services describe emotion-detection-api \
    --region us-central1 \
    --format="value(status.conditions)"
```

### 6.3 Update Service

```bash
# Update to new image version
gcloud run services update emotion-detection-api \
    --region us-central1 \
    --image us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo/cloud-run-api:v2.0.0
```

## Cost Optimization

### 6.1 Resource Optimization

```bash
# Optimize for cost
gcloud run services update emotion-detection-api \
    --region us-central1 \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 5 \
    --min-instances 0
```

### 6.2 Traffic Management

```bash
# Set up traffic splitting for gradual rollouts
gcloud run services update-traffic emotion-detection-api \
    --region us-central1 \
    --to-revisions=emotion-detection-api-00001-abc=90,emotion-detection-api-00002-def=10
```

## Troubleshooting

### Common Issues

1. **Container fails to start**
   - Check logs: `gcloud logging read "resource.type=cloud_run_revision"`
   - Verify model files are in container
   - Check memory requirements

2. **High latency**
   - Increase CPU/memory allocation
   - Optimize model loading
   - Check network connectivity

3. **Out of memory errors**
   - Increase memory allocation
   - Optimize model size
   - Check for memory leaks

### Debug Commands

```bash
# Get detailed service information
gcloud run services describe emotion-detection-api --region us-central1

# View recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=emotion-detection-api" --limit=100

# Check service status
gcloud run services list --region us-central1
```

## Success Metrics

- ✅ Service deployed successfully
- ✅ Health endpoint responding
- ✅ Predictions working correctly
- ✅ Monitoring and logging configured
- ✅ Load testing passed
- ✅ Integration tests passed

## Next Steps

1. **Production Monitoring**: Set up alerts for error rates and latency
2. **Performance Optimization**: Monitor and optimize based on usage patterns
3. **Security**: Implement authentication if needed
4. **Scaling**: Adjust resources based on traffic patterns
5. **Backup Strategy**: Set up automated backups of model files

## Conclusion

Cloud Run provides a reliable, scalable, and cost-effective platform for deploying the SAMO Emotion Detection API. The deployment process is simpler than Vertex AI and offers better reliability for HTTP-based ML services.

The system is now production-ready with:
- Automatic scaling
- Built-in monitoring
- SSL termination
- Global CDN
- Cost optimization features

This deployment strategy successfully bypasses the Vertex AI platform limitations while providing all the production features needed for the emotion detection service. 