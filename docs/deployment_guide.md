# SAMO Emotion Detection - Deployment Guide

## Overview

This guide covers deployment of the SAMO Emotion Detection API for both local development and production environments. The system supports multiple deployment options including local development, Docker containers, and cloud platforms.

## Prerequisites

### System Requirements

- **Python**: 3.8+ (3.12 recommended)
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for model and dependencies
- **Network**: Internet access for initial setup

### Software Dependencies

- **Python packages**: See `requirements.txt`
- **Model files**: Pre-trained emotion detection model
- **Optional**: Docker for containerized deployment

## Local Development Deployment

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SAMO--DL
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Verify model files**
   ```bash
   ls -la local_deployment/model/
   # Should show: config.json, pytorch_model.bin, tokenizer files, etc.
   ```

4. **Start the API server**
   ```bash
   cd local_deployment
   python api_server.py
   ```

5. **Test the deployment**
   ```bash
   # In another terminal
   python test_api.py
   ```

### Detailed Local Setup

#### Step 1: Environment Preparation

```bash
# Navigate to project directory
cd /path/to/SAMO--DL

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Model Verification

```bash
# Check model directory structure
ls -la local_deployment/model/

# Expected files:
# - config.json
# - pytorch_model.bin
# - tokenizer.json
# - tokenizer_config.json
# - vocab.txt
# - special_tokens_map.json
```

#### Step 3: Configuration

The API server uses default configuration. For customization, modify `local_deployment/api_server.py`:

```python
# Rate limiting (requests per minute)
RATE_LIMIT_MAX_REQUESTS = 100

# Server port
app.run(host='0.0.0.0', port=8000, debug=False)
```

#### Step 4: Start Server

```bash
cd local_deployment
python api_server.py
```

Expected output:
```
🔧 Loading emotion detection model...
Loading model from: /path/to/SAMO--DL/local_deployment/model
⚠️ CUDA not available, using CPU
✅ Model loaded successfully
🌐 Starting enhanced local API server...
📋 Available endpoints:
   GET  / - API documentation
   GET  /health - Health check with metrics
   GET  /metrics - Detailed server metrics
   POST /predict - Single prediction
   POST /predict_batch - Batch prediction
🚀 Server starting on http://localhost:8000
🔒 Rate limiting: 100 requests per 60 seconds
📊 Monitoring: Comprehensive metrics and logging enabled
```

#### Step 5: Verification

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'

# Run comprehensive tests
python test_api.py
```

### Troubleshooting Local Deployment

#### Common Issues

**Port 8000 already in use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
# Edit api_server.py and change port=8000 to port=8001
```

**Model loading errors**
```bash
# Check model files
ls -la local_deployment/model/

# Reinstall transformers
pip install --upgrade transformers torch

# Check Python version
python --version  # Should be 3.8+
```

**Memory issues**
```bash
# Monitor memory usage
htop

# Reduce batch size in api_server.py if needed
# Look for max_length parameter in tokenizer
```

**Permission errors**
```bash
# Fix file permissions
chmod +x local_deployment/start.sh
chmod 755 local_deployment/model/
```

## Docker Deployment

### Build and Run with Docker

1. **Build the image**
   ```bash
   cd deployment
   docker build -t samo-emotion-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 samo-emotion-api
   ```

3. **Test the deployment**
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Compose

For easier management, use Docker Compose:

```bash
cd deployment
docker-compose up -d
```

The `docker-compose.yml` file includes:
- API service
- Volume mounts for logs
- Port mapping
- Health checks

### Custom Docker Configuration

Edit `deployment/dockerfile` for custom configurations:

```dockerfile
# Change base image
FROM python:3.12-slim

# Add custom dependencies
RUN pip install --no-cache-dir gunicorn

# Change working directory
WORKDIR /app

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Use production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "api_server:app"]
```

## Production Deployment

### GCP/Vertex AI Deployment

#### Prerequisites

1. **Google Cloud SDK**
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   gcloud init
   ```

2. **Enable APIs**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

3. **Authentication**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

#### Deployment Steps

1. **Prepare deployment package**
   ```bash
   python scripts/deploy_to_gcp_vertex_ai.py
   ```

2. **Deploy to Vertex AI**
   ```bash
   # The script will handle the deployment
   # Monitor progress in the output
   ```

3. **Verify deployment**
   ```bash
   # Get endpoint URL from script output
   curl https://your-endpoint-url/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I am happy"}'
   ```

### AWS Deployment

#### Using AWS Lambda

1. **Package the application**
   ```bash
   # Create deployment package
   pip install -r requirements.txt -t package/
   cp -r local_deployment/* package/
   cd package
   zip -r ../lambda-deployment.zip .
   ```

2. **Deploy to Lambda**
   ```bash
   aws lambda create-function \
     --function-name samo-emotion-api \
     --runtime python3.9 \
     --handler api_server.app \
     --zip-file fileb://lambda-deployment.zip \
     --timeout 30 \
     --memory-size 1024
   ```

#### Using AWS ECS

1. **Create ECS task definition**
   ```json
   {
     "family": "samo-emotion-api",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "samo-emotion-api",
         "image": "your-ecr-repo/samo-emotion-api:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ]
       }
     ]
   }
   ```

2. **Deploy to ECS**
   ```bash
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   aws ecs create-service --cluster your-cluster --service-name samo-emotion-api --task-definition samo-emotion-api
   ```

### Azure Deployment

#### Using Azure Container Instances

1. **Build and push image**
   ```bash
   # Build image
   docker build -t samo-emotion-api .
   
   # Tag for Azure
   docker tag samo-emotion-api your-registry.azurecr.io/samo-emotion-api:latest
   
   # Push to Azure Container Registry
   docker push your-registry.azurecr.io/samo-emotion-api:latest
   ```

2. **Deploy to ACI**
   ```bash
   az container create \
     --resource-group your-rg \
     --name samo-emotion-api \
     --image your-registry.azurecr.io/samo-emotion-api:latest \
     --ports 8000 \
     --dns-name-label samo-emotion-api
   ```

## Monitoring and Logging

### Local Monitoring

1. **View logs**
   ```bash
   tail -f local_deployment/api_server.log
   ```

2. **Check metrics**
   ```bash
   curl http://localhost:8000/metrics | python -m json.tool
   ```

3. **Health monitoring**
   ```bash
   curl http://localhost:8000/health
   ```

### Production Monitoring

#### GCP Monitoring

```bash
# Enable monitoring
gcloud services enable monitoring.googleapis.com

# Create monitoring dashboard
# Use the /metrics endpoint data
```

#### AWS CloudWatch

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name "SAMO-Emotion-API" \
  --dashboard-body file://dashboard.json
```

#### Azure Monitor

```bash
# Enable Application Insights
az monitor app-insights component create \
  --app samo-emotion-api \
  --location eastus \
  --resource-group your-rg
```

## Security Considerations

### Authentication

For production deployments, implement authentication:

```python
# Add to api_server.py
from functools import wraps
import os

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Apply to endpoints
@app.route('/predict', methods=['POST'])
@require_api_key
@rate_limit
def predict():
    # ... existing code
```

### HTTPS

For production, always use HTTPS:

```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update server configuration
app.run(host='0.0.0.0', port=8000, ssl_context=('cert.pem', 'key.pem'))
```

### Environment Variables

Use environment variables for sensitive configuration:

```bash
# Create .env file
API_KEY=your-secret-api-key
MODEL_PATH=/path/to/model
RATE_LIMIT_MAX_REQUESTS=100
LOG_LEVEL=INFO
```

## Performance Optimization

### Model Optimization

1. **Quantization**
   ```python
   # Add to model loading
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **ONNX Export**
   ```python
   import torch.onnx
   
   # Export model to ONNX
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

### Server Optimization

1. **Use Gunicorn**
   ```bash
   pip install gunicorn
   gunicorn --workers 4 --bind 0.0.0.0:8000 api_server:app
   ```

2. **Enable Caching**
   ```python
   from flask_caching import Cache
   
   cache = Cache(app, config={'CACHE_TYPE': 'simple'})
   
   @cache.memoize(timeout=300)
   def cached_predict(text):
       return model.predict(text)
   ```

## Backup and Recovery

### Model Backup

```bash
# Create backup
tar -czf model_backup_$(date +%Y%m%d).tar.gz local_deployment/model/

# Restore from backup
tar -xzf model_backup_20231201.tar.gz -C local_deployment/
```

### Configuration Backup

```bash
# Backup configuration
cp local_deployment/api_server.py local_deployment/api_server.py.backup

# Restore configuration
cp local_deployment/api_server.py.backup local_deployment/api_server.py
```

## Scaling

### Horizontal Scaling

1. **Load Balancer Configuration**
   ```nginx
   upstream samo_api {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
       server 127.0.0.1:8002;
   }
   
   server {
       listen 80;
       location / {
           proxy_pass http://samo_api;
       }
   }
   ```

2. **Multiple Instances**
   ```bash
   # Start multiple instances
   python api_server.py --port 8000 &
   python api_server.py --port 8001 &
   python api_server.py --port 8002 &
   ```

### Vertical Scaling

1. **Increase Resources**
   ```bash
   # For Docker
   docker run -p 8000:8000 --memory=4g --cpus=2 samo-emotion-api
   
   # For Kubernetes
   resources:
     requests:
       memory: "2Gi"
       cpu: "1"
     limits:
       memory: "4Gi"
       cpu: "2"
   ```

## Maintenance

### Regular Tasks

1. **Log Rotation**
   ```bash
   # Set up logrotate
   sudo logrotate /etc/logrotate.d/samo-api
   ```

2. **Health Checks**
   ```bash
   # Create monitoring script
   #!/bin/bash
   response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
   if [ $response -ne 200 ]; then
       echo "API is down!"
       # Send alert
   fi
   ```

3. **Performance Monitoring**
   ```bash
   # Monitor response times
   curl -s http://localhost:8000/metrics | jq '.server_metrics.average_response_time_ms'
   ```

### Updates

1. **Model Updates**
   ```bash
   # Backup current model
   cp -r local_deployment/model local_deployment/model_backup_$(date +%Y%m%d)
   
   # Deploy new model
   cp -r new_model/* local_deployment/model/
   
   # Restart server
   pkill -f api_server
   python api_server.py &
   ```

2. **Code Updates**
   ```bash
   # Pull latest code
   git pull origin main
   
   # Update dependencies
   pip install -r requirements.txt
   
   # Restart server
   pkill -f api_server
   python api_server.py &
   ```

## Support and Troubleshooting

### Common Issues

1. **Server won't start**
   - Check port availability
   - Verify Python environment
   - Check model files

2. **High response times**
   - Monitor system resources
   - Check for memory leaks
   - Optimize model loading

3. **Rate limiting issues**
   - Adjust rate limit settings
   - Check client implementation
   - Monitor request patterns

### Getting Help

1. **Check logs**
   ```bash
   tail -f local_deployment/api_server.log
   ```

2. **Run diagnostics**
   ```bash
   python scripts/diagnose_issues.py
   ```

3. **Review metrics**
   ```bash
   curl http://localhost:8000/metrics
   ```

## Conclusion

This deployment guide covers the essential steps for deploying the SAMO Emotion Detection API in various environments. The system is designed to be robust, scalable, and production-ready with comprehensive monitoring and error handling.

For additional support or questions, refer to the API documentation and troubleshooting sections.
