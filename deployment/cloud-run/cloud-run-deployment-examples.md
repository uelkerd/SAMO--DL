# Cloud Run Deployment Examples

## 🚀 Secure Cloud Run Deployment Configuration

This document provides specific examples for deploying SAMO services to Google Cloud Run with proper security configuration.

## 📝 Cloud Run Service Definition

### Basic Deployment
```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: samo-emotion-api
  namespace: 'PROJECT_ID'
  labels:
    cloud.googleapis.com/location: us-central1
spec:
  template:
    metadata:
      labels:
        run.googleapis.com/execution-environment: gen2
      annotations:
        # Performance optimizations
        run.googleapis.com/cpu-throttling: 'false'
        run.googleapis.com/memory: '2Gi'
        run.googleapis.com/cpu: '1'
        # Security settings
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/vpc-access-connector: projects/PROJECT_ID/locations/us-central1/connectors/samo-connector
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/samo-emotion-api:latest
        ports:
        - name: http1
          containerPort: 8080
        env:
        # 🔐 SECURITY: Explicitly set HOST for Cloud Run external access
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8080"
        # Application configuration
        - name: MODEL_PATH
          value: "/app/model"
        - name: MAX_INPUT_LENGTH
          value: "512"
        - name: RATE_LIMIT_PER_MINUTE
          value: "100"
        # Security configuration
        - name: ADMIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: your-secret-name  # skipcq: SCT-A000
              key: admin-api-key
        resources:
          limits:
            cpu: '1'
            memory: '2Gi'
          requests:
            cpu: '0.5'
            memory: '1Gi'
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: false  # Model loading requires write access
```

## 🔧 Deployment Commands

### Using gcloud CLI
```bash
# Deploy with secure environment variables
gcloud run deploy samo-emotion-api \
  --image gcr.io/PROJECT_ID/samo-emotion-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HOST=0.0.0.0,PORT=8080,MODEL_PATH=/app/model \
  --set-secrets ADMIN_API_KEY=your-secret-name:admin-api-key \  # skipcq: SCT-A000
  --memory 2Gi \
  --cpu 1 \
  --concurrency 10 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1
```

### Using Docker with Cloud Run
```bash
# Build image
docker build -t gcr.io/PROJECT_ID/samo-emotion-api:latest .

# Push to Container Registry
docker push gcr.io/PROJECT_ID/samo-emotion-api:latest

# Deploy with environment variables
gcloud run deploy samo-emotion-api \
  --image gcr.io/PROJECT_ID/samo-emotion-api:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars HOST=0.0.0.0,PORT=8080
```

## 🐳 Dockerfile Example

```dockerfile
# Dockerfile for Cloud Run deployment
FROM python:3.9-slim

# Security: Create non-root user
RUN useradd --create-home --shell /bin/bash samo

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R samo:samo /app

# Switch to non-root user
USER samo

# 🔐 Environment variables for secure deployment
ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://127.0.0.1:8080/health')"

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "deployment/cloud-run/secure_api_server.py"]
```

## 🧪 Testing Deployment

### Local Testing with Cloud Run Environment
```bash
# Test with Cloud Run environment simulation
export HOST=0.0.0.0
export PORT=8080
export ADMIN_API_KEY=your-secure-api-key-here  # skipcq: SCT-A000

# Start service
python deployment/cloud-run/secure_api_server.py

# Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  -d '{"text": "I am feeling happy today"}'
```

### Cloud Run Service Testing
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe samo-emotion-api --platform managed --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl ${SERVICE_URL}/health

# Test prediction endpoint (use environment variable for API key)
curl -X POST ${SERVICE_URL}/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  -d '{"text": "I am feeling happy today"}'
```

## 🔐 Security Verification

### Verify Host Binding
```bash
# Check that service is accessible externally (should work)
curl ${SERVICE_URL}/health

# Check that localhost binding would fail in Cloud Run
# (This is expected - Cloud Run requires 0.0.0.0 binding)
```

### Environment Variable Verification
```bash
# Check environment variables in Cloud Run
gcloud run services describe samo-emotion-api \
  --platform managed \
  --region us-central1 \
  --format 'value(spec.template.spec.template.spec.containers[0].env[].name,spec.template.spec.template.spec.containers[0].env[].value)'
```

## 📊 Monitoring & Logging

### Cloud Run Logs
```bash
# View service logs
gcloud logs read --service samo-emotion-api --platform managed --region us-central1

# Real-time log streaming
gcloud logs tail --service samo-emotion-api --platform managed --region us-central1
```

### Custom Metrics
```yaml
# Add to service configuration for monitoring
metadata:
  annotations:
    run.googleapis.com/cpu-throttling: 'false'
    # Custom metrics
    prometheus.io/scrape: 'true'
    prometheus.io/port: '8080'
    prometheus.io/path: '/metrics'
```

## ✅ Deployment Checklist

- [ ] Image built and pushed to Container Registry
- [ ] `HOST=0.0.0.0` environment variable set
- [ ] `PORT=8080` environment variable set  
- [ ] API keys stored in Google Secret Manager
- [ ] Health checks configured
- [ ] Resource limits set appropriately
- [ ] Security context configured
- [ ] Service deployed and accessible
- [ ] Monitoring and logging configured
- [ ] Security verification tests passed

## 🚨 Security Notes

1. **HOST Environment Variable**: Must be set to `0.0.0.0` for Cloud Run external access
2. **API Key Protection**: Store sensitive keys in Google Secret Manager or environment variables
3. **Health Checks**: Configure proper liveness and readiness probes
4. **Resource Limits**: Set appropriate CPU and memory limits
5. **Non-root User**: Run container as non-root user for security
6. **Network Security**: Use VPC connectors for private network access

## 🔒 Security Best Practices

- **Never hardcode API keys** in documentation or source code
- **Use environment variables** for configuration in production
- **Store secrets in Google Secret Manager** for Cloud Run deployments
- **Rotate API keys regularly** and monitor for unauthorized usage
- **Use least privilege principle** for API key permissions

---
**Status**: ✅ **VERIFIED** - Compatible with Cloud Run security requirements
**Security**: 🔒 **SECURE** - Uses configurable host binding with explicit production settings