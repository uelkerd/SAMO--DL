# SAMO Deep Learning - Deployment & Infrastructure Guide

## 📋 Overview

This document provides comprehensive instructions for deploying SAMO Deep Learning models to production environments. It covers Docker configurations, environment management, scaling procedures, and infrastructure requirements.

## 🚀 Deployment Environments

### Development

```bash
# Clone the repository
git clone https://github.com/organization/samo-dl.git
cd samo-dl

# Set up environment
conda env create -f environment.yml
conda activate samo-dl

# Set up environment variables
cp .env.template .env
# Edit .env with appropriate values

# Run development server
python -m src.unified_ai_api
```

### Testing

```bash
# Set up testing environment
conda env create -f environment.yml
conda activate samo-dl

# Set environment to testing
export SAMO_ENV=testing

# Run tests
pytest tests/
```

### Production

Production deployment uses Docker containers orchestrated with Kubernetes.

## 🐳 Docker Configuration

### Base Docker Image

```dockerfile
# docker/Dockerfile.prod
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SAMO_ENV=production

# Run the application
CMD ["python", "-m", "src.unified_ai_api"]
```

### Docker Compose for Local Testing

```yaml
# docker-compose.yml
version: '3'

services:
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - MODEL_PATH=/app/models/checkpoints/
    volumes:
      - ./models/checkpoints:/app/models/checkpoints
    depends_on:
      - db

  db:
    image: postgres:14
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/database/schema.sql:/docker-entrypoint-initdb.d/schema.sql

volumes:
  postgres_data:
```

## ☁️ Kubernetes Deployment

### Namespace Setup

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: samo-dl
```

### Deployment Configuration

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: samo-dl-api
  namespace: samo-dl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: samo-dl-api
  template:
    metadata:
      labels:
        app: samo-dl-api
    spec:
      containers:
      - name: api
        image: ${ECR_REPOSITORY_URI}:${IMAGE_TAG}
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: samo-dl-secrets
              key: database-url
        - name: MODEL_PATH
          value: "/app/models/checkpoints"
        volumeMounts:
        - name: model-volume
          mountPath: /app/models/checkpoints
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-storage-pvc
```

### Service Configuration

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: samo-dl-api
  namespace: samo-dl
spec:
  selector:
    app: samo-dl-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

```yaml
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: samo-dl-api-ingress
  namespace: samo-dl
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.samo.ai
    secretName: samo-tls-secret
  rules:
  - host: api.samo.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: samo-dl-api
            port:
              number: 80
```

## 🔐 Secrets Management

### Kubernetes Secrets

```yaml
# kubernetes/secrets.yaml (Do not commit this file)
apiVersion: v1
kind: Secret
metadata:
  name: samo-dl-secrets
  namespace: samo-dl
type: Opaque
data:
  database-url: <base64-encoded-db-url>
  api-key: <base64-encoded-api-key>
```

### Environment Variables

Production environment variables are managed through Kubernetes secrets and ConfigMaps. For local development, use `.env` files (never commit these to version control).

## 📊 Scaling Strategies

### Horizontal Pod Autoscaler

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: samo-dl-api-hpa
  namespace: samo-dl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: samo-dl-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Model Serving Scaling

For high-throughput inference, consider:

1. **Batch Processing**: Group requests for more efficient processing
2. **Model Quantization**: Use compressed models in production
3. **GPU Acceleration**: For high-volume production environments

## 🔄 Continuous Deployment

### CI/CD Pipeline

Our CircleCI pipeline automatically deploys to production when changes are merged to the `main` branch:

1. Build and test application
2. Build Docker image
3. Push to ECR repository
4. Update Kubernetes deployment

### Deployment Commands

```bash
# Deploy manually (if needed)
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/hpa.yaml

# Check deployment status
kubectl get pods -n samo-dl
kubectl get services -n samo-dl
kubectl get ingress -n samo-dl
```

## 🔍 Health Checks

### Liveness Probe

```yaml
# Add to container spec in deployment.yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Readiness Probe

```yaml
# Add to container spec in deployment.yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## 📝 Deployment Checklist

Before deploying to production:

1. ✅ Run all tests (`pytest tests/`)
2. ✅ Check security vulnerabilities (`safety check`)
3. ✅ Verify model performance metrics
4. ✅ Update database schema if needed
5. ✅ Validate environment variables
6. ✅ Test Docker build locally
7. ✅ Verify API endpoints with Postman/curl

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model path in environment variables
   - Verify model file permissions in container
   - Ensure model format compatibility

2. **Database Connection Issues**
   - Verify database URL in secrets
   - Check network policies allow connection
   - Confirm database service is running

3. **Memory/CPU Limits**
   - Adjust resource requests/limits in deployment.yaml
   - Monitor resource usage with Prometheus
   - Consider model optimization if hitting limits

## 📈 Performance Optimization

### Production Optimizations

1. **Model Serving**
   - Use ONNX Runtime for inference
   - Implement request batching
   - Consider TorchServe for high-volume deployments

2. **Database**
   - Use connection pooling
   - Implement query caching
   - Consider read replicas for high-load scenarios

3. **API Performance**
   - Enable response compression
   - Implement appropriate caching headers
   - Use async processing for long-running tasks 