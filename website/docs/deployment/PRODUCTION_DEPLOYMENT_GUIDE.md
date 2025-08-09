# Production Deployment Guide - SAMO-DL

## 🚀 Overview

This guide provides comprehensive instructions for deploying the SAMO-DL emotion detection API to production environments with enterprise-grade security and reliability.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection
- **OS**: Ubuntu 20.04+ or equivalent

### Software Requirements
- **Docker**: 20.10+
- **Python**: 3.10+
- **Git**: Latest version
- **Make**: For automation scripts

### Cloud Platform Requirements
- **Google Cloud Platform** (recommended)
  - Cloud Run enabled
  - Container Registry enabled
  - IAM permissions configured
- **Alternative**: AWS, Azure, or self-hosted

## 🔐 Security Checklist

### Before Deployment
- [ ] **Environment Variables**: All secrets configured
- [ ] **API Keys**: Generated and secured
- [ ] **SSL Certificates**: Valid certificates installed
- [ ] **Firewall Rules**: Properly configured
- [ ] **Access Control**: IAM roles assigned
- [ ] **Monitoring**: Logging and alerting configured

### Security Configuration
```bash
# Required environment variables
# SECURITY NOTE: Replace ALL placeholder values with actual secure credentials
export DATABASE_URL="postgresql://USERNAME:PASSWORD@HOSTNAME:PORT/DATABASE"
export SECRET_KEY="REPLACE_WITH_SECURE_SECRET_KEY"
export API_KEY="REPLACE_WITH_SECURE_API_KEY"
export ENVIRONMENT="production"
export OPENAI_API_KEY="REPLACE_WITH_OPENAI_API_KEY"
export GOOGLE_CLOUD_CREDENTIALS="path/to/credentials.json"
```

## 🏗️ Deployment Options

### Option 1: Google Cloud Run (Recommended)

#### Step 1: Prepare Environment
```bash
# Clone repository
git clone https://github.com/uelkerd/SAMO--DL.git
cd SAMO--DL

# Set up authentication
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

#### Step 2: Build and Deploy
```bash
# Build container
docker build -f deployment/cloud-run/Dockerfile -t gcr.io/YOUR_PROJECT_ID/samo-dl-api .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/samo-dl-api

# Deploy to Cloud Run
gcloud run deploy samo-dl-api \
  --image gcr.io/YOUR_PROJECT_ID/samo-dl-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production
```

#### Step 3: Configure Domain and SSL
```bash
# Map custom domain
gcloud run domain-mappings create \
  --service samo-dl-api \
  --domain api.samo-project.com \
  --region us-central1
```

### Option 2: Self-Hosted with Docker

#### Step 1: Prepare Server
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### Step 2: Deploy with Docker Compose
```bash
# Create production docker-compose file
cat > docker-compose.prod.yml << EOF
version: '3.8'
services:
  samo-dl-api:
    build:
      context: .
      dockerfile: deployment/cloud-run/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=\${DATABASE_URL}
      - SECRET_KEY=\${SECRET_KEY}
      - API_KEY=\${API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - ./logs:/app/logs
      - ./models:/app/model
EOF

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Option 3: Kubernetes Deployment

#### Step 1: Create Kubernetes Manifests
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: samo-dl-api
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
      - name: samo-dl-api
        image: gcr.io/YOUR_PROJECT_ID/samo-dl-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: samo-dl-secrets
              key: database-url
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: samo-dl-secrets
              key: secret-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
---
apiVersion: v1
kind: Service
metadata:
  name: samo-dl-api-service
spec:
  selector:
    app: samo-dl-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Step 2: Deploy to Kubernetes
```bash
# Create secrets
kubectl create secret generic samo-dl-secrets \
  --from-literal=database-url="$DATABASE_URL" \
  --from-literal=secret-key="$SECRET_KEY" \
  --from-literal=api-key="$API_KEY"

# Deploy application
kubectl apply -f k8s-deployment.yaml
```

## 🔧 Configuration

### Environment Variables
| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ENVIRONMENT` | Yes | Environment name | `production` |
| `DATABASE_URL` | Yes | Database connection string | `postgresql://...` |
| `SECRET_KEY` | Yes | Application secret key | `your-secret-key` |
| `API_KEY` | Yes | API authentication key | `your-api-key` |
| `OPENAI_API_KEY` | No | OpenAI API key | `sk-...` |
| `GOOGLE_CLOUD_CREDENTIALS` | No | GCP credentials path | `/path/to/credentials.json` |
| `LOG_LEVEL` | No | Logging level | `INFO` |
| `PORT` | No | Server port | `8080` |

### Security Headers
The application automatically includes security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'`

### Rate Limiting
- **Per API Key**: 60 requests per minute
- **Per User**: 100 requests per hour
- **Burst Limit**: 10 requests

## 📊 Monitoring and Logging

### Health Checks
```bash
# Check API health
curl https://api.samo-project.com/health

# Expected response
{
  "status": "healthy",
  "model_status": "loaded",
  "port": "8080",
  "timestamp": 1640995200.0
}
```

### Logging Configuration
```yaml
# Log levels by environment
production:
  log_level: "WARNING"
  log_to_file: true
  log_to_console: false

development:
  log_level: "DEBUG"
  log_to_file: true
  log_to_console: true
```

### Monitoring Endpoints
- `/health` - Basic health check
- `/model_status` - Detailed model status
- `/metrics` - Prometheus metrics (if enabled)

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./deployment/cloud-run/Dockerfile
        push: true
        tags: gcr.io/${{ secrets.GCP_PROJECT_ID }}/samo-dl-api:${{ github.sha }}
    
    - name: Deploy to Cloud Run
      uses: google-github-actions/deploy-cloudrun@v1
      with:
        service: samo-dl-api
        image: gcr.io/${{ secrets.GCP_PROJECT_ID }}/samo-dl-api:${{ github.sha }}
        region: us-central1
        env_vars: ENVIRONMENT=production
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check model files
ls -la /app/model/

# Check model logs
docker logs samo-dl-api | grep -i model

# Verify model path in container
docker exec -it samo-dl-api ls -la /app/model/
```

#### 2. Memory Issues
```bash
# Check memory usage
docker stats samo-dl-api

# Increase memory limits
docker run --memory=4g --memory-swap=4g ...
```

#### 3. Rate Limiting Issues
```bash
# Check rate limit headers
curl -I https://api.samo-project.com/predict

# Monitor rate limit logs
docker logs samo-dl-api | grep -i "rate limit"
```

#### 4. Database Connection Issues
```bash
# Test database connection
docker exec -it samo-dl-api python -c "
import os
import psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
print('Database connection successful')
conn.close()
"
```

### Performance Optimization

#### 1. Model Optimization
- Use model quantization for faster inference
- Implement model caching
- Consider model serving frameworks (TorchServe, TensorFlow Serving)

#### 2. Database Optimization
- Add database indexes
- Implement connection pooling
- Use read replicas for heavy read workloads

#### 3. Caching Strategy
- Implement Redis for session storage
- Cache frequently requested predictions
- Use CDN for static assets

## 🔒 Security Best Practices

### 1. Secrets Management
- Use external secrets management (HashiCorp Vault, AWS Secrets Manager)
- Rotate secrets regularly (90 days)
- Never commit secrets to version control

### 2. Network Security
- Use VPC for network isolation
- Implement proper firewall rules
- Use HTTPS for all communications

### 3. Container Security
- Run containers as non-root user
- Use read-only filesystem
- Drop unnecessary capabilities
- Scan images for vulnerabilities

### 4. API Security
- Implement proper authentication
- Use rate limiting
- Validate all inputs
- Log security events

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple instances
- Implement proper session management
- Consider database sharding for large datasets

### Vertical Scaling
- Monitor resource usage
- Scale up based on metrics
- Implement auto-scaling policies

### Performance Monitoring
- Use APM tools (New Relic, DataDog)
- Monitor response times
- Track error rates
- Set up alerting

## 🆘 Support and Maintenance

### Regular Maintenance
- **Weekly**: Security updates and dependency checks
- **Monthly**: Performance reviews and optimization
- **Quarterly**: Security audits and penetration testing

### Backup Strategy
- **Database**: Daily automated backups
- **Models**: Version-controlled model storage
- **Configuration**: Infrastructure as Code (IaC)

### Disaster Recovery
- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)
- **Backup Locations**: Multiple regions
- **Testing**: Monthly disaster recovery drills

## 📞 Support Contacts

- **Technical Issues**: tech-support@samo-project.com
- **Security Issues**: security@samo-project.com
- **Emergency**: +1-555-0123 (24/7)

---

**Last Updated**: August 5, 2025
**Version**: 1.0.0
**Maintainer**: SAMO-DL Team 