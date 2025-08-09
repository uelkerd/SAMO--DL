# 🚀 Deployment Guide

*Complete deployment guide for SAMO Brain's AI services across different environments and platforms.*

## 🎯 Deployment Overview

### Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        A[Load Balancer]
        B[API Gateway]
        C[Kubernetes Cluster]
        D[Database Cluster]
        E[Redis Cluster]
        F[Monitoring Stack]
    end
    
    subgraph "Development Environment"
        G[Local Docker]
        H[Development Database]
        I[Local Redis]
    end
    
    subgraph "Staging Environment"
        J[Staging Load Balancer]
        K[Staging API Gateway]
        L[Staging Kubernetes]
        M[Staging Database]
    end
    
    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    
    G --> H
    G --> I
    
    J --> K
    K --> L
    L --> M
    
    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style J fill:#e8f5e8
```

## 🏠 Local Development Setup

### Prerequisites

```bash
# Required software
- Python 3.12+
- Docker & Docker Compose
- Git
- Make (optional)

# System requirements
- 8GB RAM minimum
- 4 CPU cores minimum
- 10GB free disk space
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/samo-brain.git
cd samo-brain

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/samobrain
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=your-secret-key
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=samobrain
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

### Environment Configuration

```bash
# .env file
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/samobrain
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-here
API_KEY_SALT=your-api-key-salt-here

# Model Configuration
MODEL_PATH=/app/models/emotion_detection
MODEL_CACHE_SIZE=1000
MODEL_QUANTIZATION=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=INFO

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_REQUESTS_PER_DAY=10000

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Development
DEBUG=true
RELOAD=true
```

## ☁️ Cloud Deployment

### Google Cloud Platform (GCP)

#### GCP Setup Script

```bash
#!/bin/bash
# deploy-gcp.sh

set -e

# Configuration
PROJECT_ID="your-project-id"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="samo-brain-cluster"
NAMESPACE="samo-brain"

echo "🚀 Deploying SAMO Brain to GCP..."

# Enable required APIs
echo "📋 Enabling GCP APIs..."
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable redis.googleapis.com

# Create GKE cluster
echo "🏗️ Creating GKE cluster..."
gcloud container clusters create $CLUSTER_NAME \
    --zone=$ZONE \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10 \
    --enable-autorepair \
    --enable-autoupgrade

# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Create namespace
kubectl create namespace $NAMESPACE

# Create secrets
echo "🔐 Creating Kubernetes secrets..."
kubectl create secret generic samo-brain-secrets \
    --from-literal=database-url="$DATABASE_URL" \
    --from-literal=redis-url="$REDIS_URL" \
    --from-literal=jwt-secret="$JWT_SECRET_KEY" \
    --namespace=$NAMESPACE

# Deploy application
echo "📦 Deploying application..."
kubectl apply -f k8s/ -n $NAMESPACE

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl wait --for=condition=available --timeout=300s deployment/samo-brain-api -n $NAMESPACE

# Get service URL
echo "🌐 Getting service URL..."
kubectl get service samo-brain-service -n $NAMESPACE

echo "✅ Deployment completed successfully!"
```

#### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: samo-brain-api
  labels:
    app: samo-brain-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: samo-brain-api
  template:
    metadata:
      labels:
        app: samo-brain-api
    spec:
      containers:
      - name: api
        image: gcr.io/PROJECT_ID/samo-brain:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: samo-brain-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: samo-brain-secrets
              key: redis-url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: samo-brain-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: samo-brain-service
spec:
  selector:
    app: samo-brain-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: samo-brain-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "samo-brain-ip"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.samobrain.com
    secretName: samo-brain-tls
  rules:
  - host: api.samobrain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: samo-brain-service
            port:
              number: 80

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: samo-brain-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: samo-brain-api
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

### AWS Deployment

#### AWS EKS Setup

```bash
#!/bin/bash
# deploy-aws.sh

set -e

# Configuration
CLUSTER_NAME="samo-brain-cluster"
REGION="us-west-2"
NAMESPACE="samo-brain"

echo "🚀 Deploying SAMO Brain to AWS EKS..."

# Create EKS cluster
echo "🏗️ Creating EKS cluster..."
eksctl create cluster \
    --name $CLUSTER_NAME \
    --region $REGION \
    --nodegroup-name standard-workers \
    --node-type t3.medium \
    --nodes 3 \
    --nodes-min 1 \
    --nodes-max 10 \
    --managed

# Update kubeconfig
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

# Create namespace
kubectl create namespace $NAMESPACE

# Deploy application
echo "📦 Deploying application..."
kubectl apply -f k8s/aws/ -n $NAMESPACE

echo "✅ Deployment completed successfully!"
```

#### AWS-Specific Kubernetes Configs

```yaml
# k8s/aws/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2
  resources:
    requests:
      storage: 10Gi

---
# k8s/aws/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: samo-brain-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-scheme: internet-facing
spec:
  selector:
    app: samo-brain-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Azure AKS Setup

```bash
#!/bin/bash
# deploy-azure.sh

set -e

# Configuration
RESOURCE_GROUP="samo-brain-rg"
CLUSTER_NAME="samo-brain-cluster"
LOCATION="eastus"
NAMESPACE="samo-brain"

echo "🚀 Deploying SAMO Brain to Azure AKS..."

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create AKS cluster
echo "🏗️ Creating AKS cluster..."
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $CLUSTER_NAME \
    --node-count 3 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --node-vm-size Standard_D2s_v3

# Get credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Create namespace
kubectl create namespace $NAMESPACE

# Deploy application
echo "📦 Deploying application..."
kubectl apply -f k8s/azure/ -n $NAMESPACE

echo "✅ Deployment completed successfully!"
```

## 🔧 CI/CD Pipeline

### GitHub Actions Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy SAMO Brain

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/samo-brain-api \
          api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n samo-brain-staging
        kubectl rollout status deployment/samo-brain-api -n samo-brain-staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/samo-brain-api \
          api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -n samo-brain
        kubectl rollout status deployment/samo-brain-api -n samo-brain
```

### GitLab CI Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.12
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest tests/ --cov=src --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/samo-brain-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n samo-brain-staging
    - kubectl rollout status deployment/samo-brain-api -n samo-brain-staging
  environment:
    name: staging
  only:
    - main

deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/samo-brain-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n samo-brain
    - kubectl rollout status deployment/samo-brain-api -n samo-brain
  environment:
    name: production
  when: manual
  only:
    - main
```

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'samo-brain-api'
    static_configs:
      - targets: ['samo-brain-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

```json
// monitoring/grafana-dashboard.json
{
  "dashboard": {
    "id": null,
    "title": "SAMO Brain - API Metrics",
    "tags": ["samo-brain", "api"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(samo_brain_requests_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(samo_brain_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(samo_brain_requests_total{status_code=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: samo-brain-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(samo_brain_requests_total{status_code=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(samo_brain_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: ModelMemoryHigh
        expr: samo_brain_model_memory_bytes > 1073741824
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model memory usage is high"
          description: "Model is using {{ $value }} bytes of memory"
```

## 🔒 Security Configuration

### SSL/TLS Setup

```bash
#!/bin/bash
# setup-ssl.sh

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat << EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@samobrain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Network Policies

```yaml
# k8s/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: samo-brain-network-policy
  namespace: samo-brain
spec:
  podSelector:
    matchLabels:
      app: samo-brain-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: redis
    ports:
    - protocol: TCP
      port: 6379
```

## 🔄 Database Migration

### Alembic Migration Script

```python
# scripts/migrate.py
import os
import sys
from alembic import command
from alembic.config import Config

def run_migrations():
    """Run database migrations."""
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Create Alembic configuration
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "migrations")
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    
    try:
        # Run migration
        command.upgrade(alembic_cfg, "head")
        print("✅ Database migration completed successfully")
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migrations()
```

### Migration Commands

```bash
#!/bin/bash
# migrate.sh

echo "🔄 Running database migrations..."

# Run migrations
python scripts/migrate.py

# Verify migration
python scripts/verify_migration.py

echo "✅ Migration completed successfully!"
```

## 📈 Performance Optimization

### Resource Optimization

```yaml
# k8s/resource-limits.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: samo-brain-quota
  namespace: samo-brain
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "5"
    services: "10"
    services.loadbalancers: "2"

---
apiVersion: v1
kind: LimitRange
metadata:
  name: samo-brain-limits
  namespace: samo-brain
spec:
  limits:
  - default:
      cpu: 500m
      memory: 1Gi
    defaultRequest:
      cpu: 250m
      memory: 512Mi
    type: Container
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: samo-brain-hpa-advanced
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: samo-brain-api
  minReplicas: 3
  maxReplicas: 20
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
  - type: Object
    object:
      metric:
        name: requests-per-second
      describedObject:
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        name: samo-brain-ingress
      target:
        type: Value
        value: 1000
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

## 🚨 Disaster Recovery

### Backup Script

```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="samo-brain-backup-$DATE"

echo "🔄 Creating backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup database
echo "📊 Backing up database..."
pg_dump $DATABASE_URL > "$BACKUP_DIR/$BACKUP_NAME/database.sql"

# Backup Redis
echo "🔴 Backing up Redis..."
redis-cli --rdb "$BACKUP_DIR/$BACKUP_NAME/redis.rdb"

# Backup models
echo "🤖 Backing up models..."
cp -r /app/models "$BACKUP_DIR/$BACKUP_NAME/"

# Backup configuration
echo "⚙️ Backing up configuration..."
cp -r /app/config "$BACKUP_DIR/$BACKUP_NAME/"

# Create archive
tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"

# Clean up
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

echo "✅ Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
```

### Restore Script

```bash
#!/bin/bash
# restore.sh

set -e

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file>"
    exit 1
fi

echo "🔄 Restoring from backup: $BACKUP_FILE"

# Extract backup
BACKUP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$BACKUP_DIR"
BACKUP_NAME=$(ls "$BACKUP_DIR")

echo "📊 Restoring database..."
psql $DATABASE_URL < "$BACKUP_DIR/$BACKUP_NAME/database.sql"

echo "🔴 Restoring Redis..."
redis-cli flushall
redis-cli --pipe < "$BACKUP_DIR/$BACKUP_NAME/redis.rdb"

echo "🤖 Restoring models..."
cp -r "$BACKUP_DIR/$BACKUP_NAME/models"/* /app/models/

echo "⚙️ Restoring configuration..."
cp -r "$BACKUP_DIR/$BACKUP_NAME/config"/* /app/config/

# Clean up
rm -rf "$BACKUP_DIR"

echo "✅ Restore completed successfully!"
```

## 📋 Deployment Checklist

### Pre-Deployment Checklist

- [ ] **Environment Configuration**
  - [ ] All environment variables set
  - [ ] Database connection tested
  - [ ] Redis connection tested
  - [ ] Model files present

- [ ] **Security**
  - [ ] Secrets properly configured
  - [ ] SSL certificates installed
  - [ ] Network policies applied
  - [ ] RBAC configured

- [ ] **Monitoring**
  - [ ] Prometheus configured
  - [ ] Grafana dashboards deployed
  - [ ] Alerting rules configured
  - [ ] Log aggregation set up

- [ ] **Performance**
  - [ ] Resource limits configured
  - [ ] Autoscaling enabled
  - [ ] Load balancing configured
  - [ ] Caching strategy implemented

### Post-Deployment Checklist

- [ ] **Health Checks**
  - [ ] API health endpoint responding
  - [ ] Database connectivity verified
  - [ ] Redis connectivity verified
  - [ ] Model loading successful

- [ ] **Functionality**
  - [ ] Emotion prediction working
  - [ ] Batch processing working
  - [ ] Voice processing working
  - [ ] Rate limiting working

- [ ] **Monitoring**
  - [ ] Metrics being collected
  - [ ] Dashboards populated
  - [ ] Alerts configured
  - [ ] Logs flowing

- [ ] **Performance**
  - [ ] Response times acceptable
  - [ ] Throughput meeting requirements
  - [ ] Resource usage within limits
  - [ ] Autoscaling working

---

*This deployment guide provides comprehensive instructions for deploying SAMO Brain across different environments, from local development to production cloud deployments, with security, monitoring, and disaster recovery considerations.* 